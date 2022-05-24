import sys
sys.path.insert(1, '/home/arturo/workspace/pycharm_projects/data_loader_ml/DataLoaderML')
import wandb
import os
import json
import argparse
import time
import csv
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import pandas as pd
import data_loader_ml.tools.custom_transforms as CT
from data_loader_ml.dataset import YaakIterableDataset
from data_loader_ml.tools.utils import load_test_drive_ids_from_txt_file
from utils import tensor2array, save_checkpoint, count_parameters, print_batch, normalize_image
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from torch.utils.tensorboard import SummaryWriter

camera_view_choices = YaakIterableDataset.get_camera_view_choices() + ['all']

parser = argparse.ArgumentParser(
    description='Structure from Motion Learner -- Yaak Dataset 3.0 (Dynamic Scenes, Static Masks)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--dataset-path', dest='dataset_path', default=None, metavar='PATH',help='Path where the video sequences are stored in MP4 format (.mp4).')
parser.add_argument('--cam-calib-path', dest='cam_calib_path', default=None, metavar='PATH',help='Path where the camera calibration files are stored in JSON format (e.g., path/calib.json).')
parser.add_argument('--dataset-name', type=str, choices=['yaak'], default='yaak', help='Path to the dataset (i.e., video sequences).')
parser.add_argument('--camera-view-train', type=str, choices=camera_view_choices, default='cam_front_center', help='Selected camera view(s) for model training.')
parser.add_argument('--camera-view-val', type=str, choices=camera_view_choices, default='cam_front_center', help='Selected camera view(s) for model validation.')
parser.add_argument('--test-drive-id-train', type=str, default=None, help='Selected test drive ID(s) for model training.')
parser.add_argument('--test-drive-id-val', type=str, default=None, help='Selected test drive ID(s) for model validation.')
parser.add_argument('--frame-scaling-factor', type=float, choices=[0.25, 0.5, 0.75, 1.0], default=0.5, metavar='M', help='Scaling factor applied to the frames along x and y axes.')
parser.add_argument('--video-clip-length', type=int, metavar='N', help='Video clip length used for training the model.', default=3)
parser.add_argument('--video-clip-step', type=int, metavar='N', help='Video clip step used for training the model.', default=1)
parser.add_argument('--video-clip-fps', default=30, type=int, metavar='N', help='Video clip frames per second (fps).')
parser.add_argument('--oversampling-iterations-train', default=1, type=int, metavar='N', help='Number of iterations to over-sample each video in the training set.')
parser.add_argument('--oversampling-iterations-val', default=1, type=int, metavar='N', help='Number of iterations to over-sample each video in the validation set.')
parser.add_argument('--telemetry-data-params-train', default='{"minimum_speed": 8.0, "camera_view": "cam_rear", "return_telemetry": "True"}', type=str, help='Parameters used to load telemetry data from the JSON files (e.g., metadata.json). This data is used in the training stage.')
parser.add_argument('--telemetry-data-params-val', default='{"minimum_speed": 8.0, "camera_view": "cam_rear", "return_telemetry": "True"}', type=str, help='Parameters used to load telemetry data from the JSON files (e.g., metadata.json). This data is used in the validation stage.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Number of total epochs.')
parser.add_argument('--max-train-iterations', default=10, type=int, metavar='N', help='Maximum number of iterations in the training set per epoch.')
parser.add_argument('--max-val-iterations', default=10, type=int, metavar='N', help='Maximum number of iterations in the validation set.')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='Mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD, alpha parameter for Adam.')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='Beta parameters for Adam.')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='Weight decay.')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='Print frequency.')
parser.add_argument('--seed', default=0, type=int, help='Seed for random functions, and network initialization.')
parser.add_argument('--checkpoints-path', default='checkpoints', metavar='PATH', help='Path to store the checkpoints (e.g., models, tensorboard data, etc.).')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='CSV where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='CSV where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='If enabled, Dispnet outputs will be saved at each validation step.')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='Number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='The number of scales.', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='Weight for photometric loss.', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='Weight for disparity smoothness loss.', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-loss-weight', type=float, help='Weight for depth consistency loss.', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='With SSIM or not.')
parser.add_argument('--with-mask', type=int, default=1, help='With the the mask for moving objects and occlusions or not.')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='With the the mask for stationary points.')
parser.add_argument('--with-pretrain', type=int,  default=1, help='With or without imagenet pretrain for resnet.')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='Path to pre-trained Disparity model.')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='Path to pre-trained Pose model.')
parser.add_argument('--freeze-disp-encoder-parameters', action='store_true', help='If enabled, the encoder parameters of the disparity network will not be updated during training.')
parser.add_argument('--use-mask-static-objects-train', action='store_true', help='If enabled, a mask to select and suppress static objects will be used during training.')
parser.add_argument('--initial-model-val-iterations', type=int,  default=0, help='Number of iterations to evaluate the model on the validation set before training.')
parser.add_argument('--experiment-name', dest='experiment_name', type=str, required=True, help='Name of the experiment. Checkpoints are stored in checkpoints_path/experiment_name.')
parser.add_argument('--wandb-project-name', type=str, default=None, help='Weights & Biases project name. If set to None, weights & biases will be disabled.')
parser.add_argument('--rotation-matrix-mode', type=str, choices=['euler', 'quat'], default='euler', help='Rotation matrix representation.')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border', 'reflection'], default='zeros', help='Padding mode for '
    'image warping: this is important for photometric differenciation when going outside target image.'
    ' zeros will null gradients outside target image. border will only null gradients of the coordinate outside (x or y)'
)

best_error = -1
n_iter = 0
torch.autograd.set_detect_anomaly(True)


def main():

    ####################################################################################################################
    #
    # Initialization
    #
    ####################################################################################################################

    # Get the current device.
    device = \
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('[ Current device = {} ]'.format(torch.cuda.current_device()))
    print(' ')

    # Get the number of GPUs.
    num_gpus = torch.cuda.device_count()
    print('[ Number of GPUs available for training ] N = {}'.format(num_gpus))
    print(' ')

    # global best_error, n_iter, device
    global best_error, n_iter

    args = parser.parse_args()

    # Time instant.
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    # Experiment name
    experiment_name = '{}/frozen_disp_encoder_params'.format(args.experiment_name) \
        if args.freeze_disp_encoder_parameters else '{}/optim_all_params'.format(args.experiment_name)

    # Path to save data.
    args.save_path = '{}/{}/{}'.format(args.checkpoints_path, experiment_name, timestamp)

    print('[ Experimental results ] Save path: {}'.format(args.save_path))

    # If the path does not exist, it is created.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('\t- Creating directory: {}'.format(args.save_path))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Check the value of the arguments pretrained_disp and pretrained_pose.
    args.pretrained_disp = None if args.pretrained_disp == "None" else args.pretrained_disp
    args.pretrained_pose = None if args.pretrained_pose == "None" else args.pretrained_pose

    # ------------------------------------------------------------------------------------------------------------------
    # Save hyper-parameters in CSV file.
    # ------------------------------------------------------------------------------------------------------------------

    hparams_list = [
        ('device', device),
        ('dataset-path', args.dataset_path),
        ('cam-calib-path', args.cam_calib_path),
        ('dataset-name', args.dataset_name),
        ('camera-view-train', args.camera_view_train),
        ('camera-view-val', args.camera_view_val),
        ('test-drive-id-train', args.test_drive_id_train),
        ('test-drive-id-val', args.test_drive_id_val),
        ('frame-scaling-factor', args.frame_scaling_factor),
        ('video-clip-length', args.video_clip_length),
        ('video-clip-step', args.video_clip_step),
        ('video-clip-fps', args.video_clip_fps),
        ('oversampling-iterations-train', args.oversampling_iterations_train),
        ('oversampling-iterations-val', args.oversampling_iterations_val),
        ('telemetry-data-params-train', args.telemetry_data_params_train),
        ('telemetry-data-params-val', args.telemetry_data_params_val),
        ('workers', args.workers),
        ('epochs', args.epochs),
        ('max-train-iterations', args.max_train_iterations),
        ('max-val-iterations', args.max_val_iterations),
        ('batch-size', args.batch_size),
        ('learning-rate', args.lr),
        ('momentum', args.momentum),
        ('beta', args.beta),
        ('weight-decay', args.weight_decay),
        ('seed', args.seed),
        ('checkpoints-path', args.checkpoints_path),
        ('resnet-layers', args.resnet_layers),
        ('num-scales', args.num_scales),
        ('photo-loss-weight', args.photo_loss_weight),
        ('smooth-loss-weight', args.smooth_loss_weight),
        ('geometry-consistency-loss-weight', args.geometry_consistency_loss_weight),
        ('with-ssim', args.with_ssim),
        ('with-mask', args.with_mask),
        ('with-auto-mask', args.with_auto_mask),
        ('with-pretrain', args.with_pretrain),
        ('pretrained-disp', args.pretrained_disp),
        ('pretrained-pose', args.pretrained_pose),
        ('padding_model', args.padding_mode),
        ('experiment_name', experiment_name),
        ('wandb-project-name', args.wandb_project_name),
        ('initial-model-val-iterations', args.initial_model_val_iterations),
        ('freeze-disp-encoder-parameters', args.freeze_disp_encoder_parameters),
        ('use-mask-static-objects-train', args.use_mask_static_objects_train)
    ]

    # Splits the hyperparameter names and values in two lists.
    N_hp = len(hparams_list)
    hparams_names_list = [hparams_list[i][0] for i in range(N_hp)]
    hparams_values_list = [hparams_list[i][1] for i in range(N_hp)]

    # Create a data frame...
    hparams_df = pd.DataFrame(
        {
            'Name': hparams_names_list,
            'Value': hparams_values_list
        }
    )

    # Save the hyper_parameters in a CSV file...
    hparams_ffname = '{}/hyper_parameters.csv'.format(args.save_path)
    hparams_df.to_csv( hparams_ffname, index=False)

    print('\n[ Hyper-parameters ] Save in: {}'.format(hparams_ffname))

    # ------------------------------------------------------------------------------------------------------------------
    # Init Weights & Biases...
    # ------------------------------------------------------------------------------------------------------------------

    args.wandb_project_name = None if args.wandb_project_name == "None" else args.wandb_project_name

    # Start a W & B run, passing `sync_tensorboard=True`, to plot your TensorBoard files
    if args.wandb_project_name is not None:

        # If the path does not exist, it is created.
        wandb_path = "{:s}/{:s}".format(args.save_path, 'wandb')
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)

        # When using several event log directories,
        # please call `wandb.tensorboard.patch(root_logdir="...")` before `wandb.init`
        wandb.tensorboard.patch(root_logdir=args.save_path)

        wandb.init(
            dir=wandb_path,
            config=dict(hparams_list),
            project=args.wandb_project_name,
            sync_tensorboard=True
        )

        print('[ Initializing Weights & Biases ]')
        print('\t- Project name: {}'.format(args.wandb_project_name))
        print('\t- Path: {}'.format(wandb_path))
        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Summary writers...
    # ------------------------------------------------------------------------------------------------------------------

    training_writer = SummaryWriter(args.save_path, flush_secs=10, max_queue=100)

    num_output_writers = 1
    output_writers = []
    if args.log_output:
        for i in range(num_output_writers):
            output_writers.append(SummaryWriter('{}/{}/{}'.format(args.save_path, 'valid', str(i))))

    # ------------------------------------------------------------------------------------------------------------------
    # Frame target size, crop size, and scaling factor.
    # ------------------------------------------------------------------------------------------------------------------

    # Frame target size.
    frame_target_size = (1080, 1920)

    # Frame scaling factor.
    frame_scaling_factor = args.frame_scaling_factor

    # Frame crop size.
    # frame_crop_size = {
    #     '0.25': (256, -1),
    #     '0.50': (512, -1),
    #     '0.75': (768, -1),
    #     '1.00': (-1, -1),
    # }

    # Offsets and scales to resize and crop each frame.
    frame_scale_offset_dict = {
        # Frame resolution at 25%: 256x480
        '0.25': {'scale_factor': 0.27, 'offset': (2, 33, 19, 19)},
        # Frame resolution at 50%: 512x960
        '0.50': {'scale_factor': 0.539, 'offset': (5, 65, 37, 37)},
        # Frame resolution at 75%: 768x1440
        '0.75': {'scale_factor': 0.809, 'offset': (8, 97, 56, 57)},
        # Frame resolution at 100%: 1024x1920
        '1.00': {'scale_factor': 1.079, 'offset': (10, 131, 75, 76)},
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations applied on data.
    # ------------------------------------------------------------------------------------------------------------------

    # Means used for data normalization.
    mean_per_channel = [0.45, 0.45, 0.45]

    # Standard deviations used for data normalization.
    std_per_channel = [0.225, 0.225, 0.225]

    # Transformations applied on training data.
    train_transform = CT.Compose([
        CT.ScaleCustomCrop(
            offset=frame_scale_offset_dict['{:0.2f}'.format(frame_scaling_factor)]['offset'],
            scaling_factor=frame_scale_offset_dict['{:0.2f}'.format(frame_scaling_factor)]['scale_factor'],
        ),
        CT.RandomHorizontalFlip(),
        CT.RandomScaleCrop(),
        CT.ToFloat32TensorCHW(),
        CT.Normalize(mean=mean_per_channel, std=std_per_channel)
    ])

    # Transformations applied on validation data.
    val_transform = CT.Compose([
        CT.ScaleCustomCrop(
            offset=frame_scale_offset_dict['{:0.2f}'.format(frame_scaling_factor)]['offset'],
            scaling_factor=frame_scale_offset_dict['{:0.2f}'.format(frame_scaling_factor)]['scale_factor'],
        ),
        CT.ToFloat32TensorCHW(),
        CT.Normalize(mean=mean_per_channel, std=std_per_channel)
    ])

    # ------------------------------------------------------------------------------------------------------------------
    # Create train/validation datasets.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating train and validation datasets ]")
    print("\t- Dataset name: {}".format(args.dataset_name))
    print("\t- Video sequences path: {}".format(args.dataset_path))
    print("\t- Camera calibration path: {}".format(args.cam_calib_path))

    train_dataset, val_dataset = None, None

    if args.dataset_name == 'yaak':

        # --------------------------------------------------------------------------------------------------------------
        # Parameters to be used to load telemetry data from the JSON files.
        # --------------------------------------------------------------------------------------------------------------

        # Parameters to be used to load telemetry data in the training stage.
        tele_params_train_dict = json.loads(args.telemetry_data_params_train)

        # Parameters to be used to load telemetry data in the validation stage.
        tele_params_val_dict = json.loads(args.telemetry_data_params_val)

        # --------------------------------------------------------------------------------------------------------------
        # Convert the value corresponding to the key 'return_telemetry' to boolean...
        # --------------------------------------------------------------------------------------------------------------

        assert 'return_telemetry' in tele_params_train_dict.keys(), \
            '[ Error ] return_telemetry key is missing in tele_params_train_dict'

        assert 'return_telemetry' in tele_params_val_dict.keys(), \
            '[ Error ] return_telemetry key is missing in tele_params_val_dict'

        tele_params_train_dict['return_telemetry'] = \
            True if tele_params_train_dict['return_telemetry'] == "True" else False

        tele_params_val_dict['return_telemetry'] = \
            True if tele_params_val_dict['return_telemetry'] == "True" else False

        # --------------------------------------------------------------------------------------------------------------
        # Show the contents to the telemetry data parameters dicts...
        # --------------------------------------------------------------------------------------------------------------

        print('\t- Parameters used to load telemetry data (training stage): ')
        for k, v in tele_params_train_dict.items():
            print('\t\t[ {} ] {}'.format(k, v))

        print('\t- Parameters used to load telemetry data (validation stage):')
        for k, v in tele_params_val_dict.items():
            print('\t\t[ {} ] {}'.format(k, v))

        # --------------------------------------------------------------------------------------------------------------
        # Create dictionaries of telemetry parameters to tbe passed to the YaakIterableDataset class.
        # --------------------------------------------------------------------------------------------------------------

        telemetry_data_params_dict = {
            'train': (
                ('minimum_speed', float(tele_params_train_dict['minimum_speed'])),
                ('camera_view', str(tele_params_train_dict['camera_view']))
            ),
            'val': (
                ('minimum_speed', float(tele_params_val_dict['minimum_speed'])),
                ('camera_view', str(tele_params_val_dict['camera_view'])),
            ),
        }

        # --------------------------------------------------------------------------------------------------------------
        # Define which data should be returned by the YaakIterableDataset class.
        # --------------------------------------------------------------------------------------------------------------

        if tele_params_train_dict['return_telemetry']:
            print('\t- Telemetry data will be returned in the training stage.')

        if tele_params_val_dict['return_telemetry']:
            print('\t- Telemetry data will be returned in the validation stage.')

        return_data_dict = {
            'train': (
                ('processed_videos', True),
                ('tgt_ref_frames', True),
                ('telemetry', tele_params_train_dict['return_telemetry']),
                ('camera_intrinsics', True),
                ('camera_distortion', False),
                ('mask_static_objects', args.use_mask_static_objects_train),
            ),
            'val': (
                ('processed_videos', True),
                ('tgt_ref_frames', True),
                ('telemetry', tele_params_val_dict['return_telemetry']),
                ('camera_intrinsics', True),
                ('camera_distortion', False),
                ('mask_static_objects', False),
            ),
        }

        # --------------------------------------------------------------------------------------------------------------
        # Define the camera view for model training and validation.
        # --------------------------------------------------------------------------------------------------------------

        camera_view_dict = {
            'train': args.camera_view_train,
            'val': args.camera_view_val,
        }

        # --------------------------------------------------------------------------------------------------------------
        # Define test drive ids for model training and validation.
        # --------------------------------------------------------------------------------------------------------------

        test_drive_id_train, test_drive_id_val = None, None

        # Test drive IDs: Training set.
        if args.test_drive_id_train.endswith(".txt"):
            print('[ Loading test drive IDs from a TXT file ] Training set: \n')
            test_drive_id_train = load_test_drive_ids_from_txt_file(fname=args.test_drive_id_train, verbose=True)
        else:
            test_drive_id_train = None if args.test_drive_id_train == "None" else args.test_drive_id_train

        # Test drive IDs: Validation set.
        if args.test_drive_id_val.endswith(".txt"):
            print('[ Loading test drive IDs from a TXT file ] Validation set: \n')
            test_drive_id_val = load_test_drive_ids_from_txt_file(fname=args.test_drive_id_val, verbose=True)
        else:
            test_drive_id_val = None if args.test_drive_id_val == "None" else args.test_drive_id_val

        # Dictionary of test drive IDs.
        test_drive_id_dict = {
            'train': test_drive_id_train,
            'val': test_drive_id_val,
        }

        # --------------------------------------------------------------------------------------------------------------
        # Create train dataset.
        # --------------------------------------------------------------------------------------------------------------

        print('\t- Creating the train dataset using the YaakIterableDataset class.')
        print('\t\t- Every video in the training set will be over-sampled by N = {} iterations.'.format(
                args.oversampling_iterations_train
            )
        )
        print(' ')

        train_dataset = YaakIterableDataset(
            dataset_path=args.dataset_path,
            cam_calib_path=args.cam_calib_path,
            test_drive_id=test_drive_id_dict['train'],
            camera_view=camera_view_dict['train'],
            video_extension='mp4',
            telemetry_filename='metadata.json',
            video_clip_length=args.video_clip_length,
            video_clip_step=args.video_clip_step,
            video_clip_memory_format='default',
            video_clip_output_dtype='default',
            video_clip_fps=args.video_clip_fps,
            frame_target_size=frame_target_size,
            processed_videos_filename_suffix='-force-key.defish',
            telemetry_data_params=telemetry_data_params_dict['train'],
            oversampling_iterations=args.oversampling_iterations_train,
            return_data=return_data_dict['train'],
            transform=train_transform,
            device_id=torch.cuda.current_device(),
            device_name='gpu' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )

        # --------------------------------------------------------------------------------------------------------------
        # Create validation dataset.
        # --------------------------------------------------------------------------------------------------------------
        print(' ')
        print('\t- Creating the validation dataset using the YaakIterableDataset class.')
        print('\t\t- Every video in the validation set will be over-sampled by N = {} iterations.'.format(
                args.oversampling_iterations_val
            )
        )
        print(' ')

        val_dataset = YaakIterableDataset(
            dataset_path=args.dataset_path,
            cam_calib_path=args.cam_calib_path,
            test_drive_id=test_drive_id_dict['val'],
            camera_view=camera_view_dict['val'],
            video_extension='mp4',
            telemetry_filename='metadata.json',
            video_clip_length=args.video_clip_length,
            video_clip_step=args.video_clip_step,
            video_clip_memory_format='default',
            video_clip_output_dtype='default',
            video_clip_fps=args.video_clip_fps,
            frame_target_size=frame_target_size,
            processed_videos_filename_suffix='-force-key.defish',
            telemetry_data_params=telemetry_data_params_dict['val'],
            oversampling_iterations=args.oversampling_iterations_val,
            return_data=return_data_dict['val'],
            transform=val_transform,
            device_id=torch.cuda.current_device(),
            device_name='gpu' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )

        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Create train/validation data loaders.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating data loaders ]")
    print("\t- Creating train data loader.")

    # Train loader.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    print("\t- Creating validation data loader.")

    # Validation loader.
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Create models: Disparity and Pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    if args.with_pretrain:
        print('\n[ The DispResNet/PoseResNet encoder network will be initialized with pretrained weights (ImageNet) ]')

    # Disparity network.
    disp_net = \
        models.DispResNet(num_layers=args.resnet_layers, pretrained=args.with_pretrain, verbose=False).to(device)

    # Pose network.
    pose_net = models.PoseResNet(num_layers=18, pretrained=args.with_pretrain).to(device)

    print("\n[ Creating models ] Disparity (DispResNet) and Pose (PoseResNet) networks")
    print("\t- DispResNet (num_layers = {}, pretrain = {}) | Device = {}".format(
            args.resnet_layers,
            args.with_pretrain,
            device
        )
    )
    print("\t- PoseResNet (num_layers = 18, pretrain = {}) | Device = {}".format(
            args.with_pretrain,
            device
        )
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize DispResNet/PoseResNet networks with pre-trained weights stored on disk...
    # ------------------------------------------------------------------------------------------------------------------

    if args.pretrained_disp:
        print('\n[ Initializing DispResNet network with pretrained weights stored on disk ]')
        print("\t- File: {}".format(args.pretrained_disp))
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
        print("\t- The model parameters have been updated.")

    # Load pre-trained pose network parameters.
    if args.pretrained_pose:
        print('\n[ Initializing PoseResNet network with pretrained weights stored on disk ]')
        print("\t- File: {}".format(args.pretrained_pose))
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
        print("\t- The model parameters have been updated.")

    # ------------------------------------------------------------------------------------------------------------------
    # Count the number of parameters of the disparity and pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    # Disparity network parameters expressed in M.
    disp_net_num_params = count_parameters(disp_net) / 1e6

    # Pose network parameters in M.
    pose_net_num_params = count_parameters(pose_net) / 1e6

    print('\n[ Count model parameters ]')
    print('\t- DispResNet | N = {:0.2f}M params'.format(disp_net_num_params))
    print('\t- PoseResNet | N = {:0.2f}M params'.format(pose_net_num_params))

    # Adding the disparity network parameter count to tensorboard.
    training_writer.add_text(
        tag='Parameter_count/disp_resnet',
        text_string='{:0.2f}M'.format(disp_net_num_params),
        global_step=0
    )

    # Adding the pose network parameter count to tensorboard.
    training_writer.add_text(
        tag='Parameter_count/pose_resnet',
        text_string='{:0.2f}M'.format(pose_net_num_params),
        global_step=0
    )

    training_writer.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Freezing the encoder parameters of the disparity network...
    # ------------------------------------------------------------------------------------------------------------------

    if args.freeze_disp_encoder_parameters:

        print('\n[ Freezing the encoder parameters of the disparity network ]')

        for param in disp_net.encoder.parameters():
            param.requires_grad = False

        # Disparity network parameters expressed in M.
        disp_net_num_params = count_parameters(disp_net) / 1e6

        print('\t- DispResNet | N = {:0.2f}M params'.format(disp_net_num_params))

        # Adding the disparity network parameter count to tensorboard.
        training_writer.add_text(
            tag='Parameter_count/disp_resnet_with_frozen_enc',
            text_string='{:0.2f}M'.format(disp_net_num_params),
            global_step=0
        )

        training_writer.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Wrapping the models with torch.nn.DataParallel...
    # ------------------------------------------------------------------------------------------------------------------

    disp_net = torch.nn.DataParallel(disp_net, output_device=0)
    pose_net = torch.nn.DataParallel(pose_net, output_device=0)

    disp_net = disp_net.to(device)
    pose_net = pose_net.to(device)

    print('\n[ Data parallelism with torch.nn.DataParallel ]')
    print('\t- [ Disparity network ] Device IDs = {} | Output device = {}'.format(
            disp_net.device_ids,
            disp_net.output_device
        )
    )
    print('\t- [ Pose network ] Device IDs = {} | Output device = {}'.format(
            pose_net.device_ids,
            pose_net.output_device
        )
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Create optimizers for the Disparity and Pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Creating optimizers for the Disparity and Pose networks ] Optimizer = Adam')

    # Selecting parameters to be optimized.
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]

    # Optimizer...
    optimizer = torch.optim.Adam(
        optim_params,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Creating a CSV file to log data.
    # ------------------------------------------------------------------------------------------------------------------

    # Log summary file.
    log_summary_ffname = '{}/{}'.format(args.save_path, args.log_summary)

    # Log full file.
    log_full_ffname = '{}/{}'.format(args.save_path, args.log_full)

    print('\n[ Creating a CSV file to log data ]')
    print('\t- Log summary | File: {}'.format(log_summary_ffname))
    print('\t- Log full | File: {}'.format(log_full_ffname))

    with open(log_summary_ffname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(log_full_ffname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photometric_loss', 'disparity_smoothness_loss', 'geometry_consistency_loss'])

    ####################################################################################################################
    #
    # Initial model evaluation...
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the model on the validation set before training...
    # ------------------------------------------------------------------------------------------------------------------

    if args.initial_model_val_iterations > 0:

        print('\n[ Evaluating the model on the validation set before training... ]')
        print('[ Creating a logger for initial model evaluation) ]')

        logger_init = TermLogger(
            n_epochs=args.epochs,
            train_size=0,
            valid_size=args.initial_model_val_iterations,
        )

        logger_init.reset_valid_bar()
        logger_init.valid_bar.update(0)

        for val_index in range(args.initial_model_val_iterations):

            errors, error_names = \
                validate_without_gt(
                    args=args,
                    val_loader=val_loader,
                    disp_net=disp_net,
                    pose_net=pose_net,
                    epoch=val_index,
                    max_iterations=1,
                    logger=logger_init,
                    train_writer=training_writer,
                    output_writers=output_writers,
                    return_telemetry=tele_params_val_dict['return_telemetry'],
                    show_progress_bar=True,
                    initial_model_evaluation=True,
                    device=device,
                    verbose=False,
                )

            logger_init.valid_bar.update(val_index+1)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

            logger_init.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, val_index)

        logger_init.valid_bar.update(args.initial_model_val_iterations)

    ####################################################################################################################
    #
    # Training stage...
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Creating a Logger.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Start model training ] Epochs = {}'.format(args.epochs))
    print('[ Creating a Logger ]')

    logger = TermLogger(
        n_epochs=args.epochs,
        train_size=args.max_train_iterations,
        valid_size=args.max_val_iterations,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Start model training...
    # ------------------------------------------------------------------------------------------------------------------

    init_epoch=0

    logger.epoch_bar.start()

    for epoch in range(init_epoch, args.epochs+init_epoch):

        logger.epoch_bar.update(epoch)

        # --------------------------------------------------------------------------------------------------------------
        # Train the model, on the training set, for N = max_train_iterations iterations.
        # --------------------------------------------------------------------------------------------------------------

        logger.reset_train_bar()

        train_loss = train(
            args=args,
            train_loader=train_loader,
            disp_net=disp_net,
            pose_net=pose_net,
            optimizer=optimizer,
            epoch=epoch,
            max_iterations=args.max_train_iterations,
            logger=logger,
            train_writer=training_writer,
            return_telemetry=tele_params_train_dict['return_telemetry'],
            show_progress_bar=True,
            device=device,
            verbose=(False, False)
        )

        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate the model, on the validation set, for N = max_val_iterations iterations.
        # --------------------------------------------------------------------------------------------------------------

        logger.reset_valid_bar()

        errors, error_names = \
            validate_without_gt(
                args=args,
                val_loader=val_loader,
                disp_net=disp_net,
                pose_net=pose_net,
                epoch=epoch,
                max_iterations=args.max_val_iterations,
                logger=logger,
                train_writer=training_writer,
                output_writers=output_writers,
                return_telemetry=tele_params_val_dict['return_telemetry'],
                show_progress_bar=True,
                initial_model_evaluation=False,
                device=device,
                verbose=False,
            )

        error_string = \
            ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # --------------------------------------------------------------------------------------------------------------
        # Up to you to chose the most relevant error to measure your model's performance,
        # careful some measures are to maximize (such as a1,a2,a3)
        # --------------------------------------------------------------------------------------------------------------

        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # --------------------------------------------------------------------------------------------------------------
        # Remember lowest error and save checkpoint.
        # --------------------------------------------------------------------------------------------------------------

        is_best = decisive_error < best_error

        best_error = min(best_error, decisive_error)

        save_checkpoint(
            args.save_path,
            {'epoch': epoch + 1, 'state_dict': disp_net.module.state_dict()},
            {'epoch': epoch + 1, 'state_dict': pose_net.module.state_dict()},
            is_best
        )

        # --------------------------------------------------------------------------------------------------------------
        # Update the CSV file (log summary).
        # --------------------------------------------------------------------------------------------------------------

        with open(log_summary_ffname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])

        # --------------------------------------------------------------------------------------------------------------
        # Flushes the event file to disk.
        # --------------------------------------------------------------------------------------------------------------

        training_writer.flush()

        for ow in output_writers:
            ow.flush()

    logger.epoch_bar.finish()

    # ------------------------------------------------------------------------------------------------------------------
    # Closing operations...
    # ------------------------------------------------------------------------------------------------------------------

    training_writer.close()

    for ow in output_writers:
        ow.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Finish the wandb run to upload the TensorBoard logs to W & B.
    # ------------------------------------------------------------------------------------------------------------------

    if args.wandb_project_name is not None:
        wandb.finish()


def train(
    args,
    train_loader,
    disp_net,
    pose_net,
    optimizer,
    epoch,
    max_iterations,
    logger,
    train_writer,
    return_telemetry=False,
    show_progress_bar=False,
    device=torch.device("cpu"),
    verbose=(False, False)
):

    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

    # global n_iter, device
    global n_iter

    batch_time = AverageMeter()

    data_time = AverageMeter()

    losses = AverageMeter(precision=4)

    end = time.time()

    if show_progress_bar:
        logger.train_bar.update(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize total training losses.
    # ------------------------------------------------------------------------------------------------------------------

    total_loss = 0.0
    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_loss_3 = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # Pose variable names...
    # ------------------------------------------------------------------------------------------------------------------

    pose_var_names_dict = {
        0: 'x',
        1: 'y',
        2: 'z',
        3: 'theta_x',
        4: 'theta_y',
        5: 'theta_z'
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Training loss coefficients.
    # ------------------------------------------------------------------------------------------------------------------

    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_loss_weight

    # ------------------------------------------------------------------------------------------------------------------
    # Set the models to "training mode".
    # ------------------------------------------------------------------------------------------------------------------

    disp_net.train()
    pose_net.train()

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over batches of data using the train data-loader...
    #   Deprecated:
    #       for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
    # ------------------------------------------------------------------------------------------------------------------

    i = 0

    for batch in train_loader:

        log_losses = i > 0 and n_iter % args.print_freq == 0

        # Measure data loading time.
        data_time.update(time.time() - end)

        # --------------------------------------------------------------------------------------------------------------
        # Get a batch of data: Target image, reference images, and camera intrinsics.
        # --------------------------------------------------------------------------------------------------------------

        # 1) Target image
        #   Deprecated:
        #       tgt_img = tgt_img.to(device)
        #
        tgt_img = batch['target_frame'].to(device)

        # 2) Reference images.
        #   Deprecated:
        #       ref_imgs = [img.to(device) for img in ref_imgs]
        #
        ref_imgs = [batch[key].to(device) for key in ['reference_frame:0', 'reference_frame:1']]

        # 3) Camera intrinsics.
        #   Deprecated:
        #       intrinsics = intrinsics.to(device)
        #
        intrinsics = batch['camera_intrinsics'].to(device)

        # 4) Telemetry data: speed
        #
        speed_data = None
        if return_telemetry:
            speed_data = batch['telemetry_data/speed']

        # 5) Camera view, test drive ID, video clip indices.
        #
        camera_view = batch['camera_view']
        test_drive_id = batch['test_drive_id']
        video_clip_indices = batch['video_clip_indices'].tolist()

        # --------------------------------------------------------------------------------------------------------------
        # Print essential information of the batch of data.
        # --------------------------------------------------------------------------------------------------------------

        if verbose[0]:
            print_batch(batch_index=i, target_image=tgt_img, reference_images=ref_imgs, intrinsics=intrinsics)

        # --------------------------------------------------------------------------------------------------------------
        # Forward pass using the depth and pose networks.
        # --------------------------------------------------------------------------------------------------------------

        # Estimate depth using the neural network model.
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, verbose=verbose[1])

        # Estimate the camera pose using the neural network model.
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs, verbose=verbose[1])

        # --------------------------------------------------------------------------------------------------------------
        # Compute losses.
        # --------------------------------------------------------------------------------------------------------------

        loss_1, loss_3 = \
            compute_photo_and_geometry_loss(
                tgt_img=tgt_img,
                ref_imgs=ref_imgs,
                intrinsics=intrinsics,
                tgt_depth=tgt_depth,
                ref_depths=ref_depths,
                poses=poses,
                poses_inv=poses_inv,
                max_scales=args.num_scales,
                with_ssim=args.with_ssim,
                with_mask=args.with_mask,
                with_auto_mask=args.with_auto_mask,
                padding_mode=args.padding_mode,
                rotation_mode=args.rotation_matrix_mode,
                writer_obj_tag='Train',
                writer_obj_step=epoch,
                writer_obj=train_writer if i + 1 == max_iterations else None,
                device=device
            )

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        # --------------------------------------------------------------------------------------------------------------
        # Log losses per iteration...
        # --------------------------------------------------------------------------------------------------------------

        if log_losses:

            train_writer.add_scalar(
                tag='Train_loss_per_iter/photometric_loss', scalar_value=loss_1.item(), global_step=n_iter
            )
            train_writer.add_scalar(
                tag='Train_loss_per_iter/disparity_smoothness_loss', scalar_value=loss_2.item(), global_step=n_iter
            )
            train_writer.add_scalar(
                tag='Train_loss_per_iter/geometry_consistency_loss', scalar_value=loss_3.item(), global_step=n_iter
            )
            train_writer.add_scalar(
                tag='Train_loss_per_iter/total_loss', scalar_value=loss.item(), global_step=n_iter
            )

            # Flushes the event file to disk.
            train_writer.flush()

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # --------------------------------------------------------------------------------------------------------------
        # Compute the gradients and perform a single gradient descent step.
        # --------------------------------------------------------------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------------------------------------------------------
        # Measure elapsed time
        # --------------------------------------------------------------------------------------------------------------

        batch_time.update(time.time() - end)
        end = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # Update logs...
        # --------------------------------------------------------------------------------------------------------------

        with open('{}/{}'.format(args.save_path, args.log_full), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])

        if show_progress_bar:
            logger.train_bar.update(i+1)

        if (i+1) % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

        # --------------------------------------------------------------------------------------------------------------
        # Log poses.
        # --------------------------------------------------------------------------------------------------------------

        log_poses = i + 1 == max_iterations

        if log_poses:

            # Loop over pose data.
            for img_idx, pose_data in enumerate(poses):

                pose_vector_str = ""

                # Loop over the dimensions of the pose vector.
                for pose_idx in range(pose_data.size()[1]):

                    # Pose variable value.
                    pose_var_value = pose_data[0, pose_idx].item()

                    # Pose variable name.
                    pose_var_name = pose_var_names_dict[pose_idx]

                    # Show the components of the 6D pose vector in a plot.
                    train_writer.add_scalar(
                        tag='Train_pose_target_wrt_ref{:d}/{:s}'.format(img_idx, pose_var_name),
                        scalar_value=pose_var_value,
                        global_step=epoch
                    )

                    if pose_idx <= 4:
                        pose_vector_str += "{:s} = {:0.4f}, ".format(pose_var_name, pose_var_value)
                    else:
                        pose_vector_str += "{:s} = {:0.4f}".format(pose_var_name, pose_var_value)

                # Show the components of the 6D pose vector as a string.
                train_writer.add_text(
                    tag='Train_pose_target_wrt_ref{:d}/pose_vector_6d'.format(img_idx),
                    text_string=pose_vector_str,
                    global_step=epoch
                )

            # Flushes the event file to disk.
            train_writer.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Compute average losses per epoch...
        # --------------------------------------------------------------------------------------------------------------

        total_loss += loss.item() / max_iterations
        total_loss_1 += loss_1.item() / max_iterations
        total_loss_2 += loss_2.item() / max_iterations
        total_loss_3 += loss_3.item() / max_iterations

        # --------------------------------------------------------------------------------------------------------------
        # Log losses, images, and estimated depth maps into Tensorboard every epoch. Only the data corresponding
        # to the last training iteration: (i + 1 == max_iterations).
        # --------------------------------------------------------------------------------------------------------------

        if i + 1 == max_iterations:

            # ----------------------------------------------------------------------------------------------------------
            # 1) Intrinsic matrix.
            # ----------------------------------------------------------------------------------------------------------

            # Batch sample idx...
            sample_idx = 0

            # Intrinsic matrix elements: Row 0.
            k00 = intrinsics[sample_idx, 0, 0]
            k01 = intrinsics[sample_idx, 0, 1]
            k02 = intrinsics[sample_idx, 0, 2]

            # Intrinsic matrix elements: Row 1.
            k10 = intrinsics[sample_idx, 1, 0]
            k11 = intrinsics[sample_idx, 1, 1]
            k12 = intrinsics[sample_idx, 1, 2]

            # Intrinsic matrix elements: Row 2.
            k20 = intrinsics[sample_idx, 2, 0]
            k21 = intrinsics[sample_idx, 2, 1]
            k22 = intrinsics[sample_idx, 2, 2]

            # String representation of the intrinsic matrix.
            k_matrix_str = \
                'K = {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f}'.format(
                    k00, k01, k02,
                    k10, k11, k12,
                    k20, k21, k22,
                )

            train_writer.add_text(
                tag='Intrinsic_matrix/train',
                text_string=k_matrix_str,
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 2) Camera view, test drive, and video clip indices...
            # ----------------------------------------------------------------------------------------------------------

            video_clip_time = (
                np.asarray(video_clip_indices) / float(args.video_clip_fps)
            ).tolist()

            video_clip_time_str = ['{:0.2f}'.format(t) for t in video_clip_time[0]]

            video_clip_info = \
                'test_drive_id = {} | camera_view = {} | video_clip_indices = {} | video_clip_time = {}'.format(
                    test_drive_id[0],
                    camera_view[0],
                    video_clip_indices[0],
                    video_clip_time_str
                )

            train_writer.add_text(tag='Video_clip_info/train', text_string=video_clip_info, global_step=epoch)

            # ----------------------------------------------------------------------------------------------------------
            # 3) Speed data.
            # ----------------------------------------------------------------------------------------------------------

            if speed_data is not None and return_telemetry:
                train_writer.add_scalar(
                    tag='Telemetry_data/train_speed', scalar_value=speed_data[0], global_step=epoch
                )

            # ----------------------------------------------------------------------------------------------------------
            # 4) Training losses...
            # ----------------------------------------------------------------------------------------------------------

            train_writer.add_scalar(
                tag='Train_loss/photometric_loss', scalar_value=total_loss_1, global_step=epoch
            )
            train_writer.add_scalar(
                tag='Train_loss/disparity_smoothness_loss', scalar_value=total_loss_2, global_step=epoch
            )
            train_writer.add_scalar(
                tag='Train_loss/geometry_consistency_loss', scalar_value=total_loss_3, global_step=epoch
            )
            train_writer.add_scalar(
                tag='Train_loss/total_loss', scalar_value=total_loss, global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 5) Image differences...
            # ----------------------------------------------------------------------------------------------------------

            # Image difference loss: | ref_image_0 - target_image |
            image_diff_loss_0 = (ref_imgs[0][0] - tgt_img[0]).abs()

            # Image difference loss: | ref_image_1 - target_image |
            image_diff_loss_1 = (ref_imgs[1][0] - tgt_img[0]).abs()

            # Image data...
            train_writer.add_image(
                tag='Train_image_difference_map/ref0_target',
                img_tensor=tensor2array(image_diff_loss_0, max_value=None, colormap='magma'),
                global_step=epoch
            )

            train_writer.add_image(
                tag='Train_image_difference_map/ref1_target',
                img_tensor=tensor2array(image_diff_loss_1, max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Losses...
            train_writer.add_scalar(
                tag='Train_image_difference_loss/ref0_target',
                scalar_value=image_diff_loss_0.sum(),
                global_step=epoch
            )

            train_writer.add_scalar(
                tag='Train_image_difference_loss/ref1_target',
                scalar_value=image_diff_loss_1.sum(),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 6) Input image data...
            # ----------------------------------------------------------------------------------------------------------

            # Reference and target frames.
            train_writer.add_image(
                tag='Train_model_input/ref0_target_ref1',
                img_tensor=tensor2array(torch.cat([ref_imgs[0][0], tgt_img[0], ref_imgs[1][0]], dim=2)),
                global_step=epoch
            )

            # Target frame.
            train_writer.add_image(
                tag='Train_model_output/input_target_frame',
                img_tensor=tensor2array(tgt_img[0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 7. Depth and disparity.
            #
            #   - alpha = 100, beta = 0.01
            #   - Disparity:  D(x) = (alpha * x + beta)
            #   - Depth:      Z(x) = 1. / D(x)
            #
            # ----------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------
            # 7.1. Disparity.
            # ----------------------------------------------------------------------------------------------------------

            # Estimated disparity map: Normalized w.r.t. maximum value.
            train_writer.add_image(
                tag='Train_model_output/disparity_normalized_wrt_max',
                img_tensor=tensor2array(1. / tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Estimated disparity map: Normalized to the range 0 to 1.
            train_writer.add_image(
                tag='Train_model_output/disparity_normalized_0_to_1',
                img_tensor=normalize_image(1. / tgt_depth[0][0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 7.2. Depth.
            # ----------------------------------------------------------------------------------------------------------

            # Estimated depth map: Normalized w.r.t. maximum value.
            train_writer.add_image(
                tag='Train_model_output/depth_normalized_wrt_max',
                img_tensor=tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Estimated depth map: Normalized w.r.t. maximum value = 30.
            beta_value = 10
            train_writer.add_image(
                tag='Train_model_output/depth_normalized_wrt_max_{:d}'.format(beta_value),
                img_tensor=tensor2array(tgt_depth[0][0], max_value=beta_value, colormap='magma'),
                global_step=epoch
            )

            # Estimated depth map: Normalized to the range 0 to 1.
            train_writer.add_image(
                tag='Train_model_output/depth_normalized_0_to_1',
                img_tensor=normalize_image(tgt_depth[0][0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 8) Histograms...
            # ----------------------------------------------------------------------------------------------------------

            # Histogram of the estimated disparity map.
            train_writer.add_histogram(
                tag='Train_histograms/disparity',
                values=1./tgt_depth[0][0],
                global_step=epoch
            )

            # Histogram of the estimated depth map.
            train_writer.add_histogram(
                tag='Train_histograms/depth',
                values=tgt_depth[0][0],
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # Flushes the event file to disk.
            # ----------------------------------------------------------------------------------------------------------

            train_writer.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Condition to break the loop:
        #   Do max iterations have been reached?
        # --------------------------------------------------------------------------------------------------------------

        if max_iterations > 0:
            if i + 1 >= max_iterations:
                break

        # --------------------------------------------------------------------------------------------------------------
        # Increase the counters.
        # --------------------------------------------------------------------------------------------------------------

        # Local iterations.
        i += 1

        # Global iterations.
        n_iter += 1

        # --------------------------------------------------------------------------------------------------------------
        # Flushes the event file to disk.
        # --------------------------------------------------------------------------------------------------------------

        train_writer.flush()

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(
    args,
    val_loader,
    disp_net,
    pose_net,
    epoch,
    max_iterations,
    logger,
    train_writer,
    output_writers=[],
    return_telemetry=False,
    show_progress_bar=False,
    initial_model_evaluation=False,
    device=torch.device("cpu"),
    verbose=False,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

    # Prefix used to log data in tensorboard...
    writer_prefix_tag = 'Val_init' if initial_model_evaluation else 'Val'

    # global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # Set the networks in evaluation mode.
    disp_net.eval()
    pose_net.eval()

    # End time.
    end = time.time()

    # Initialize progress bar.
    if show_progress_bar and not initial_model_evaluation:
        logger.valid_bar.update(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Pose variable names...
    # ------------------------------------------------------------------------------------------------------------------

    pose_var_names_dict = {
        0: 'x',
        1: 'y',
        2: 'z',
        3: 'theta_x',
        4: 'theta_y',
        5: 'theta_z'
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over batches of data using the validation data-loader...
    #   Deprecated:
    #       for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
    # ------------------------------------------------------------------------------------------------------------------

    i = 0

    for batch in val_loader:

        # --------------------------------------------------------------------------------------------------------------
        # Get a batch of data: Target image, reference images, and camera intrinsics.
        # --------------------------------------------------------------------------------------------------------------

        # 1) Target image
        #   Deprecated:
        #       tgt_img = tgt_img.to(device)
        #
        tgt_img = batch['target_frame'].to(device)

        # 2) Reference images.
        #   Deprecated:
        #       ref_imgs = [img.to(device) for img in ref_imgs]
        #
        ref_imgs = [batch[key].to(device) for key in ['reference_frame:0', 'reference_frame:1']]

        # 3) Camera intrinsics.
        #   Deprecated:
        #       intrinsics = intrinsics.to(device)
        #       intrinsics_inv = intrinsics_inv.to(device)
        #
        intrinsics = batch['camera_intrinsics'].to(device)

        # 4) Telemetry data: speed
        #
        speed_data = None
        if return_telemetry:
            speed_data = batch['telemetry_data/speed']

        # 5) Camera view, test drive ID, video clip indices.
        #
        camera_view = batch['camera_view']
        test_drive_id = batch['test_drive_id']
        video_clip_indices = batch['video_clip_indices'].tolist()

        # --------------------------------------------------------------------------------------------------------------
        # Forward pass.
        # --------------------------------------------------------------------------------------------------------------

        # Compute output: Target depth.
        tgt_depth = [1 / disp_net(tgt_img)]

        # Compute reference depths.
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        # --------------------------------------------------------------------------------------------------------------
        # Log images and estimated depth maps into tensorboard...
        # (only the data corresponding to iterations i = 0, 1, 2)...
        # --------------------------------------------------------------------------------------------------------------

        if log_outputs and i < len(output_writers):

            # ----------------------------------------------------------------------------------------------------------
            # 1) Intrinsic matrix.
            # ----------------------------------------------------------------------------------------------------------

            # Batch sample idx...
            sample_idx = 0

            # Intrinsic matrix elements: Row 0.
            k00 = intrinsics[sample_idx, 0, 0]
            k01 = intrinsics[sample_idx, 0, 1]
            k02 = intrinsics[sample_idx, 0, 2]

            # Intrinsic matrix elements: Row 1.
            k10 = intrinsics[sample_idx, 1, 0]
            k11 = intrinsics[sample_idx, 1, 1]
            k12 = intrinsics[sample_idx, 1, 2]

            # Intrinsic matrix elements: Row 2.
            k20 = intrinsics[sample_idx, 2, 0]
            k21 = intrinsics[sample_idx, 2, 1]
            k22 = intrinsics[sample_idx, 2, 2]

            # String representation of the intrinsic matrix.
            k_matrix_str = \
                'K = {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f}'.format(
                    k00, k01, k02,
                    k10, k11, k12,
                    k20, k21, k22,
                )

            output_writers[i].add_text(
                tag='Intrinsic_matrix/{:s}_{:d}'.format(writer_prefix_tag.lower(), i),
                text_string=k_matrix_str,
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 2) Camera view, test drive, and video clip indices...
            # ----------------------------------------------------------------------------------------------------------

            video_clip_time = (
                    np.asarray(video_clip_indices) / float(args.video_clip_fps)
            ).tolist()

            video_clip_time_str = ['{:0.2f}'.format(t) for t in video_clip_time[0]]

            video_clip_info = \
                'test_drive_id = {} | camera_view = {} | video_clip_indices = {} | video_clip_time = {}'.format(
                    test_drive_id[0],
                    camera_view[0],
                    video_clip_indices[0],
                    video_clip_time_str
                )

            output_writers[i].add_text(
                tag='Video_clip_info/{:s}_{:d}'.format(writer_prefix_tag.lower(), i),
                text_string=video_clip_info,
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 3) Log speed data.
            # ----------------------------------------------------------------------------------------------------------

            if speed_data is not None and return_telemetry:
                output_writers[i].add_scalar(
                    tag='Telemetry_data/{:s}_{:d}_speed'.format(writer_prefix_tag.lower(), i),
                    scalar_value=speed_data[0],
                    global_step=epoch,
                )

            # ----------------------------------------------------------------------------------------------------------
            # 4) Image differences...
            # ----------------------------------------------------------------------------------------------------------

            # Image difference loss: | ref_image_0 - target_image |
            image_diff_loss_0 = (ref_imgs[0][0] - tgt_img[0]).abs()

            # Image difference loss: | ref_image_1 - target_image |
            image_diff_loss_1 = (ref_imgs[1][0] - tgt_img[0]).abs()

            # Image data...
            output_writers[i].add_image(
                tag='{:s}_{:d}_image_difference_map/ref0_target'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(image_diff_loss_0, max_value=None, colormap='magma'),
                global_step=epoch
            )

            output_writers[i].add_image(
                tag='{:s}_{:d}_image_difference_map/ref1_target'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(image_diff_loss_1, max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Losses...
            output_writers[i].add_scalar(
                tag='{:s}_{:d}_image_difference_loss/ref0_target'.format(writer_prefix_tag, i),
                scalar_value=image_diff_loss_0.sum(),
                global_step=epoch
            )

            output_writers[i].add_scalar(
                tag='{:s}_{:d}_image_difference_loss/ref1_target'.format(writer_prefix_tag, i),
                scalar_value=image_diff_loss_1.sum(),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 5) Input image data...
            # ----------------------------------------------------------------------------------------------------------

            # Reference and target frames.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_input/ref0_target_ref1'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(torch.cat([ref_imgs[0][0], tgt_img[0], ref_imgs[1][0]], dim=2)),
                global_step=epoch
            )

            # Target frane.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/input_target_frame'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(tgt_img[0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 6. Depth and disparity.
            #
            #   - alpha = 100, beta = 0.01
            #   - Disparity:  D(x) = (alpha * x + beta)
            #   - Depth:      Z(x) = 1. / D(x)
            #
            # ----------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------
            # 6.1. Disparity.
            # ----------------------------------------------------------------------------------------------------------

            # Estimated disparity map: Normalized w.r.t. maximum value.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/disparity_normalized_wrt_max'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(1. / tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Estimated disparity map: Normalized to the range 0 to 1.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/disparity_normalized_0_to_1'.format(writer_prefix_tag, i),
                img_tensor=normalize_image(1. / tgt_depth[0][0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 6.2. Depth.
            # ----------------------------------------------------------------------------------------------------------

            # Estimated depth map: Normalized w.r.t. maximum value.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/depth_normalized_wrt_max'.format(writer_prefix_tag, i),
                img_tensor=tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Estimated depth map: Normalized w.r.t. maximum value = 30.
            beta_value = 10
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/depth_normalized_wrt_max_{:d}'.format(writer_prefix_tag, i, beta_value),
                img_tensor=tensor2array(tgt_depth[0][0], max_value=beta_value, colormap='magma'),
                global_step=epoch
            )

            # Estimated depth map: Normalized to the range 0 to 1.
            output_writers[i].add_image(
                tag='{:s}_{:d}_model_output/depth_normalized_0_to_1'.format(writer_prefix_tag, i),
                img_tensor=normalize_image(tgt_depth[0][0]),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 7) Histograms...
            # ----------------------------------------------------------------------------------------------------------

            # Histogram of the estimated disparity map.
            output_writers[i].add_histogram(
                tag='{:s}_{:d}_histograms/disparity'.format(writer_prefix_tag, i),
                values=1./tgt_depth[0][0],
                global_step=epoch
            )

            # Histogram of the estimated depth map.
            output_writers[i].add_histogram(
                tag='{:s}_{:d}_histograms/depth'.format(writer_prefix_tag, i),
                values=tgt_depth[0][0],
                global_step=epoch
            )

            # Flushes the event file to disk.
            output_writers[i].flush()

        # --------------------------------------------------------------------------------------------------------------
        # Compute losses...
        # --------------------------------------------------------------------------------------------------------------

        # Compute camera pose...
        poses, poses_inv = compute_pose_with_inv(
            pose_net,
            tgt_img,
            ref_imgs,
            verbose=verbose
        )

        # Compute photometric and geometry consistency losses...
        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img=tgt_img,
            ref_imgs=ref_imgs,
            intrinsics=intrinsics,
            tgt_depth=tgt_depth,
            ref_depths=ref_depths,
            poses=poses,
            poses_inv=poses_inv,
            max_scales=args.num_scales,
            with_ssim=args.with_ssim,
            with_mask=args.with_mask,
            with_auto_mask=False,
            padding_mode=args.padding_mode,
            rotation_mode=args.rotation_matrix_mode,
            writer_obj_tag='{:s}_{:d}'.format(writer_prefix_tag, i),
            writer_obj_step=epoch,
            writer_obj=output_writers[i] if log_outputs and i < len(output_writers) else None,
            device=device
        )

        # Compute smooth loss..
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        # Sum all losses...
        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        # Total loss
        total_loss = loss_1 + loss_2 + loss_3

        losses.update([total_loss, loss_1, loss_2, loss_3])

        # --------------------------------------------------------------------------------------------------------------
        # Measure elapsed time.
        # --------------------------------------------------------------------------------------------------------------

        batch_time.update(time.time() - end)
        end = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # Update logger...
        # --------------------------------------------------------------------------------------------------------------

        if show_progress_bar and not initial_model_evaluation:
            logger.valid_bar.update(i+1)

        if (i+1) % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

        # --------------------------------------------------------------------------------------------------------------
        # Log poses.
        # --------------------------------------------------------------------------------------------------------------

        log_poses = i == 0

        if log_poses:

            # Loop over pose data.
            for img_idx, pose_data in enumerate(poses):

                pose_vector_str = ""

                # Loop over the dimensions of the pose vector.
                for pose_idx in range(pose_data.size()[1]):

                    # Pose variable value.
                    pose_var_value = pose_data[0, pose_idx].item()

                    # Pose variable name.
                    pose_var_name = pose_var_names_dict[pose_idx]

                    # Show the components of the 6D pose vector in a plot.
                    train_writer.add_scalar(
                        tag='{:s}_{:d}_pose_target_wrt_ref{:d}/{:s}'.format(
                            writer_prefix_tag, i,
                            img_idx,
                            pose_var_name
                        ),
                        scalar_value=pose_var_value,
                        global_step=epoch
                    )

                    # Show the components of the 6D pose vector as a string.
                    if pose_idx <= 4:
                        pose_vector_str += "{:s} = {:0.4f}, ".format(pose_var_name, pose_var_value)
                    else:
                        pose_vector_str += "{:s} = {:0.4f}".format(pose_var_name, pose_var_value)

                train_writer.add_text(
                    tag='{:s}_{:d}_pose_target_wrt_ref{:d}/pose_vector_6d'.format(writer_prefix_tag, i, img_idx),
                    text_string=pose_vector_str,
                    global_step=epoch
                )

            # Flushes the event file to disk.
            train_writer.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Flushes the event file to disk.
        # --------------------------------------------------------------------------------------------------------------

        train_writer.flush()

        for ow in output_writers:
            ow.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Condition to break the loop:
        #   Do max iterations have been reached?
        # --------------------------------------------------------------------------------------------------------------

        if max_iterations > 0:
            if i + 1 >= max_iterations:
                break

        # --------------------------------------------------------------------------------------------------------------
        # Increase the counter.
        # --------------------------------------------------------------------------------------------------------------

        i += 1

    # ------------------------------------------------------------------------------------------------------------------
    # Updates progress bar.
    # ------------------------------------------------------------------------------------------------------------------

    if show_progress_bar and not initial_model_evaluation:
        logger.valid_bar.update(max_iterations)

    return losses.avg, [
        '{:s}_loss/total_loss'.format(writer_prefix_tag),
        '{:s}_loss/photometric_loss'.format(writer_prefix_tag),
        '{:s}_loss/disparity_smoothness_loss'.format(writer_prefix_tag),
        '{:s}_loss/geometry_consistency_loss'.format(writer_prefix_tag),
    ]


def compute_depth(disp_net, tgt_img, ref_imgs, verbose=False):

    """ Computes depth maps for target (tgt_img) and reference (ref_imgs) images. """

    # ------------------------------------------------------------------------------------------------------------------
    # Computing depth.
    # ------------------------------------------------------------------------------------------------------------------

    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []

    for ref_img in ref_imgs:

        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    # ------------------------------------------------------------------------------------------------------------------
    # Print data information.
    # ------------------------------------------------------------------------------------------------------------------

    if verbose:
        print('[ Computing depth ][ Call --> compute_depth( . ) ]')
        print('\t[ Target depths ] N = {} depth maps'.format(len(tgt_depth)))
        for i, td in enumerate(tgt_depth):
            print('\t\t[ {:d} ] Shape = {} | Data-type = {} | Device = {}'.format(i, td.shape, td.dtype, td.device))

        print('\t[ Reference depths ] N = {} items'.format(len(ref_depths)))
        for i, rd_list in enumerate(ref_depths):
            print('\t\t[ Item = {} | Reference depths ] M = {} depth maps'.format(i, len(rd_list)))
            for j, rd in enumerate(rd_list):
                print('\t\t\t[ {:d} ] Shape = {} | Data-type = {} | Device = {}'.format(j, rd.shape, rd.dtype, rd.device))
        print(' ')

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs, verbose=True):

    """ Compute pose data for the target (tgt_img) and reference (ref_imgs) images. """

    # ------------------------------------------------------------------------------------------------------------------
    # Computing poses.
    # ------------------------------------------------------------------------------------------------------------------

    poses = []
    poses_inv = []

    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    # ------------------------------------------------------------------------------------------------------------------
    # Print data information.
    # ------------------------------------------------------------------------------------------------------------------

    if verbose:

        print('[ Computing Pose ][ Call --> compute_pose_with_inv( . ) ]')
        print('\t[ Poses ] N = {}'.format(len(poses)))

        for i, p in enumerate(poses):
            print('\t\t[ {:d} ] Shape = {} | Data-type = {} | Device = {}'.format(i, p.shape, p.dtype, p.device))

        print('\t[ Poses-Inv ] N = {}'.format(len(poses_inv)))
        for i, pinv in enumerate(poses_inv):
            print('\t\t[ {:d} ] Shape = {} | Data-type = {} | Device = {}'.format(i, pinv.shape, pinv.dtype, pinv.device))
        print(' ')

    return poses, poses_inv


if __name__ == '__main__':
    main()
