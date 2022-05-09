import sys
sys.path.insert(1, '/home/arturo/workspace/pycharm_projects/data_loader_ml/DataLoaderML')
import os
import argparse
import time
import csv
import datetime
from path import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import pandas as pd
import yaak.tools.custom_transforms as CT
from yaak.tools.dataset import YaakIterableDataset
from utils import tensor2array, save_checkpoint, count_parameters, print_batch
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from torch.utils.tensorboard import SummaryWriter

camera_view_choices = YaakIterableDataset.get_camera_view_choices() + ['all']

parser = argparse.ArgumentParser(
    description='Structure from Motion Learner -- Yaak Dataset 1.0',
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
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='Path to pre-trained Dispnet model.')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='Path to pre-trained Pose net model.')
parser.add_argument('--name', dest='name', type=str, required=True, help='Name of the experiment, checkpoints are stored in checpoints/name.')
parser.add_argument('--rotation-matrix-mode', type=str, choices=['euler', 'quat'], default='euler', help='Rotation matrix representation.')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='Padding mode for '
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
    save_path = Path(args.name)

    # Path to save data.
    args.save_path = '{}/{}/{}'.format(args.checkpoints_path, save_path, timestamp)
    print('[ Experimental results ] Save path: {}'.format(args.save_path))

    # If the path does not exist, it is created.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('\t- Creating directory: {}'.format(args.save_path))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

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
        ('name', args.name),
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
    frame_crop_size = {
        '0.25': (256, -1),
        '0.50': (512, -1),
        '0.75': (768, -1),
        '1.00': (-1, -1),
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
        CT.ScaleCenterCrop(
            crop_size=frame_crop_size['{:0.2f}'.format(frame_scaling_factor)],
            scaling_factor=frame_scaling_factor
        ),
        CT.RandomHorizontalFlip(),
        CT.RandomScaleCrop(),
        CT.ToFloat32TensorCHW(),
        CT.Normalize(mean=mean_per_channel, std=std_per_channel)
    ])

    # Transformations applied on validation data.
    val_transform = CT.Compose([
        CT.ScaleCenterCrop(
            crop_size=frame_crop_size['{:0.2f}'.format(frame_scaling_factor)],
            scaling_factor=frame_scaling_factor
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
        # Define which data should be returned by the data loader.
        # --------------------------------------------------------------------------------------------------------------

        return_data = (
            ('processed_videos', True),
            ('tgt_ref_frames', True),
            ('telemetry', False),
            ('camera_intrinsics', True),
            ('camera_distortion', False),
        )

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

        # Dictionary of test drive IDs.
        test_drive_id_dict = {
            'train': None if args.test_drive_id_train == "None" else args.test_drive_id_train,
            'val': None if args.test_drive_id_val == "None" else args.test_drive_id_val,
        }

        # --------------------------------------------------------------------------------------------------------------
        # Create train dataset.
        # --------------------------------------------------------------------------------------------------------------

        print('\t- Creating the train dataset using the YaakIterableDataset class.')

        train_dataset = YaakIterableDataset(
            dataset_path=args.dataset_path,
            cam_calib_path=args.cam_calib_path,
            test_drive_id=test_drive_id_dict['train'],
            camera_view=camera_view_dict['train'],
            video_extension='mp4',
            telemetry_filename='metadata.log',
            video_clip_length=args.video_clip_length,
            video_clip_step=args.video_clip_step,
            video_clip_memory_format='default',
            video_clip_output_dtype='default',
            frame_target_size=frame_target_size,
            video_clip_fps=30.,
            processed_videos_filename_suffix='-force-key.defish',
            return_data=return_data,
            transform=train_transform,
            device_id=torch.cuda.current_device(),
            device_name='gpu' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )

        # --------------------------------------------------------------------------------------------------------------
        # Create validation dataset.
        # --------------------------------------------------------------------------------------------------------------

        print('\t- Creating the validation dataset using the YaakIterableDataset class.')

        val_dataset = YaakIterableDataset(
            dataset_path=args.dataset_path,
            cam_calib_path=args.cam_calib_path,
            test_drive_id=test_drive_id_dict['val'],
            camera_view=camera_view_dict['val'],
            video_extension='mp4',
            telemetry_filename='metadata.log',
            video_clip_length=args.video_clip_length,
            video_clip_step=args.video_clip_step,
            video_clip_memory_format='default',
            video_clip_output_dtype='default',
            frame_target_size=frame_target_size,
            video_clip_fps=30.,
            processed_videos_filename_suffix='-force-key.defish',
            return_data=return_data,
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

    # Disparity network.
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain, verbose=False).to(device)

    # Pose network.
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

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
    # Load pre-trained model parameters.
    # ------------------------------------------------------------------------------------------------------------------

    # Load pre-trained disparity network parameters.
    print('\n[ Load pretrained model parameters ]')
    if args.pretrained_disp:
        print("\t- Using pre-trained weights for DispResNet.")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    # Load pre-trained pose network parameters.
    if args.pretrained_pose:
        print("\t- Using pre-trained weights for PoseResNet.")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Data parallelism...
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
    # Count the number of disparity/pose model parameters.
    # ------------------------------------------------------------------------------------------------------------------

    # Disparity network parameters.
    disp_net_num_params = count_parameters(disp_net)

    # Pose network parameters.
    pose_net_num_params = count_parameters(pose_net)

    print('\n[ Count model parameters ]')
    print('\t- DispResNet | N = {:0.2f}M params'.format(disp_net_num_params/1e6))
    print('\t- PoseResNet | N = {:0.2f}M params'.format(pose_net_num_params/1e6))

    # ------------------------------------------------------------------------------------------------------------------
    # Create optimizers for the Disparity and Pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Creating optimizers for the Disparity and Pose networks ] Optimizer = Adam')

    # Parameters
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]

    # Optimizer.
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
    # Training stage...
    #
    ####################################################################################################################

    print('\n[ Start model training ]...')

    # ------------------------------------------------------------------------------------------------------------------
    # Creating a Logger.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Creating a Logger ]')

    logger = TermLogger(
        n_epochs=args.epochs,
        train_size=args.max_train_iterations,
        valid_size=args.max_val_iterations,
    )

    logger.epoch_bar.start()

    # ------------------------------------------------------------------------------------------------------------------
    # Start model training...
    # ------------------------------------------------------------------------------------------------------------------

    for epoch in range(args.epochs):

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
                show_progress_bar=True,
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

    # --------------------------------------------------------------------------------------------------------------
    # Closing operation...
    # --------------------------------------------------------------------------------------------------------------

    training_writer.close()

    for ow in output_writers:
        ow.close()


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
                        tag='Train_pose_target_wrt_ref_image_{:d}/{:s}'.format(img_idx, pose_var_name),
                        scalar_value=pose_var_value,
                        global_step=epoch
                    )

                    if pose_idx <= 4:
                        pose_vector_str += "{:s} = {:0.4f}, ".format(pose_var_name, pose_var_value)
                    else:
                        pose_vector_str += "{:s} = {:0.4f}".format(pose_var_name, pose_var_value)

                # Show the components of the 6D pose vector as a string.
                train_writer.add_text(
                    tag='Train_pose_target_wrt_ref_image_{:d}/pose_vector_6d'.format(img_idx),
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
            # 1) Training losses...
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
            # 2) Image differences...
            # ----------------------------------------------------------------------------------------------------------

            # Image difference loss: | ref_image_0 - target_image |
            image_diff_loss_0 = (ref_imgs[0][0] - tgt_img[0]).abs()

            # Image difference loss: | ref_image_1 - target_image |
            image_diff_loss_1 = (ref_imgs[1][0] - tgt_img[0]).abs()

            # Image data...
            train_writer.add_image(
                tag='Train_image_difference_ref0_target',
                img_tensor=tensor2array(image_diff_loss_0, max_value=None, colormap='magma'),
                global_step=epoch
            )

            train_writer.add_image(
                tag='Train_image_difference_ref1_target',
                img_tensor=tensor2array(image_diff_loss_1, max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Losses...
            train_writer.add_scalar(
                tag='Train_image_difference/ref0_target',
                scalar_value=image_diff_loss_0.sum(),
                global_step=epoch
            )

            train_writer.add_scalar(
                tag='Train_image_difference/ref1_target',
                scalar_value=image_diff_loss_1.sum(),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 3) Image data...
            # ----------------------------------------------------------------------------------------------------------

            # Input image.
            train_writer.add_image(
                tag='Train_input',
                img_tensor=tensor2array(torch.cat([ref_imgs[0][0], tgt_img[0], ref_imgs[1][0]], dim=2)),
                global_step=epoch
            )

            # Estimated disparity map.
            train_writer.add_image(
                tag='Train_dispnet_output_normalized',
                img_tensor=tensor2array(1. / tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Depth and disparity output:
            #
            #   - alpha = 100, beta = 0.01
            #   - Disparity:  D(x) = (alpha * x + beta)
            #   - Depth:      Z(x) = 1. / D(x)
            #
            # Previous setting: max_value = tgt_depth[0][0].max().item()
            #
            beta_value = None

            # Estimated depth map.
            train_writer.add_image(
                tag='Train_depth_output',
                img_tensor=tensor2array(tgt_depth[0][0], max_value=beta_value, colormap='magma'),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 4) Histograms...
            # ----------------------------------------------------------------------------------------------------------

            # Histogram of the estimated disparity map.
            train_writer.add_histogram(
                tag='Train_histograms/Train_dispnet_output_normalized',
                values=1./tgt_depth[0][0],
                global_step=epoch
            )

            # Histogram of the estimated depth map.
            train_writer.add_histogram(
                tag='Train_histograms/Train_depth_output',
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
    show_progress_bar=False,
    device=torch.device("cpu"),
    verbose=False,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

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
    if show_progress_bar:
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
            # Image differences...
            # ----------------------------------------------------------------------------------------------------------

            # Image difference loss: | ref_image_0 - target_image |
            image_diff_loss_0 = (ref_imgs[0][0] - tgt_img[0]).abs()

            # Image difference loss: | ref_image_1 - target_image |
            image_diff_loss_1 = (ref_imgs[1][0] - tgt_img[0]).abs()

            # Image data...
            output_writers[i].add_image(
                tag='Val_image_difference_ref0_target',
                img_tensor=tensor2array(image_diff_loss_0, max_value=None, colormap='magma'),
                global_step=epoch
            )

            output_writers[i].add_image(
                tag='Val_image_difference_ref1_target',
                img_tensor=tensor2array(image_diff_loss_1, max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Losses...
            output_writers[i].add_scalar(
                tag='Val_image_difference_{}/ref0_target'.format(i),
                scalar_value=image_diff_loss_0.sum(),
                global_step=epoch
            )

            output_writers[i].add_scalar(
                tag='Val_image_difference_{}/ref1_target'.format(i),
                scalar_value=image_diff_loss_1.sum(),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # Image data...
            # ----------------------------------------------------------------------------------------------------------

            # Input image.
            output_writers[i].add_image(
                tag='Val_input',
                img_tensor=tensor2array(torch.cat([ref_imgs[0][0], tgt_img[0], ref_imgs[1][0]], dim=2)),
                global_step=epoch
            )

            # Estimated disparity map.
            output_writers[i].add_image(
                tag='Val_dispnet_output_normalized',
                img_tensor=tensor2array(1./tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=epoch
            )

            # Depth and disparity output:
            #
            #   - alpha = 100, beta = 0.01
            #   - Disparity:  D(x) = (alpha * x + beta)
            #   - Depth:      Z(x) = 1. / D(x)
            #
            # Previous setting: max_value = tgt_depth[0][0].max().item()
            #
            beta_value = None

            # Estimated depth map.
            output_writers[i].add_image(
                tag='Val_depth_output',
                img_tensor=tensor2array(tgt_depth[0][0], max_value=beta_value, colormap='magma'),
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # Histograms...
            # ----------------------------------------------------------------------------------------------------------

            # Histogram of the estimated disparity map.
            output_writers[i].add_histogram(
                tag='Val_histograms/Val_dispnet_output_normalized_{:d}'.format(i),
                values=1./tgt_depth[0][0],
                global_step=epoch
            )

            # Histogram of the estimated depth map.
            output_writers[i].add_histogram(
                tag='Val_histograms/Val_depth_output_{:d}'.format(i),
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
            writer_obj_tag='Val',
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

        if show_progress_bar:
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
                        tag='Val_pose_target_wrt_ref_image_{:d}/{:s}'.format(img_idx, pose_var_name),
                        scalar_value=pose_var_value,
                        global_step=epoch
                    )

                    # Show the components of the 6D pose vector as a string.
                    if pose_idx <= 4:
                        pose_vector_str += "{:s} = {:0.4f}, ".format(pose_var_name, pose_var_value)
                    else:
                        pose_vector_str += "{:s} = {:0.4f}".format(pose_var_name, pose_var_value)

                train_writer.add_text(
                    tag='Val_pose_target_wrt_ref_image_{:d}/pose_vector_6d'.format(img_idx),
                    text_string=pose_vector_str,
                    global_step=epoch
                )

            # Flushes the event file to disk.
            train_writer.flush()

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

        # --------------------------------------------------------------------------------------------------------------
        # Flushes the event file to disk.
        # --------------------------------------------------------------------------------------------------------------

        train_writer.flush()

        for ow in output_writers:
            ow.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Updates progress bar.
    # ------------------------------------------------------------------------------------------------------------------

    if show_progress_bar:
        logger.valid_bar.update(max_iterations)

    return losses.avg, [
        'Val_loss/total_loss',
        'Val_loss/photometric_loss',
        'Val_loss/disparity_smoothness_loss',
        'Val_loss/geometry_consistency_loss'
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
