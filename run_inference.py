import traceback
import sys
import shutil
sys.path.insert(1, '/home/arturo/workspace/pycharm_projects/data_loader_ml/DataLoaderML')
import wandb
import os
import argparse
import time
import csv
import datetime
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import toml
import random
import string
import skvideo.io
from tqdm import tqdm
from pathlib import Path
from skimage import color
from data_loader_ml.dataset import YaakIterableDataset
from data_loader_ml.tools.custom_transforms import Compose, TRANSFORM_DICT
from utils import tensor2array, save_checkpoint, count_parameters, print_batch, normalize_image
from utils import get_hyperparameters_dict
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(
    description='Model inference and validation (SfM learner) on the Yaak dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    default="configs/model_inference/inference_validation_sequential_samples.toml",
    type=Path,
    help="TOML configuration file to carry out model inference and validation on the Yaak dataset.",
)
parser.add_argument(
    '-b', '--batch-size',
    default=1,
    type=int,
    help='Batch size.'
)
parser.add_argument(
    '--max-val-iterations',
    default=-1,
    type=int,
    help='Max. number of iterations in the validation set. Use -1 to process the whole video.'
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Enable GPU usage."
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

    # ------------------------------------------------------------------------------------------------------------------
    # Read arguments from command line.
    # ------------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Read the contents of the TOML file.
    # ------------------------------------------------------------------------------------------------------------------

    if not args.config.is_file():
        print(f"Error reading {args.config}, No such file")
        return

    with args.config.open("r") as pfile:
        cfgs = toml.load(pfile)

    # The current script works with batch size 1.
    assert args.batch_size == 1, \
        "[ Error ] The current script works with batch size 1. " \
        "Support for larger batch sizes will be implemented in the future."

    # ------------------------------------------------------------------------------------------------------------------
    # Get the current device.
    # ------------------------------------------------------------------------------------------------------------------

    # Enable GPU
    enable_gpu = torch.cuda.is_available() and args.gpu

    # Selected device.
    device = torch.device("cuda") if enable_gpu else torch.device("cpu")

    print('[ Current device (torch.cuda.current_device()) ] {}'.format(torch.cuda.current_device()))
    print('[ GPU enabled ] {}'.format(enable_gpu))
    print('[ Device name ] {}'.format(device))
    print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Number of GPUs
    # ------------------------------------------------------------------------------------------------------------------

    if enable_gpu:
        num_gpus = torch.cuda.device_count()
        print('[ Number of GPUs available for training ] N = {}'.format(num_gpus))
        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Seed numbers and CUDA settings..
    # ------------------------------------------------------------------------------------------------------------------

    torch.manual_seed(cfgs["experiment_settings"]["seed"])
    np.random.seed(cfgs["experiment_settings"]["seed"])

    if enable_gpu:
        cudnn.deterministic = True
        cudnn.benchmark = True

    # ------------------------------------------------------------------------------------------------------------------
    # Data loader to be used.
    # ------------------------------------------------------------------------------------------------------------------

    print(f"[ Data Loader {YaakIterableDataset.desc()}, Version {str(YaakIterableDataset.version())} ]")
    print(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Global vars.
    # ------------------------------------------------------------------------------------------------------------------

    # global best_error, n_iter, device
    global best_error, n_iter

    # ------------------------------------------------------------------------------------------------------------------
    # Paths...
    # ------------------------------------------------------------------------------------------------------------------

    # Time instant.
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H%MH")

    # Experiment name
    experiment_name = cfgs["experiment_settings"]["experiment_name"]

    # Path to save data.
    save_path = '{}/{}/{}'.format(
        cfgs["experiment_settings"]["checkpoints_path"],
        experiment_name,
        timestamp
    )

    print('[ Experimental results ] Save path: {}'.format(save_path))

    # If the path does not exist, it is created.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('\t- Creating directory: {}'.format(save_path))

    # Check the value of the arguments pretrained_disp and pretrained_pose.
    pretrained_disparity_model_file = None if cfgs["experiment_settings"]["pretrained_disp"] == "None" \
        else cfgs["experiment_settings"]["pretrained_disp"]

    pretrained_pose_model_file = None if cfgs["experiment_settings"]["pretrained_pose"] == "None" \
        else cfgs["experiment_settings"]["pretrained_pose"]

    # ------------------------------------------------------------------------------------------------------------------
    # Validation hyper-parameters.
    # ------------------------------------------------------------------------------------------------------------------

    wandb_hparams_dict = get_hyperparameters_dict(
        device=device,
        epochs=0,
        batch_size=args.batch_size,
        max_train_iterations=0,
        max_val_iterations=args.max_val_iterations,
        cfgs=cfgs,
        mode="val",
        save_path=save_path,
        verbose=True
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Init Weights & Biases...
    # ------------------------------------------------------------------------------------------------------------------

    tensorboard_root_logdir = "{}/tb".format(save_path)

    if cfgs["experiment_settings"]["wandb_enable"]:

        # The name of the project where you're sending the new run.
        wandb_project_name = cfgs["experiment_settings"]["wandb_project_name"]

        # An absolute path to a directory where metadata will be stored. When you call download() on an artifact,
        # this is the directory where downloaded files will be saved. By default this is the ./wandb directory.
        wandb_dir = cfgs["experiment_settings"]["wandb_dir"]

        # If the path does not exist, it is created.
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)

        # When using several event log directories, please call `wandb.tensorboard.patch(root_logdir="...")`
        # before `wandb.init`
        wandb.tensorboard.patch(root_logdir=tensorboard_root_logdir)

        # Starts a new run to track and log to W&B.
        # Pass `sync_tensorboard=True`, to plot your TensorBoard files
        wandb.init(
            dir=wandb_dir,
            config=wandb_hparams_dict,
            project=wandb_project_name,
            sync_tensorboard=True
        )

        print('\n[ Initializing Weights & Biases ]')
        print('\t- Project name: {}'.format(wandb_project_name))
        print('\t- Metadata path: {}'.format(wandb_dir))
        print('\t- Tensorboard root log dir: {}'.format(tensorboard_root_logdir))
        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Summary writers...
    # ------------------------------------------------------------------------------------------------------------------

    # Tensorboard train/validation log directories...
    tensorboard_val_log_dir = "{}/val".format(tensorboard_root_logdir)

    # Validation summary writer.
    val_writer = None
    if cfgs["experiment_settings"]["log_output"]:
        val_writer = SummaryWriter(tensorboard_val_log_dir, flush_secs=10, max_queue=100)

    print('\n[ Initializing Tensorboard ]')
    print('\t- Val log path: {}'.format(tensorboard_val_log_dir))

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations applied on data during the validation stage.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating validation data transformations ]")

    # Transformations applied on validation data.
    val_transform_cfgs = cfgs["val"]["configuration"]["transformation"]["series"]
    val_transform = Compose(
        [TRANSFORM_DICT[fn](**kwargs) for fn, kwargs in val_transform_cfgs.items()]
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Validation dataset paths.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Validation dataset paths ]")
    print("\t- Video sequences path: {}".format(cfgs["val"]["dataset"]["rootpath"]))
    print("\t- Camera calibration path: {}".format(cfgs["val"]["dataset"]["camera_calibration"]))

    # ------------------------------------------------------------------------------------------------------------------
    # Create the validation dataset.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Creating the validation dataset using the YaakIterableDataset class ]')
    print('\t- Every video in the validation set will be over-sampled by N = {} iterations.'.format(
            cfgs["val"]["configuration"]["sampling"]["oversampling"]
        )
    )
    print(' ')

    val_dataset = YaakIterableDataset(
        start=0,
        end=args.max_val_iterations,
        dataset=cfgs["val"]["dataset"],
        config_frames=cfgs["val"]["configuration"]["frame"],
        config_sampling=cfgs["val"]["configuration"]["sampling"],
        config_returns=cfgs["val"]["configuration"]["return"],
        transform=val_transform,
        device_id=torch.cuda.current_device(),
        device_name="gpu" if enable_gpu else "cpu",
        verbose=True,
    )
    print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Create validation data loaders.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating data loaders ]")
    print("\t- Creating a validation data loader.")

    # Validation loader.
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfgs["experiment_settings"]["workers"],
        pin_memory=False,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Create models: Disparity and Pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating models ] Disparity (DispResNet) and Pose (PoseResNet) networks.")

    if cfgs["experiment_settings"]["with_pretrain"]:
        print('\t- The disparity (encoder) and pose network parameters'
              ' will be initialized from pretrained weights (e.g., ImageNet).')

    print("\t- DispResNet (num_layers = {}, pretrain = {}) | Device = {}".format(
                cfgs["experiment_settings"]["resnet_layers"],
                cfgs["experiment_settings"]["with_pretrain"],
                device
        )
    )
    print("\t- PoseResNet (num_layers = 18, pretrain = {}) | Device = {}".format(
            cfgs["experiment_settings"]["with_pretrain"],
            device
        )
    )

    # Disparity network.
    disp_net = \
        models.DispResNet(
            num_layers=cfgs["experiment_settings"]["resnet_layers"],
            pretrained=cfgs["experiment_settings"]["with_pretrain"],
            verbose=False
        ).to(device)

    # Pose network.
    pose_net = models.PoseResNet(
        num_layers=18,
        pretrained=cfgs["experiment_settings"]["with_pretrain"]
    ).to(device)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize DispResNet/PoseResNet networks with pre-trained weights stored on disk...
    # ------------------------------------------------------------------------------------------------------------------

    # Load pre-trained disparity network parameters.
    if pretrained_disparity_model_file:
        print('\n[ Initializing disparity network (DispResNet) with pretrained weights stored on disk ]')
        print("\t- File: {}".format(pretrained_disparity_model_file))
        weights = torch.load(pretrained_disparity_model_file)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
        print("\t- The model parameters have been updated.")

    # Load pre-trained pose network parameters.
    if pretrained_pose_model_file:
        print('\n[ Initializing the pose network (PoseResNet) with pretrained weights stored on disk ]')
        print("\t- File: {}".format(pretrained_pose_model_file))
        weights = torch.load(pretrained_pose_model_file)
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
    print(' ')

    # Adding the disparity network's parameter count to tensorboard.
    val_writer.add_text(
        tag='Parameter_count/disp_resnet',
        text_string='{:0.2f}M'.format(disp_net_num_params),
        global_step=0
    )

    # Adding the pose network's parameter count to tensorboard.
    val_writer.add_text(
        tag='Parameter_count/pose_resnet',
        text_string='{:0.2f}M'.format(pose_net_num_params),
        global_step=0
    )

    val_writer.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Wrapping the models with torch.nn.DataParallel...
    # ------------------------------------------------------------------------------------------------------------------

    disp_net = torch.nn.DataParallel(disp_net, output_device=0)
    pose_net = torch.nn.DataParallel(pose_net, output_device=0)

    disp_net = disp_net.to(device)
    pose_net = pose_net.to(device)

    print('\n[ Data parallelism (with torch.nn.DataParallel) ]')
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
    # CSV file to store video data, validation losses, camera translation and orientation (between two frames):
    # ------------------------------------------------------------------------------------------------------------------

    # CSV file.
    csv_log_full_ffname = '{}/{}'.format(save_path, cfgs["experiment_settings"]["log_full"])

    # CSV header: 20 variables.
    csv_log_full_header_list = [
        #  Index, test drive id, and video information.
        'index',
        'test_drive_id',
        'camera_view',
        'video_clip_indices',
        # Losses.
        'total_loss',
        'photometric_loss',
        'smoothness_loss',
        'geometry_consistency_loss',
        # Camera pose: Target frame w.r.t. reference frame 0.
        'x0',
        'y0',
        'z0',
        'theta_x0',
        'theta_y0',
        'theta_z0',
        # Camera pose: Target frame w.r.t. reference frame 1.
        'x1',
        'y1',
        'z1',
        'theta_x1',
        'theta_y1',
        'theta_z1',
        # Telemetry data: latitude and longitude.
        'latitude',
        'longitude'
    ]

    print(f'\n[ Creating a CSV file to save data per iteration (e.g., validation losses and odometry). ]')
    print(f'\t- CSV File: {csv_log_full_ffname}')
    print(f'\t- CSV Header: ')
    for h in csv_log_full_header_list:
        print(f"\t\t- {h}")

    with open(csv_log_full_ffname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(csv_log_full_header_list)

    ####################################################################################################################
    #
    # Model inference and validation...
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Creating a Logger.
    # ------------------------------------------------------------------------------------------------------------------

    total_videos = val_dataset.get_num_videos

    print('\n[ Start model inference and validation ] Total videos = {}'.format(total_videos))
    print('')

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the model, on the validation set, for N = max_val_iterations iterations.
    # ------------------------------------------------------------------------------------------------------------------

    errors, error_names = \
        validate_without_gt(
            cfgs=cfgs,
            val_loader=val_loader,
            disp_net=disp_net,
            pose_net=pose_net,
            writer_obj=val_writer,
            writer_obj_tag="Val",
            save_path=save_path,
            return_telemetry=cfgs["val"]["configuration"]["return"]["telemetry"],
            device=device,
            verbose=True,
        )

    # Validation loss.
    error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
    print('[ Average validation loss ] {}'.format(error_string))
    print(' ')

    # Write validation losses in tensorboard.
    for error, name in zip(errors, error_names):
        val_writer.add_scalar(tag=name, scalar_value=error, global_step=0)

    # Flushes the event file to disk.
    val_writer.flush()

    # Closing operations...
    val_writer.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Finish the wandb run to upload the TensorBoard logs to W & B.
    # ------------------------------------------------------------------------------------------------------------------

    if cfgs["experiment_settings"]["wandb_enable"]:
        wandb.finish()


@torch.no_grad()
def validate_without_gt(
    cfgs,
    val_loader,
    disp_net,
    pose_net,
    writer_obj,
    writer_obj_tag="Val",
    save_path=None,
    return_telemetry=False,
    device=torch.device("cpu"),
    verbose=False,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

    # Prefix used to log data in tensorboard...
    writer_prefix_tag = writer_obj_tag

    # Time
    batch_time = AverageMeter()

    # Losses.
    losses = AverageMeter(i=4, precision=4)

    # Enable logs.
    log_outputs = cfgs["experiment_settings"]["log_output"]

    # Print frequency.
    print_freq = cfgs["experiment_settings"]["print_freq"]

    # Set the networks in evaluation mode.
    disp_net.eval()
    pose_net.eval()

    # End time.
    end = time.time()

    # ------------------------------------------------------------------------------------------------------------------
    # Create a path to store the video data
    # ------------------------------------------------------------------------------------------------------------------

    # Number of random characters.
    num_random_chars = 10

    # Random video id.
    source_video_random_id = \
        ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(num_random_chars))

    # Source path to store the video data locally.
    source_video_path = f"temp_videos/{source_video_random_id}"

    # Destiny path store the video data (the video will be moved here after the inference process is completed).
    destiny_video_path = f"{save_path}/videos"

    if not os.path.exists(source_video_path):
        os.makedirs(source_video_path)
        if verbose:
            print(f"[ Creating SOURCE directory to store video data ] Source path: {source_video_path}")
            print(" ")

    if not os.path.exists(destiny_video_path):
        os.makedirs(destiny_video_path)
        if verbose:
            print(f"[ Creating DESTINY directory to store video data ] Destiny path: {destiny_video_path}")
            print(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Create a video writer object to store RGB and disparity frames on a single video...
    # ------------------------------------------------------------------------------------------------------------------

    video_writer_obj = None

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

    pbar = tqdm(val_loader)

    for batch in pbar:

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

        # 4) Telemetry data: Latitude, longitude, speed data.
        #
        speed_data = None
        latitude_data = None
        longitude_data = None
        if return_telemetry:
            # speed_data = batch['telemetry_data/speed'] <-- Not ready.
            latitude_data = batch["telemetry_data/latitude"]
            longitude_data = batch["telemetry_data/longitude"]

        # 5) Camera view, test drive ID, video clip indices.
        #
        camera_view = batch['camera_view']
        test_drive_id = batch['test_drive_id']
        video_clip_indices = batch['video_clip_indices'].tolist()
        video_clip_indices_str = " ".join(list(map(str, video_clip_indices)))

        # --------------------------------------------------------------------------------------------------------------
        # Create a video writer object to store RGB and disparity frames on a single video...
        # --------------------------------------------------------------------------------------------------------------

        if i == 0:

            try:

                source_video_ffname = f'{source_video_path}/{test_drive_id[0]}_rgb_disparity.mp4'

                video_writer_obj = skvideo.io.FFmpegWriter(
                    source_video_ffname,
                    # outputdict={'-vcodec': 'libx264', '-b': '300000000'}
                )

                if verbose:
                    print("[ Video writer ] Object created.")
                    print(f"\t- RGB and disparity frames will be saved in: {source_video_ffname}")
                    print(" ")

            except Exception as err:
                print(f"[ Exception ] Error while creating an video writer object with skvideo.io.FFmpegWriter: {err}")
                traceback.print_exc()
                print(" ")

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
        # Condition to log data.
        # --------------------------------------------------------------------------------------------------------------

        enable_log_data = i % print_freq == 0 and log_outputs

        # --------------------------------------------------------------------------------------------------------------
        # Log images and estimated depth maps into tensorboard...
        # (only the data corresponding to iterations i = 0, 1, 2)...
        # --------------------------------------------------------------------------------------------------------------

        if enable_log_data:

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

            writer_obj.add_text(
                tag='Intrinsic_matrix/{:s}'.format(writer_prefix_tag.lower()),
                text_string=k_matrix_str,
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # 2) Camera view, test drive, and video clip indices...
            # ----------------------------------------------------------------------------------------------------------

            video_clip_time = (
                np.asarray(video_clip_indices) / float(cfgs["val"]["configuration"]["frame"]["frame_fps"])
            ).tolist()

            video_clip_time_str = ['{:0.2f}'.format(t) for t in video_clip_time[0]]

            video_clip_info = \
                'test_drive_id = {} | camera_view = {} | video_clip_indices = {} | video_clip_time = {}'.format(
                    test_drive_id[0],
                    camera_view[0],
                    video_clip_indices[0],
                    video_clip_time_str
                )

            writer_obj.add_text(
                tag='Video_clip_info/{:s}'.format(writer_prefix_tag.lower()),
                text_string=video_clip_info,
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # 3) Log telemetry data: Speed, latitude, and longitude data.
            # ----------------------------------------------------------------------------------------------------------

            # Speed data.
            if speed_data is not None and return_telemetry:
                writer_obj.add_scalar(
                    tag='Telemetry_data/{:s}_speed'.format(writer_prefix_tag.lower()),
                    scalar_value=speed_data[0],
                    global_step=i,
                )

            # Latitude data.
            if latitude_data is not None and return_telemetry:
                writer_obj.add_scalar(
                    tag='Telemetry_data/{:s}_latitude'.format(writer_prefix_tag.lower()),
                    scalar_value=latitude_data[0],
                    global_step=i,
                )

            # Longitude data.
            if longitude_data is not None and return_telemetry:
                writer_obj.add_scalar(
                    tag='Telemetry_data/{:s}_longitude'.format(writer_prefix_tag.lower()),
                    scalar_value=longitude_data[0],
                    global_step=i,
                )

            # ----------------------------------------------------------------------------------------------------------
            # 4) Image differences...
            # ----------------------------------------------------------------------------------------------------------

            # Image difference loss: | ref_image_0 - target_image |
            image_diff_loss_0 = (ref_imgs[0][0] - tgt_img[0]).abs()

            # Image difference loss: | ref_image_1 - target_image |
            image_diff_loss_1 = (ref_imgs[1][0] - tgt_img[0]).abs()

            # Image data...
            writer_obj.add_image(
                tag='{:s}_image_difference_map/ref0_target'.format(writer_prefix_tag),
                img_tensor=tensor2array(image_diff_loss_0, max_value=None, colormap='magma'),
                global_step=i
            )

            writer_obj.add_image(
                tag='{:s}_image_difference_map/ref1_target'.format(writer_prefix_tag),
                img_tensor=tensor2array(image_diff_loss_1, max_value=None, colormap='magma'),
                global_step=i
            )

            # Losses...
            writer_obj.add_scalar(
                tag='{:s}_image_difference_loss/ref0_target'.format(writer_prefix_tag),
                scalar_value=image_diff_loss_0.sum(),
                global_step=i
            )

            writer_obj.add_scalar(
                tag='{:s}_image_difference_loss/ref1_target'.format(writer_prefix_tag),
                scalar_value=image_diff_loss_1.sum(),
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # 5) Input image data...
            # ----------------------------------------------------------------------------------------------------------

            # Reference and target frames.
            writer_obj.add_image(
                tag='{:s}_model_input/ref0_target_ref1'.format(writer_prefix_tag),
                img_tensor=tensor2array(torch.cat([ref_imgs[0][0], tgt_img[0], ref_imgs[1][0]], dim=2)),
                global_step=i
            )

            # Target frane.
            writer_obj.add_image(
                tag='{:s}_model_output/input_target_frame'.format(writer_prefix_tag),
                img_tensor=tensor2array(tgt_img[0]),
                global_step=i
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
            writer_obj.add_image(
                tag='{:s}_model_output/disparity_normalized_wrt_max'.format(writer_prefix_tag),
                img_tensor=tensor2array(1. / tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=i
            )

            # Estimated disparity map: Normalized to the range 0 to 1.
            writer_obj.add_image(
                tag='{:s}_model_output/disparity_normalized_0_to_1'.format(writer_prefix_tag),
                img_tensor=normalize_image(1. / tgt_depth[0][0]),
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # 6.2. Depth.
            # ----------------------------------------------------------------------------------------------------------

            # Estimated depth map: Normalized w.r.t. maximum value.
            writer_obj.add_image(
                tag='{:s}_model_output/depth_normalized_wrt_max'.format(writer_prefix_tag),
                img_tensor=tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                global_step=i
            )

            # Estimated depth map: Normalized w.r.t. maximum value = 30.
            beta_value = 10
            writer_obj.add_image(
                tag='{:s}_model_output/depth_normalized_wrt_max_{:d}'.format(writer_prefix_tag, beta_value),
                img_tensor=tensor2array(tgt_depth[0][0], max_value=beta_value, colormap='magma'),
                global_step=i
            )

            # Estimated depth map: Normalized to the range 0 to 1.
            writer_obj.add_image(
                tag='{:s}_model_output/depth_normalized_0_to_1'.format(writer_prefix_tag),
                img_tensor=normalize_image(tgt_depth[0][0]),
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # 7) Histograms...
            # ----------------------------------------------------------------------------------------------------------

            # Histogram of the estimated disparity map.
            writer_obj.add_histogram(
                tag='{:s}_histograms/disparity'.format(writer_prefix_tag),
                values=1./tgt_depth[0][0],
                global_step=i
            )

            # Histogram of the estimated depth map.
            writer_obj.add_histogram(
                tag='{:s}_histograms/depth'.format(writer_prefix_tag),
                values=tgt_depth[0][0],
                global_step=i
            )

            # Flushes the event file to disk.
            writer_obj.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Compute losses...
        # --------------------------------------------------------------------------------------------------------------

        # Compute camera pose...
        poses, poses_inv = compute_pose_with_inv(
            pose_net,
            tgt_img,
            ref_imgs,
            verbose=False
        )

        # Compute photometric and geometry consistency losses...
        loss_1, loss_3, _ = compute_photo_and_geometry_loss(
            tgt_img=tgt_img,
            ref_imgs=ref_imgs,
            intrinsics=intrinsics,
            tgt_depth=tgt_depth,
            ref_depths=ref_depths,
            poses=poses,
            poses_inv=poses_inv,
            max_scales=cfgs["experiment_settings"]["num_scales"],
            with_ssim=cfgs["experiment_settings"]["with_ssim"],
            with_mask=cfgs["experiment_settings"]["with_mask"],
            with_auto_mask=False,
            padding_mode=cfgs["experiment_settings"]["padding_mode"],
            rotation_mode=cfgs["experiment_settings"]["rotation_matrix_mode"],
            velocity_supervision_loss_params=None,
            writer_obj_tag='{:s}'.format(writer_prefix_tag),
            writer_obj_step=i,
            writer_obj=writer_obj if enable_log_data else None,
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

        elapsed_time = time.time() - end
        batch_time.update(elapsed_time)
        end = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # Progress bar...
        # --------------------------------------------------------------------------------------------------------------

        batch_size = tgt_img.size()[0]

        pbar_str = \
            "[ Total frames = {:d} ][ Batch {:d} of size {:d} ][ Video clip indices = {} ][ Elapsed time = {:0.4f} ] " \
            "Losses: total_loss = {:0.4f}, photo_loss = {:0.4f}, smooth_loss = {:0.4f}, geom_loss = {:0.4f}".format(
                batch['total_frames'][0], i+1, batch_size, video_clip_indices_str, elapsed_time,
                total_loss, loss_1, loss_2, loss_3,
            )

        pbar.set_description(pbar_str)
        pbar.update(1)

        # --------------------------------------------------------------------------------------------------------------
        # Log poses to tensorboard.
        # --------------------------------------------------------------------------------------------------------------

        if enable_log_data:

            # ----------------------------------------------------------------------------------------------------------
            # 4) Training losses...
            # ----------------------------------------------------------------------------------------------------------

            writer_obj.add_scalar(
                tag='{:s}_loss/photometric_loss'.format(writer_prefix_tag),
                scalar_value=loss_1,
                global_step=i
            )

            writer_obj.add_scalar(
                tag='{:s}_loss/smoothness_loss'.format(writer_prefix_tag),
                scalar_value=loss_2,
                global_step=i
            )

            writer_obj.add_scalar(
                tag='{:s}_loss/geometry_consistency_loss'.format(writer_prefix_tag),
                scalar_value=loss_3,
                global_step=i
            )

            writer_obj.add_scalar(
                tag='{:s}_loss/total_loss'.format(writer_prefix_tag),
                scalar_value=total_loss,
                global_step=i
            )

            # ----------------------------------------------------------------------------------------------------------
            # Log poses to tensorboard.
            # ----------------------------------------------------------------------------------------------------------

            for img_idx, pose_data in enumerate(poses):

                pose_vector_str = ""

                # ------------------------------------------------------------------------------------------------------
                # Loop over the dimensions of the pose vector.
                # ------------------------------------------------------------------------------------------------------

                # Sample index.
                sample_idx = 0

                # Number of pose variables.
                num_pose_vars = pose_data.size()[1]

                for pose_idx in range(num_pose_vars):

                    # Pose variable value.
                    pose_var_value = pose_data[sample_idx, pose_idx].item()

                    # Pose variable name.
                    pose_var_name = pose_var_names_dict[pose_idx]

                    # --------------------------------------------------------------------------------------------------
                    # Log data in tensorboard as a scalar.
                    # --------------------------------------------------------------------------------------------------

                    # Show the components of the 6D pose vector in a plot.
                    writer_obj.add_scalar(
                        tag='{:s}_pose_target_wrt_ref{:d}/{:s}'.format(
                            writer_prefix_tag,
                            img_idx,
                            pose_var_name
                        ),
                        scalar_value=pose_var_value,
                        global_step=i
                    )

                    # Show the components of the 6D pose vector as a string.
                    if pose_idx <= 4:
                        pose_vector_str += "{:s} = {:0.4f}, ".format(pose_var_name, pose_var_value)
                    else:
                        pose_vector_str += "{:s} = {:0.4f}".format(pose_var_name, pose_var_value)

                # ------------------------------------------------------------------------------------------------------
                # Log data in tensorboard as text.
                # ------------------------------------------------------------------------------------------------------

                writer_obj.add_text(
                    tag='{:s}_pose_target_wrt_ref{:d}/pose_vector_6d'.format(writer_prefix_tag, img_idx),
                    text_string=pose_vector_str,
                    global_step=i
                )

            # Flushes the event file to disk.
            writer_obj.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Get poses data every iteration to be stored in a CSV file.
        # --------------------------------------------------------------------------------------------------------------

        # List of camera poses, i.e., target w.r.t. reference frames 0/1.
        camera_poses_data_list = []

        # Loop over pose data.
        for img_idx, pose_data in enumerate(poses):

            # Batch size.
            batch_size = pose_data.size()[0]

            # Number of pose variables.
            num_pose_vars = pose_data.size()[1]

            for sample_idx in range(batch_size):

                # Loop over the dimensions of the pose vector.
                for pose_idx in range(num_pose_vars):

                    # Pose variable value.
                    pose_var_value = pose_data[sample_idx, pose_idx].item()

                    # Pose variable name.
                    pose_var_name = pose_var_names_dict[pose_idx]

                    # Updating the list of camera poses.
                    # camera_poses_data_list.append("{:s}{:d}:{:0.6f}".format(pose_var_name, img_idx, pose_var_value))
                    camera_poses_data_list.append(pose_var_value)

        # --------------------------------------------------------------------------------------------------------------
        # Add rows to the CSV file.
        # --------------------------------------------------------------------------------------------------------------

        sample_idx = 0

        # Video information list (index, test drive id, camera video, video clip indices).
        video_info_list = [i, test_drive_id[sample_idx], camera_view[sample_idx], video_clip_indices_str]

        # Losses data list.
        losses_data_list = [
            total_loss,
            loss_1,
            loss_2,
            loss_3,
        ]

        # Telemetry data: Latitude and longitude.
        telemetry_data_list = [
            latitude_data[sample_idx],
            longitude_data[sample_idx]
        ]

        # Single row in the CSV file.
        single_row_csv_file = video_info_list + losses_data_list + camera_poses_data_list + telemetry_data_list

        with open('{}/{}'.format(save_path, cfgs["experiment_settings"]["log_full"]), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(single_row_csv_file)

        # --------------------------------------------------------------------------------------------------------------
        # Write frame to disk.
        # --------------------------------------------------------------------------------------------------------------

        # Output video frame. Change the dimensions order: TCHW --> THWC
        # output_rgb_frame = (variance_image * tgt_img.permute(0, 2, 3, 1) + mean_image) * 255.0

        sample_idx = 0

        # Output frame RGB.
        #   - Dimensions order:   012 --> 120
        #   - Permute dimensions: CHW --> HWC
        output_frame_rgb = (
            tensor2array(tgt_img[sample_idx]).transpose(1, 2, 0) * 255.0
        ).astype(np.uint8)

        # Output disparity frame RGB.
        #   - Dimensions order:   012 --> 120
        #   - Permute dimensions: CHW --> HWC
        output_disp_rgba = tensor2array(
            1. / tgt_depth[0][sample_idx], max_value=None, colormap='magma'
        ).transpose(1, 2, 0)

        assert output_disp_rgba.shape[2] == 4, \
            "[ Error ] The variable output_disp_rgba must have four channels at dimension 2. " \
            "Currently it has shape: {}".format(output_disp_rgba.shape)

        output_disparity_frame_rgb = (
                color.rgba2rgb(output_disp_rgba) * 255.0
            ).astype(np.uint8)

        assert output_disparity_frame_rgb.shape == output_frame_rgb.shape, \
            "[ Error ] The RGB and disparity images should have the same dimensions. " \
            "Currently: The RGB image shape is {} and the disparity image shape is {}.".format(
                output_frame_rgb.shape,
                output_disparity_frame_rgb.shape
            )

        # Concatenate the RGB and disparity frames along the width dimension (i.e., axis=1).
        output_rgb_disparity_frame = np.concatenate((output_frame_rgb, output_disparity_frame_rgb), axis=1)

        try:

            # Write frame on disk.
            video_writer_obj.writeFrame(output_rgb_disparity_frame[np.newaxis, :, :, :])

        except Exception as err:
            print(f"[ Exception ] Error while writing a frame with video_writer_obj.writeFrame(...): {err}")
            traceback.print_exc()
            print(" ")

        # --------------------------------------------------------------------------------------------------------------
        # Flushes the event file to disk.
        # --------------------------------------------------------------------------------------------------------------

        writer_obj.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Increase the counter.
        # --------------------------------------------------------------------------------------------------------------

        i += 1

    # ------------------------------------------------------------------------------------------------------------------
    # Close video writer object.
    # ------------------------------------------------------------------------------------------------------------------

    video_writer_obj.close()

    if verbose:
        print("[ Video writer ] Closing operation.")
        print(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Moving video data.
    # ------------------------------------------------------------------------------------------------------------------

    try:

        shutil.move(source_video_ffname, destiny_video_path)

        if verbose:
            print("[ Moving video data ]")
            print(f"\t- Source: {source_video_ffname}")
            print(f"\t- Destiny: {destiny_video_path}")
            print(" ")

    except Exception as err:
        print(f"[ Exception ] Error while moving video data from "
              f"source ({source_video_ffname}) to destiny ({destiny_video_path}): {err}")
        traceback.print_exc()
        print(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Deleting video data
    # ------------------------------------------------------------------------------------------------------------------

    if source_video_path.startswith(f"temp_videos/{source_video_random_id}"):
        shutil.rmtree(source_video_path)

        if verbose:
            print("[ Deleting the local directory to store video data ]")
            print(f"\t- Source path: {source_video_path}")
            print(" ")

    return losses.avg, [
        '{:s}_average_loss/total_loss'.format(writer_prefix_tag),
        '{:s}_average_loss/photometric_loss'.format(writer_prefix_tag),
        '{:s}_average_loss/smoothness_loss'.format(writer_prefix_tag),
        '{:s}_average_loss/geometry_consistency_loss'.format(writer_prefix_tag),
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
