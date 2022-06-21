import sys
sys.path.insert(1, '/home/arturo/workspace/pycharm_projects/data_loader_ml/DataLoaderML')
import wandb
import os
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
import toml
from pathlib import Path
from data_loader_ml.dataset import YaakIterableDataset
from data_loader_ml.tools.custom_transforms import Compose, TRANSFORM_DICT
from utils import tensor2array, save_checkpoint, count_parameters, print_batch, normalize_image
from utils import get_hyperparameters_dict
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(
    description='Model training and validation (SfM learner) on the Yaak dataset (with velocity supervision loss).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    default="configs/model_training/dynamic_scenes_single_scale_velsuploss_fstep_5.toml",
    type=Path,
    help="TOML configuration file to carry out model training and validation on the Yaak dataset.",
)
parser.add_argument(
    '-b', '--batch-size',
    default=4,
    type=int,
    help='Batch size.'
)
parser.add_argument(
    '--epochs',
    default=25,
    type=int,
    help='Total number of training epochs epochs.'
)
parser.add_argument(
    '--max-train-iterations',
    default=25,
    type=int,
    help='Max. number of iterations in the training set (per epoch).'
)
parser.add_argument(
    '--max-val-iterations',
    default=5,
    type=int,
    help='Max. number of iterations in the validation set.'
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
    experiment_name = None
    if cfgs["experiment_settings"]["freeze_disp_encoder_parameters"]:
        experiment_name = '{}/frozen_disp_encoder_params'.format(cfgs["experiment_settings"]["experiment_name"])
    else:
        experiment_name = '{}/optim_all_params'.format(cfgs["experiment_settings"]["experiment_name"])

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
    # Training and validation hyper-parameters.
    # ------------------------------------------------------------------------------------------------------------------

    wandb_hparams_dict = get_hyperparameters_dict(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_iterations=args.max_train_iterations,
        max_val_iterations=args.max_val_iterations,
        cfgs=cfgs,
        mode="train",
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
    tensorboard_train_log_dir = "{}/train".format(tensorboard_root_logdir)
    tensorboard_val_log_dir = "{}/val".format(tensorboard_root_logdir)

    # Training summary writer.
    training_writer = SummaryWriter(tensorboard_train_log_dir, flush_secs=10, max_queue=100)

    # Validation summary writer.
    num_output_writers = 1
    output_writers = []
    if cfgs["experiment_settings"]["log_output"]:
        for i in range(num_output_writers):
            output_writers.append(
                SummaryWriter('{}/{}'.format(tensorboard_val_log_dir, str(i)))
            )

    print('\n[ Initializing Tensorboard ]')
    print('\t- Train log path: {}'.format(tensorboard_train_log_dir))
    print('\t- Val log path: {}'.format(tensorboard_val_log_dir))

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations applied on data.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating training/validation data transformations ]")

    # Transformations applied on training data.
    train_transform_cfgs = cfgs["train"]["configuration"]["transformation"]["series"]
    train_transform = Compose(
        [TRANSFORM_DICT[fn](**kwargs) for fn, kwargs in train_transform_cfgs.items()]
    )

    # Transformations applied on validation data.
    val_transform_cfgs = cfgs["val"]["configuration"]["transformation"]["series"]
    val_transform = Compose(
        [TRANSFORM_DICT[fn](**kwargs) for fn, kwargs in val_transform_cfgs.items()]
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Train/validation dataset paths.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Train and validation dataset paths ]")
    print("\t- Training data: ")
    print("\t\t- Video sequences path: {}".format(cfgs["train"]["dataset"]["rootpath"]))
    print("\t\t- Camera calibration path: {}".format(cfgs["train"]["dataset"]["camera_calibration"]))
    print("\t- Validation data: ")
    print("\t\t- Video sequences path: {}".format(cfgs["val"]["dataset"]["rootpath"]))
    print("\t\t- Camera calibration path: {}".format(cfgs["val"]["dataset"]["camera_calibration"]))

    # --------------------------------------------------------------------------------------------------------------
    # Create the train dataset.
    # --------------------------------------------------------------------------------------------------------------

    print('\n[ Creating the train dataset using the YaakIterableDataset class ]')
    print('\t- Every video in the training set will be over-sampled by N = {} iterations.'.format(
            cfgs["train"]["configuration"]["sampling"]["oversampling"]
        )
    )
    print(' ')

    train_dataset = YaakIterableDataset(
        start=0,
        end=args.max_train_iterations * args.batch_size,
        dataset=cfgs["train"]["dataset"],
        config_frames=cfgs["train"]["configuration"]["frame"],
        config_sampling=cfgs["train"]["configuration"]["sampling"],
        config_returns=cfgs["train"]["configuration"]["return"],
        transform=train_transform,
        device_id=torch.cuda.current_device(),
        device_name="gpu" if enable_gpu else "cpu",
        verbose=True,
    )

    # --------------------------------------------------------------------------------------------------------------
    # Create the validation dataset.
    # --------------------------------------------------------------------------------------------------------------

    print('\n[ Creating the validation dataset using the YaakIterableDataset class ]')
    print('\t- Every video in the validation set will be over-sampled by N = {} iterations.'.format(
            cfgs["val"]["configuration"]["sampling"]["oversampling"]
        )
    )
    print(' ')

    val_dataset = YaakIterableDataset(
        start=0,
        end=args.max_val_iterations * args.batch_size,
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
    # Create train/validation data loaders.
    # ------------------------------------------------------------------------------------------------------------------

    print("\n[ Creating data loaders ]")
    print("\t- Creating a train data loader.")

    # Train loader.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfgs["experiment_settings"]["workers"],
        pin_memory=False,
    )

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

    if cfgs["experiment_settings"]["freeze_disp_encoder_parameters"]:

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
    # Create optimizers for the Disparity and Pose networks.
    # ------------------------------------------------------------------------------------------------------------------

    print('\n[ Creating optimizers for the Disparity and Pose networks ] Optimizer = Adam')

    # Selecting parameters to be optimized.
    optim_params = [
        {'params': disp_net.parameters(), 'lr': cfgs["experiment_settings"]["learning_rate"]},
        {'params': pose_net.parameters(), 'lr': cfgs["experiment_settings"]["learning_rate"]}
    ]

    # Optimizer...
    optimizer = torch.optim.Adam(
        optim_params,
        betas=(cfgs["experiment_settings"]["momentum"], cfgs["experiment_settings"]["beta"]),
        weight_decay=cfgs["experiment_settings"]["weight_decay"]
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Creating a CSV file to log data.
    # ------------------------------------------------------------------------------------------------------------------

    # Log summary file.
    log_summary_ffname = '{}/{}'.format(save_path, cfgs["experiment_settings"]["log_summary"])

    # Log full file.
    log_full_ffname = '{}/{}'.format(save_path, cfgs["experiment_settings"]["log_full"])

    print('\n[ Creating a CSV file to log data ]')
    print('\t- Log summary | File: {}'.format(log_summary_ffname))
    print('\t- Log full | File: {}'.format(log_full_ffname))

    with open(log_summary_ffname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(log_full_ffname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([
                'train_loss',
                'photometric_loss',
                'smoothness_loss',
                'geometry_consistency_loss',
                'velocity_supervision_loss'
            ]
        )

    ####################################################################################################################
    #
    # Initial model evaluation...
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the model on the validation set before training...
    # ------------------------------------------------------------------------------------------------------------------

    if cfgs["experiment_settings"]["initial_model_val_iterations"] > 0:

        print('\n[ Evaluating the model on the validation set before training... ]')
        print('[ Creating a logger for initial model evaluation) ]')

        logger_init = TermLogger(
            n_epochs=args.epochs,
            train_size=0,
            valid_size=cfgs["experiment_settings"]["initial_model_val_iterations"],
        )

        logger_init.reset_valid_bar()
        logger_init.valid_bar.update(0)

        for val_index in range(cfgs["experiment_settings"]["initial_model_val_iterations"]):

            errors, error_names = \
                validate_without_gt(
                    cfgs=cfgs,
                    val_loader=val_loader,
                    disp_net=disp_net,
                    pose_net=pose_net,
                    epoch=val_index,
                    max_iterations=1,
                    logger=logger_init,
                    train_writer=training_writer,
                    save_path=None,
                    output_writers=output_writers,
                    return_telemetry=cfgs["val"]["configuration"]["return"]["telemetry"],
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

        logger_init.valid_bar.update(cfgs["experiment_settings"]["initial_model_val_iterations"])

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
            cfgs=cfgs,
            train_loader=train_loader,
            disp_net=disp_net,
            pose_net=pose_net,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=args.batch_size,
            max_iterations=args.max_train_iterations,
            logger=logger,
            train_writer=training_writer,
            save_path=save_path,
            return_telemetry=cfgs["train"]["configuration"]["return"]["telemetry"],
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
                cfgs=cfgs,
                val_loader=val_loader,
                disp_net=disp_net,
                pose_net=pose_net,
                epoch=epoch,
                max_iterations=args.max_val_iterations,
                logger=logger,
                train_writer=training_writer,
                save_path=None,
                output_writers=output_writers,
                return_telemetry=cfgs["val"]["configuration"]["return"]["telemetry"],
                show_progress_bar=True,
                initial_model_evaluation=False,
                device=device,
                verbose=False,
            )

        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
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
            save_path,
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

    if cfgs["experiment_settings"]["wandb_enable"]:
        wandb.finish()


def train(
    cfgs,
    train_loader,
    disp_net,
    pose_net,
    optimizer,
    epoch,
    batch_size,
    max_iterations,
    logger,
    train_writer,
    save_path,
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
    total_loss_4 = 0.0

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

    # w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_loss_weight

    # Photometric loss weight.
    w1 = cfgs["experiment_settings"]["photo_loss_weight"]

    # Disparity smoothness loss weight.
    w2 = cfgs["experiment_settings"]["smooth_loss_weight"]

    # Geometry consistency loss weight.
    w3 = cfgs["experiment_settings"]["geometry_consistency_loss_weight"]

    # Velocity-scaling weight loss.
    w4 = cfgs["experiment_settings"]["velocity_scaling_weight"]

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

        log_losses = i > 0 and n_iter % cfgs["experiment_settings"]["print_freq"] == 0

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
            speed_data = batch['telemetry_data/speed'].to(device)

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
        # Velocity supervision loss parameters.
        # --------------------------------------------------------------------------------------------------------------

        # Condition to compute the velocity supervision loss.
        condition_loss_4 = (speed_data is not None) and (w4 > 0.)

        # Parameters to compute the velocity supervision loss.
        velocity_supervision_loss_params_dict = {
            # Conversion: Km/h -> m/s. Speed data is of size: [batch_size, 1]
            "speed_data": (1000. * speed_data.unsqueeze(-1)) / (60. ** 2),
            # Frame step.
            "frame_step": cfgs["train"]["configuration"]["frame"]["frame_step"],
            # Frames per second.
            "frame_fps": cfgs["train"]["configuration"]["frame"]["frame_fps"]
        } if condition_loss_4 else None

        # --------------------------------------------------------------------------------------------------------------
        # Computing the total loss and its components:
        #   - Photometric loss (loss_1).
        #   - Smooth loss (loss_2).
        #   - Geometry consistency loss (loss_3).
        #   - Velocity supervision loss (loss_4).
        # --------------------------------------------------------------------------------------------------------------

        loss_1, loss_3, loss_4 = \
            compute_photo_and_geometry_loss(
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
                with_auto_mask=cfgs["experiment_settings"]["with_auto_mask"],
                padding_mode=cfgs["experiment_settings"]["padding_mode"],
                rotation_mode=cfgs["experiment_settings"]["rotation_matrix_mode"],
                velocity_supervision_loss_params=velocity_supervision_loss_params_dict,
                writer_obj_tag='Train',
                writer_obj_step=epoch,
                writer_obj=train_writer if i + 1 == max_iterations else None,
                device=device
            )

        # Compute smooth loss for the disparity image (loss_2).
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        # Total loss.
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4

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
        # Update CSV file (i.e., total loss and its components):
        #
        #   - train_loss
        #   - photometric_loss
        #   - smoothness_loss
        #   - geometry_consistency_loss
        #   - velocity_supervision_loss
        #
        # --------------------------------------------------------------------------------------------------------------

        with open('{}/{}'.format(save_path, cfgs["experiment_settings"]["log_full"]), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([
                loss.item(),
                loss_1.item(),
                loss_2.item(),
                loss_3.item(),
                loss_4.item() if condition_loss_4 else loss_4
            ])

        # --------------------------------------------------------------------------------------------------------------
        # Update logger.
        # --------------------------------------------------------------------------------------------------------------

        # Record training loss and batch size.
        losses.update(loss.item(), batch_size)

        if show_progress_bar:
            logger.train_bar.update(i+1)

        if (i+1) % cfgs["experiment_settings"]["print_freq"] == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

        # --------------------------------------------------------------------------------------------------------------
        # Log losses per iteration...
        # --------------------------------------------------------------------------------------------------------------

        if log_losses:

            # Photometric loss (per N iterations).
            train_writer.add_scalar(
                tag='Train_loss_per_iter/photometric_loss',
                scalar_value=loss_1.item(),
                global_step=n_iter
            )

            # Smoothness loss (per N iterations).
            train_writer.add_scalar(
                tag='Train_loss_per_iter/smoothness_loss',
                scalar_value=loss_2.item(),
                global_step=n_iter
            )

            # Geometry consistency loss (per N iterations).
            train_writer.add_scalar(
                tag='Train_loss_per_iter/geometry_consistency_loss',
                scalar_value=loss_3.item(),
                global_step=n_iter
            )

            # Velocity supervision loss (per N iterations).
            train_writer.add_scalar(
                tag='Train_loss_per_iter/velocity_supervision_loss',
                scalar_value=loss_4.item() if condition_loss_4 else loss_4,
                global_step=n_iter
            )

            # Total loss (per N iterations).
            train_writer.add_scalar(
                tag='Train_loss_per_iter/total_loss',
                scalar_value=loss.item(),
                global_step=n_iter
            )

            # Flushes the event file to disk.
            train_writer.flush()

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

        if condition_loss_4:
            total_loss_4 += loss_4.item() / max_iterations
        else:
            total_loss_4 += loss_4 / max_iterations

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
                np.asarray(video_clip_indices) / float(cfgs["train"]["configuration"]["frame"]["frame_fps"])
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
            # 4-1) Training losses: Unscaled components.
            # ----------------------------------------------------------------------------------------------------------

            # Photometric loss.
            train_writer.add_scalar(
                tag='Train_loss/photometric_loss',
                scalar_value=total_loss_1,
                global_step=epoch
            )

            # Smoothness loss.
            train_writer.add_scalar(
                tag='Train_loss/smoothness_loss',
                scalar_value=total_loss_2,
                global_step=epoch
            )

            # Geometry consistency loss.
            train_writer.add_scalar(
                tag='Train_loss/geometry_consistency_loss',
                scalar_value=total_loss_3,
                global_step=epoch
            )

            # Velocity supervision loss.
            train_writer.add_scalar(
                tag='Train_loss/velocity_supervision_loss',
                scalar_value=total_loss_4,
                global_step=epoch
            )

            # Total training loss.
            train_writer.add_scalar(
                tag='Train_loss/total_loss',
                scalar_value=total_loss,
                global_step=epoch
            )

            # ----------------------------------------------------------------------------------------------------------
            # 4-2) Training losses: Scaled components.
            # ----------------------------------------------------------------------------------------------------------

            # Photometric loss.
            train_writer.add_scalar(
                tag='Train_loss_scaled_components/photometric_loss',
                scalar_value=w1 * total_loss_1,
                global_step=epoch
            )

            # Smoothness loss.
            train_writer.add_scalar(
                tag='Train_loss_scaled_components/smoothness_loss',
                scalar_value=w2 * total_loss_2,
                global_step=epoch
            )

            # Geometry consistency loss.
            train_writer.add_scalar(
                tag='Train_loss_scaled_components/geometry_consistency_loss',
                scalar_value=w3 * total_loss_3,
                global_step=epoch
            )

            # Velocity supervision loss.
            train_writer.add_scalar(
                tag='Train_loss_scaled_components/velocity_supervision_loss',
                scalar_value=w4 * total_loss_4,
                global_step=epoch
            )

            # Total training loss.
            train_writer.add_scalar(
                tag='Train_loss_scaled_components/total_loss',
                scalar_value=total_loss,
                global_step=epoch
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
    cfgs,
    val_loader,
    disp_net,
    pose_net,
    epoch,
    max_iterations,
    logger,
    train_writer,
    save_path,
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
            speed_data = batch['telemetry_data/speed'].to(device)

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

        if (i+1) % cfgs["experiment_settings"]["print_freq"] == 0:
            logger.valid_writer.write('Validation: Time {} Loss {}'.format(batch_time, losses))

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
        '{:s}_loss/smoothness_loss'.format(writer_prefix_tag),
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

    """ Compute pose data for the target image (tgt_img) w.r.t. one or more reference images (ref_imgs). """

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
