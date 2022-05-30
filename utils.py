from __future__ import division
import os
import shutil
import numpy as np
import torch
import pandas as pd
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):

    tensor = tensor.detach().cpu()

    if max_value is None:
        max_value = tensor.max().item()

    if tensor.ndimension() == 2 or tensor.size(0) == 1:

        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:

        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225

    return array


def normalize_image(x):

    """

    Rescale image pixels to span range [0, 1].
    - From: https://github.com/nianticlabs/monodepth2.

    """

    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)

    d = ma - mi if ma != mi else 1e5

    return (x - mi) / d


def disp_to_depth(disp, min_depth, max_depth):

    """

    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.

    From: https://github.com/nianticlabs/monodepth2

    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

    return scaled_disp, depth


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):

    file_prefixes = ['dispnet', 'exp_pose']

    states = [dispnet_state, exp_pose_state]

    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, '{}/{}_{}'.format(save_path, prefix, filename))

    if is_best:
        for prefix in file_prefixes:

            shutil.copyfile(
                '{}/{}_{}'.format(save_path, prefix, filename),
                '{}/{}_model_best.pth.tar'.format(save_path, prefix)
            )


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_batch(batch_index, target_image, reference_images, intrinsics):

    print('[ Batch {:d} ]'.format(batch_index))

    print('\t[ Target images ] Shape = {} | Data-type = {} | Device = {} | Min, Max = {:0.4f}, {:0.4f}'.format(
            target_image.shape,
            target_image.dtype,
            target_image.device,
            target_image.min(),
            target_image.max(),
        )
    )

    print('\t[ Reference images ] N = {} images'.format(len(reference_images)))

    for ri_idx, ri in enumerate(reference_images):

        print('\t\t[ {} ] Shape = {} | Data-type = {} | Device = {} ls -'.format(
                ri_idx,
                ri.shape,
                ri.dtype,
                ri.device,
                ri.min(),
                ri.max(),
            )
        )

    print('\t[ Intrinsic matrix ] Shape = {} | Data-type = {} | Device = {}'.format(
            intrinsics.shape,
            intrinsics.dtype,
            intrinsics.device
        )
    )
    print(' ')


def get_hyperparameters_dict(
    device,
    epochs,
    batch_size,
    max_train_iterations,
    max_val_iterations,
    cfgs,
    save_path=None,
    verbose=True
):

    """
        Creates a dictionary of training and validation hyper-parameters.
        If save_path is provided, such data is stored in a CSV file.
    """

    hparams_list = [
        # Device (e.g., GPU ID).
        ('device', device),
        # Seed number.
        ('seed', cfgs["experiment_settings"]["seed"]),
        # Frame information: Training.
        ('train_frame_size', cfgs["train"]["configuration"]["frame"]["frame_size"]),
        ('train_frame_count', cfgs["train"]["configuration"]["frame"]["frame_count"]),
        ('train_frame_step', cfgs["train"]["configuration"]["frame"]["frame_step"]),
        # Frame information: Validation.
        ('val_frame_size', cfgs["val"]["configuration"]["frame"]["frame_size"]),
        ('val_frame_count', cfgs["val"]["configuration"]["frame"]["frame_count"]),
        ('val_frame_step', cfgs["val"]["configuration"]["frame"]["frame_step"]),
        # Sampling information: Training.
        ('train_sampling_speed', cfgs["train"]["configuration"]["sampling"]["speed"]),
        ('train_sampling_camera_view', cfgs["train"]["configuration"]["sampling"]["camera_view"]),
        ('train_sampling_oversampling', cfgs["train"]["configuration"]["sampling"]["oversampling"]),
        # Sampling information: Validation.
        ('val_sampling_speed', cfgs["val"]["configuration"]["sampling"]["speed"]),
        ('val_sampling_camera_view', cfgs["val"]["configuration"]["sampling"]["camera_view"]),
        ('val_sampling_oversampling', cfgs["val"]["configuration"]["sampling"]["oversampling"]),
        # Camera views: Training & Validation.
        ('train_camera_view', cfgs["train"]["dataset"]["camera_view"]),
        ('val_camera_view', cfgs["val"]["dataset"]["camera_view"]),
        # Drive ID: Training & Validation.
        ('train_drive_ids', cfgs["train"]["dataset"]["drive_ids"]),
        ('val_drive_ids', cfgs["val"]["dataset"]["drive_ids"]),
        # Common hyper-parameters.
        ('epochs', epochs),
        ('batch_size', batch_size),
        ('max_train_iterations', max_train_iterations),
        ('max_val_iterations', max_val_iterations),
        ('initial_model_val_iterations', cfgs["experiment_settings"]["initial_model_val_iterations"]),
        # Number of resnet layers.
        ('resnet_layers', cfgs["experiment_settings"]["resnet_layers"]),
        # Optimizer settings.
        ('learning_rate', cfgs["experiment_settings"]["learning_rate"]),
        ('momentum', cfgs["experiment_settings"]["momentum"]),
        ('beta', cfgs["experiment_settings"]["beta"]),
        ('weight_decay', cfgs["experiment_settings"]["weight_decay"]),
        # Number of scales to train the model.
        ('num_scales', cfgs["experiment_settings"]["num_scales"]),
        # Loss function....
        ('photo_loss_weight', cfgs["experiment_settings"]["photo_loss_weight"]),
        ('smooth_loss_weight', cfgs["experiment_settings"]["smooth_loss_weight"]),
        ('geometry_consistency_loss_weight', cfgs["experiment_settings"]["geometry_consistency_loss_weight"]),
        ('with_ssim', cfgs["experiment_settings"]["with_ssim"]),
        ('with_mask', cfgs["experiment_settings"]["with_mask"]),
        ('with_auto_mask', cfgs["experiment_settings"]["with_auto_mask"]),
        # Model parameters...
        ('with_pretrain', cfgs["experiment_settings"]["with_pretrain"]),
        ('pretrained_disp', cfgs["experiment_settings"]["pretrained_disp"]),
        ('pretrained_pose', cfgs["experiment_settings"]["pretrained_pose"]),
        ('freeze_disp_encoder_parameters', cfgs["experiment_settings"]["freeze_disp_encoder_parameters"]),
        # Rotation matrix and padding mode...
        ('rotation_matrix_mode', cfgs["experiment_settings"]["rotation_matrix_mode"]),
        ('padding_mode', cfgs["experiment_settings"]["padding_mode"])
    ]

    if save_path is not None:

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Create a data frame...
        hparams_df = pd.DataFrame(
            {
                'Name': [hparams_list[i][0] for i in range(len(hparams_list))],
                'Value': [hparams_list[i][1] for i in range(len(hparams_list))]
            }
        )

        # Saving the hyper_parameters in a CSV file...
        hparams_ffname = '{}/hyper_parameters.csv'.format(save_path)
        hparams_df.to_csv(hparams_ffname, index=False)

        if verbose:
            print('\n[ Training/validation hyper-parameters ]')
            print('\t- Saved in: {}'.format(hparams_ffname))

    return dict(hparams_list)
