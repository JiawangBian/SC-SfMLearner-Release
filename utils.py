from __future__ import division
import shutil
import numpy as np
import torch
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
