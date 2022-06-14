from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch import linalg as LA
from inverse_warp import inverse_warp2
from utils import tensor2array

########################################################################################################################
#
# Structural Similarity Index Loss
#
########################################################################################################################


class SSIM(nn.Module):

    """
    Layer to compute the Structural Similarity Index (SSIM) loss between a pair of images.
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM()


########################################################################################################################
#
# Velocity supervision loss.
#
########################################################################################################################


def compute_velocity_supervision_loss(
    speed_data,
    poses,
    poses_inv,
    frame_step,
    frames_fps,
    convert_speed_kmph_to_mps,
):

    """

    Computes the velocity supervision loss, Lv, as suggested in [1]:

        Lv = | || x || - (s * dt) |

    Where:

        * | . |: is the L1 loss.
        * || x ||: is the L2-norm of the vector x (i.e., its magnitude).
        * x: It's the translation vector estimated by the pose network between target and reference frames.
        * s: It's the speed.
        * dt: It's the time difference between the target and reference frames.

    Notes:

        * The pose network estimates 6 DoF parameters in a single vector X = [ tx, ty, tz, rx, ry, rz].
        Here ti and ri are the translation and rotation parameters estimated between two consecutive frames,
        respectively. During training, this vector has size of [batch_size, 6].

    References:

        [1] Guizilini, Vitor, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon.
        "3d packing for self-supervised monocular depth estimation." In Proceedings of the IEEE/CVF
        Conference on Computer Vision and Pattern Recognition, pp. 2485-2494. 2020.

    """

    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

    # If convert_speed_kmph_to_mps is True, the vehicle's speed is converted from kilometers per hour (mk/h)
    # to meters per second (m/s). Otherwise, the current speed is kept (in km/h).
    scaled_speed_data = \
        (1000. * speed_data.unsqueeze(-1)) / (60. ** 2) \
        if convert_speed_kmph_to_mps else speed_data.unsqueeze(-1)

    # Sampling period (single frame) in seconds.
    sampling_period = 1. / float(frames_fps)

    # Total time difference between target and reference frames.
    delta_time = sampling_period * frame_step

    # Ground-truth distance estimated from speed data and the sampling period, between target and reference frames.
    gt_distance = scaled_speed_data * delta_time

    # Check the dimensions of the ground-truth distance.
    assert (len(gt_distance.shape) == 2) and (gt_distance.shape[1] == 1), \
        f"[ Error ] The ground-truth distance data must have two dimensions and the second one must be 1. " \
        f"Currently, it has a size of: {gt_distance.shape}"

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over pose data to get the translation vector between target w.r.t. reference frames.
    # Such data is used to compute the velocity supervision loss.
    # ------------------------------------------------------------------------------------------------------------------

    # Total loss.
    total_loss = 0.0

    # Count iterations.
    count_iterations = 0

    for pose, pose_inv in zip(poses, poses_inv):

        # --------------------------------------------------------------------------------------------------------------
        # Check the dimensions of the pose and ground-truth distance data.
        # --------------------------------------------------------------------------------------------------------------

        # Dimensions of the ground-truth distance and pose data (i.e., batch sizes must be the same).
        assert (gt_distance.shape[0] == pose.shape[0]) and (gt_distance.shape[0] == pose_inv.shape[0]), \
            f"[ Error ] The first dimension of ground-truth distance data must be equal to the batch size. " \
            f"Currently, it has a size of: {gt_distance.shape}"

        # Dimensions of the pose vector (i.e., its size must be [batch_size, 6]).
        assert (pose.shape[1] == 6) and (len(pose.shape) == 2), \
            f"[ Error ] The pose vector should have size of [batch_size, 6]. " \
            f"Currently, it has a size of: {pose.shape}"

        # Dimensions of the inverse pose vector (i.e., its size must be [batch_size, 6]).
        assert (pose_inv.shape[1] == 6) and (len(pose_inv.shape) == 2), \
            f"[ Error ] The inverse pose vector should have size of [batch_size, 6]. " \
            f"Currently, it has a size of: {pose_inv.shape}"

        # --------------------------------------------------------------------------------------------------------------
        # Get the translation vectors and compute their magnitude.
        # --------------------------------------------------------------------------------------------------------------

        # Translation vector (e.g., computed from pose_net(tgt_img, ref_img)) of size [batch_size, 3].
        x = pose[:, :3]

        # Inverse translation vector (e.g., computed from pose_net(ref_img, tgt_img) of size [batch_size, 3].
        x_inv = pose_inv[:, :3]

        # --------------------------------------------------------------------------------------------------------------
        # Compute the magnitude fo the translation vectors (L2-norm).
        # --------------------------------------------------------------------------------------------------------------

        # Predicted distance computed as the L2-norm of the translation vector.
        pred_distance = LA.vector_norm(x, ord=2, dim=-1)

        # Predicted distance computed as the L2-norm of the inverse translation vector.
        pred_distance_inv = LA.vector_norm(x_inv, ord=2, dim=-1)

        # --------------------------------------------------------------------------------------------------------------
        # Velocity supervision loss.
        # --------------------------------------------------------------------------------------------------------------

        # First component: Difference between L2-norm of the predicted translation vector and the
        # ground-truth distance.
        total_loss += (pred_distance - gt_distance).abs().mean()

        # Second component: Difference between L2-norm of the predicted inverse translation vector and the
        # ground-truth distance.
        total_loss += (pred_distance_inv - gt_distance).abs().mean()

        # --------------------------------------------------------------------------------------------------------------
        # Count iterations
        # --------------------------------------------------------------------------------------------------------------

        count_iterations += 1

    total_loss /= count_iterations

    return total_loss


########################################################################################################################
#
# Photometric and geometry consistency losses.
#
########################################################################################################################


def compute_photo_and_geometry_loss(
    tgt_img,
    ref_imgs,
    intrinsics,
    tgt_depth,
    ref_depths,
    poses,
    poses_inv,
    max_scales,
    with_ssim,
    with_mask,
    with_auto_mask,
    padding_mode,
    rotation_mode='euler',
    writer_obj_tag='photo_geom_loss',
    writer_obj_step=0,
    writer_obj=None,
    device=torch.device("cpu")
):

    """
        Computes photometric and geometry consistency losses.

        Parameters:
        -----------

        Returns:
        --------
            The photometric and geometry consistency losses (i.e., photo_loss, geometry_loss).
    """

    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over the data...
    # ------------------------------------------------------------------------------------------------------------------

    # Index for reference images.
    ref_img_idx = 0

    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):

        # --------------------------------------------------------------------------------------------------------------
        # Loop over scales.
        # --------------------------------------------------------------------------------------------------------------

        for s in range(num_scales):

            # ----------------------------------------------------------------------------------------------------------
            # Upsample depth maps.
            # ----------------------------------------------------------------------------------------------------------

            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics

            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='nearest')
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='nearest')

            # ----------------------------------------------------------------------------------------------------------
            # Show target and reference depth maps up to a scale.
            # ----------------------------------------------------------------------------------------------------------

            if writer_obj is not None and num_scales > 1:

                _, h_scaled, w_scaled = tgt_depth_scaled[0].size()

                writer_obj.add_image(
                    tag='{}_depth_scales/target_ref_depth_scaled_{}_{}x{}'.format(writer_obj_tag, s, h_scaled, w_scaled),
                    img_tensor=tensor2array(
                        torch.cat([tgt_depth_scaled[0], ref_depth_scaled[0]], dim=2),
                        max_value=None,
                        colormap='magma'
                    ),
                    global_step=writer_obj_step
                )

            # ----------------------------------------------------------------------------------------------------------
            # Compute losses.
            # ----------------------------------------------------------------------------------------------------------

            # Computing the pairwise loss: Target w.r.t. reference data.
            photo_loss1, geometry_loss1 = \
                compute_pairwise_loss(
                    tgt_img_scaled, ref_img_scaled,
                    tgt_depth_scaled, ref_depth_scaled,
                    pose, intrinsic_scaled,
                    with_ssim, with_mask, with_auto_mask,
                    padding_mode,
                    rotation_mode=rotation_mode,
                    writer_obj_tag='{}_pairwise_loss_target_wrt_ref{}'.format(writer_obj_tag, ref_img_idx),
                    writer_obj_step=writer_obj_step,
                    writer_obj=writer_obj,
                    device=device,
                )

            # Computing the pairwise loss: Target w.r.t. reference data reversed.
            photo_loss2, geometry_loss2 = \
                compute_pairwise_loss(
                    ref_img_scaled, tgt_img_scaled,
                    ref_depth_scaled, tgt_depth_scaled,
                    pose_inv, intrinsic_scaled,
                    with_ssim, with_mask, with_auto_mask,
                    padding_mode,
                    rotation_mode=rotation_mode,
                    writer_obj_tag='{}_pairwise_loss_target_wrt_ref{}_reversed'.format(writer_obj_tag, ref_img_idx),
                    writer_obj_step=writer_obj_step,
                    writer_obj=writer_obj,
                    device=device,
                )

            # Photometric loss.
            photo_loss += (photo_loss1 + photo_loss2)

            # Geometry consistency loss.
            geometry_loss += (geometry_loss1 + geometry_loss2)

        # --------------------------------------------------------------------------------------------------------------
        # Reference image index...
        # --------------------------------------------------------------------------------------------------------------

        ref_img_idx += 1

    return photo_loss, geometry_loss

########################################################################################################################
#
# Pair-wise loss.
#
########################################################################################################################


def compute_pairwise_loss(
    tgt_img,
    ref_img,
    tgt_depth,
    ref_depth,
    pose,
    intrinsic,
    with_ssim,
    with_mask,
    with_auto_mask,
    padding_mode,
    rotation_mode='euler',
    writer_obj_tag='pairwise_loss',
    writer_obj_step=0,
    writer_obj=None,
    device=torch.device("cpu"),
):

    # ------------------------------------------------------------------------------------------------------------------
    # Image synthesis...
    # ------------------------------------------------------------------------------------------------------------------

    ref_img_warped, valid_mask, projected_depth, computed_depth, pose_matrix_3x4 = \
        inverse_warp2(
            img=ref_img,
            depth=tgt_depth,
            ref_depth=ref_depth,
            pose=pose,
            intrinsics=intrinsic,
            padding_mode=padding_mode,
            rotation_mode=rotation_mode,
        )

    # Log data: Valid mask, reference image warped, and pose matrix...
    if writer_obj is not None:

        # --------------------------------------------------------------------------------------------------------------
        # Reference image warped...
        # --------------------------------------------------------------------------------------------------------------

        writer_obj.add_image(
            tag='{}/target_and_ref_image_warped'.format(writer_obj_tag),
            img_tensor=tensor2array(torch.cat([tgt_img[0], ref_img_warped[0]], dim=2)),
            global_step=writer_obj_step
        )

        # --------------------------------------------------------------------------------------------------------------
        # L1 loss between reference image warped...
        # --------------------------------------------------------------------------------------------------------------

        writer_obj.add_image(
            tag='{}/diff_target_and_ref_image_warped'.format(writer_obj_tag),
            img_tensor=tensor2array(torch.abs(tgt_img[0] - ref_img_warped[0])),
            global_step=writer_obj_step
        )

        # --------------------------------------------------------------------------------------------------------------
        # Valid mask...
        # --------------------------------------------------------------------------------------------------------------

        writer_obj.add_image(
            tag='{}/valid_mask'.format(writer_obj_tag),
            img_tensor=valid_mask[0],
            global_step=writer_obj_step
        )

        # --------------------------------------------------------------------------------------------------------------
        # Get translation vector and rotation matrix data.
        # --------------------------------------------------------------------------------------------------------------

        # Batch sample idx...
        sample_idx = 0

        # Translation vector.
        x = pose_matrix_3x4[sample_idx, 0, 3]
        y = pose_matrix_3x4[sample_idx, 1, 3]
        z = pose_matrix_3x4[sample_idx, 2, 3]

        # Rotation matrix elements: Row 0.
        r00 = pose_matrix_3x4[sample_idx, 0, 0]
        r01 = pose_matrix_3x4[sample_idx, 0, 1]
        r02 = pose_matrix_3x4[sample_idx, 0, 2]

        # Rotation matrix elements: Row 1.
        r10 = pose_matrix_3x4[sample_idx, 1, 0]
        r11 = pose_matrix_3x4[sample_idx, 1, 1]
        r12 = pose_matrix_3x4[sample_idx, 1, 2]

        # Rotation matrix elements: Row 2.
        r20 = pose_matrix_3x4[sample_idx, 2, 0]
        r21 = pose_matrix_3x4[sample_idx, 2, 1]
        r22 = pose_matrix_3x4[sample_idx, 2, 2]

        # String representation of the rotation matrix.
        rot_matrix_tvector_str = \
            'Rot = {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f} | {:0.4f}, {:0.4f}, {:0.4f};  ' \
            'X = {:0.4f}, {:0.4f}, {:0.4f}'.format(
                r00, r01, r02,
                r10, r11, r12,
                r20, r21, r22,
                x, y, z
            )

        writer_obj.add_text(
            tag='{}/rotation_matrix_translation_vector'.format(writer_obj_tag),
            text_string=rot_matrix_tvector_str,
            global_step=writer_obj_step
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Photometric and geometry consistency losses.
    # ------------------------------------------------------------------------------------------------------------------

    # Photometric loss.
    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    # Geometry consistency loss.
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    # Log data...
    if writer_obj is not None:
        writer_obj.add_image(
            tag='{}/diff_depth'.format(writer_obj_tag),
            img_tensor=diff_depth[0],
            global_step=writer_obj_step
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Auto mask...
    # ------------------------------------------------------------------------------------------------------------------

    if with_auto_mask:

        auto_mask =\
            (
                diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
            ).float() * valid_mask

        valid_mask = auto_mask

        # Log data...
        if writer_obj is not None:
            writer_obj.add_image(
                tag='{}/auto_mask'.format(writer_obj_tag),
                img_tensor=valid_mask[0],
                global_step=writer_obj_step
            )

    # ------------------------------------------------------------------------------------------------------------------
    # SSIM Loss...
    # ------------------------------------------------------------------------------------------------------------------

    if with_ssim:

        ssim_map = (compute_ssim_loss.to(device))(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    # ------------------------------------------------------------------------------------------------------------------
    # Weight mask...
    # ------------------------------------------------------------------------------------------------------------------

    if with_mask:

        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

        # Log data...
        if writer_obj is not None:
            writer_obj.add_image(
                tag='{}/weight_mask'.format(writer_obj_tag),
                img_tensor=weight_mask[0],
                global_step=writer_obj_step
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Flushes the event file to disk.
    # ------------------------------------------------------------------------------------------------------------------

    if writer_obj is not None:
        writer_obj.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Compute reconstruction and geometry consistency losses...
    # ------------------------------------------------------------------------------------------------------------------

    reconstruction_loss = mean_on_mask(diff_img, valid_mask, device=device)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask, device=device)

    return reconstruction_loss, geometry_consistency_loss

########################################################################################################################
#
# Smoothness loss...
#
########################################################################################################################


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):

    def get_smooth_loss(disp, img):

        """
        Computes the smoothness loss for a disparity image.
        The color image is used for edge-aware smoothness.
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss

########################################################################################################################
#
# Other losses...
#
########################################################################################################################


def mean_on_mask(diff, valid_mask, device=torch.device("cpu")):

    """ Compute mean value given a binary mask. """

    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)

    return mean_value


@torch.no_grad()
def compute_errors(gt, pred, dataset):

    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    """
    Crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    """

    if dataset == 'kitti':

        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':

        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):

        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

