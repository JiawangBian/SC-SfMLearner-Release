import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm
from path import Path

################### Options ######################
parser = argparse.ArgumentParser(description="NYUv2 Depth options")
parser.add_argument("--dataset", required=True, help="kitti or nyu", choices=['nyu', 'kitti'], type=str)
parser.add_argument("--pred_depth", required=True, help="depth predictions npy", type=str)
parser.add_argument("--gt_depth", required=True, help="gt depth nyu for nyu or folder for kitti", type=str)
parser.add_argument("--vis_dir", help="result directory for saving visualization", type=str)
parser.add_argument("--img_dir", help="image directory for reading image", type=str)
parser.add_argument("--ratio_name", help="names for saving ratios", type=str)

######################################################
args = parser.parse_args()


def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    Args:
        path: directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    if args.dataset == 'nyu':
        return abs_rel, log10, rmse, a1, a2, a3
    elif args.dataset == 'kitti':
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_depth = 1 / (data + 1e-6)
    vmax = np.percentile(inv_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data


def depth_pair_visualizer(pred, gt):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_pred = 1 / (pred + 1e-6)
    inv_gt = 1 / (gt + 1e-6)

    vmax = np.percentile(inv_gt, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)

    return vis_pred, vis_gt


class DepthEvalEigen():
    def __init__(self):

        self.min_depth = 1e-3

        if args.dataset == 'nyu':
            self.max_depth = 10.
        elif args.dataset == 'kitti':
            self.max_depth = 80.

    def main(self):
        pred_depths = []

        """ Get result """
        # Read precomputed result
        pred_depths = np.load(os.path.join(args.pred_depth))

        """ Evaluation """
        if args.dataset == 'nyu':
            gt_depths = np.load(args.gt_depth)
        elif args.dataset == 'kitti':
            gt_depths = []
            for gt_f in sorted(Path(args.gt_depth).files("*.npy")):
                gt_depths.append(np.load(gt_f))

        pred_depths = self.evaluate_depth(gt_depths, pred_depths, eval_mono=True)

        """ Save result """
        # create folder for visualization result
        if args.vis_dir:
            save_folder = Path(args.vis_dir)/'vis_depth'
            mkdir_if_not_exists(save_folder)

            image_paths = sorted(Path(args.img_dir).files('*.png'))

            for i in tqdm(range(len(pred_depths))):
                # reading image
                img = cv2.imread(image_paths[i], 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                h, w, _ = img.shape

                cat_img = 0
                if args.dataset == 'nyu':
                    cat_img = np.zeros((h, 3*w, 3))
                    cat_img[:, :w] = img
                    pred = pred_depths[i]
                    gt = gt_depths[i]
                    vis_pred, vis_gt = depth_pair_visualizer(pred, gt)
                    cat_img[:, w:2*w] = vis_pred
                    cat_img[:, 2*w:3*w] = vis_gt
                elif args.dataset == 'kitti':
                    cat_img = np.zeros((2*h, w, 3))
                    cat_img[:h] = img
                    pred = pred_depths[i]
                    vis_pred = depth_visualizer(pred)
                    cat_img[h:2*h, :] = vis_pred

                # save image
                cat_img = cat_img.astype(np.uint8)
                png_path = os.path.join(save_folder, "{:04}.png".format(i))
                cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

    def evaluate_depth(self, gt_depths, pred_depths, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths (NxHxW): gt depths
            pred_depths (NxHxW): predicted depths
            split (str): data split for evaluation
                - depth_eigen
            eval_mono (bool): use median scaling if True
        """
        errors = []
        ratios = []
        resized_pred_depths = []

        print("==> Evaluating depth result...")
        for i in tqdm(range(pred_depths.shape[0])):
            if pred_depths[i].mean() != -1:
                gt_depth = gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

                # resizing prediction (based on inverse depth)
                pred_inv_depth = 1 / (pred_depths[i] + 1e-6)
                pred_inv_depth = cv2.resize(pred_inv_depth, (gt_width, gt_height))
                pred_depth = 1 / (pred_inv_depth + 1e-6)

                mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
                if args.dataset == 'kitti':
                    gt_height, gt_width = gt_depth.shape
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                val_pred_depth = pred_depth[mask]
                val_gt_depth = gt_depth[mask]

                # median scaling is used for monocular evaluation
                ratio = 1
                if eval_mono:
                    ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                    ratios.append(ratio)
                    val_pred_depth *= ratio
                    # val_pred_depth *= 31.289

                resized_pred_depths.append(pred_depth * ratio)

                val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
                val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

                errors.append(compute_depth_errors(val_gt_depth, val_pred_depth))

        if eval_mono:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
            print(" Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))
            if args.ratio_name:
                np.savetxt(args.ratio_name, ratios, fmt='%.4f')

        mean_errors = np.array(errors).mean(0)

        if args.dataset == 'nyu':
            print("\n  " + ("{:>8} | " * 6).format("abs_rel", "log10", "rmse", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 6).format(*mean_errors.tolist()) + "\\\\")
        elif args.dataset == 'kitti':
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        return resized_pred_depths


eval = DepthEvalEigen()
eval.main()
