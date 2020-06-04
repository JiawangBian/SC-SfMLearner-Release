import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import models
from inverse_warp import pose_vec2mat


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=5)
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

    weights = torch.load(args.pretrained_posenet)
    seq_length = 5
    pose_net = models.PoseResNet(18, False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h, w, _ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

        squence_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.45)/0.225).to(device)
            squence_imgs.append(img)

        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])

        for iter in range(seq_length - 1):
            pose = pose_net(squence_imgs[iter], squence_imgs[iter + 1])
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :])

        final_poses = np.stack(poses, axis=0)

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        pose = pose_net(tgt_img, ref_img).unsqueeze(1)
        poses.append(pose)
    poses = torch.cat(poses, 1)
    return poses


if __name__ == '__main__':
    main()
