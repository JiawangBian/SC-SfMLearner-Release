# SC-SfMLearner

This codebase implements the system described in the paper:

 >Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video
 >
 >[Jia-Wang Bian](https://jwbian.net/), Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, Ian Reid
 >
 >**NeurIPS** 2019 [[PDF](http://papers.nips.cc/paper/8299-unsupervised-scale-consistent-depth-and-ego-motion-learning-from-monocular-video)] [[Project webpage](https://jwbian.net/sc-sfmlearner/)]

## Depth and point cloud visulization on KITTI-09 (trained on 00-08)

[![depth visualization](https://img.youtube.com/vi/OkfK3wmMnpo/0.jpg)](https://www.youtube.com/watch?v=OkfK3wmMnpo)

## Dense reconstruction (left) using the estimated depth map (bottom right)

[![reconstruction demo](https://jwbian.net/Data/reconstruction.png)](https://www.youtube.com/watch?v=i4wZr79_pD8)



## Core contributions
  1. A geometry consistency loss, which makes the predicted depths to be globally scale consistent.
  2. A self-discovered mask, which detects moving objects and occlusions effectively and efficiently.
  3. The scale-consistent predictions allow for doing Monocular Visual Odometry on long videos.



 ## If you find our work useful in your research please consider citing our paper:
 
    @inproceedings{bian2019depth,
      title={Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video},
      author={Bian, Jia-Wang and Li, Zhichao and Wang, Naiyan and Zhan, Huangying and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian},
      booktitle= {Thirty-third Conference on Neural Information Processing Systems (NeurIPS)},
      year={2019}
    }



## Updates (Compared with NeurIPS version)
Note that this is an updated and improved version, find the original version in 'Release / NeurIPS Version' for reproducing the results reported in our paper. Compared with NeurIPS version, we
(1) Change networks by using Resnet18 and Resnet50 pretrained model (on ImageNet) for depth and pose encoders.
(2) We add 'auto_mask' by Monodepth2 to remove stationary points.



## Preamble
This codebase was developed and tested with python 3.6, Pytorch 1.0.1, and CUDA 10.0 on Ubuntu 16.04. It is based on [Clement Pinard's SfMLearner implementation](https://github.com/ClementPinard/SfmLearner-Pytorch).



## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
torch >= 1.0.1
imageio
matplotlib
scipy
argparse
tensorboardX
blessings
progressbar2
path.py
```

It is also advised to have python3 bindings for opencv for tensorboard visualizations


## Preparing training data

See "scripts/run_prepare_data.sh".

For [KITTI Raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php), download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.

For [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) download the dataset with color images.



## Training

The "scripts" folder provides several examples for training and testing.

You can train the depth model on KITTI Raw by running
```bash
sh scripts/train_resnet18_depth_256.sh
```
or train the pose model on KITTI Odometry by running
```bash
sh scripts/train_resnet50_pose_256.sh
```
Then you can start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. 



## Evaluation

You can evaluate depth on Eigen's split by running
```bash
sh scripts/test_kitti_depth.sh
```
evaluate visual odometry by running
```bash
sh scripts/test_kitti_vo.sh
```
and visualize depth by running
```bash
sh scripts/run_inference.sh
```

## Pretrained Models

[Latest Models](https://1drv.ms/u/s!AiV6XqkxJHE2kxX_Gek5fEQvMGma?e=ZfrnbR)

To evaluate the [NeurIPS models](https://1drv.ms/u/s!AiV6XqkxJHE2kxSHVMYvo7DmGqNb?e=bg3tWg), please download the code from 'Release/NeurIPS version'.


### Depth Results (Updated version, KITTI raw dataset, using the Eigen's splits)

|   Models   | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-----------|-------|-------|-------|
| resnet18   | 0.119   | 0.858  | 4.949 | 0.197     | 0.863 | 0.957 | 0.981 |
| resnet50   | 0.115   | 0.814  | 4.705 | 0.191     | 0.873 | 0.960 | 0.982 |



### Visual Odometry Results (Updated version, KITTI odometry dataset, trained on 00-08)

|Metric               | Seq. 09 | Seq. 10 |
|---------------------|---------|---------|
|t_err (%)            | 7.31    | 7.79    |
|r_err (degree/100m)  | 3.05    | 4.90    | 



    
 ## Related projects
 
 * [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) (CVPR 2017, our baseline framework.)

 * [Depth-VO-Feat](https://github.com/Huangying-Zhan/Depth-VO-Feat) (CVPR 2018, trained on stereo videos for depth and visual odometry)
 
 * [DF-VO](https://github.com/Huangying-Zhan/DF-VO) (ICRA 2020, use scale-consistent depth with optical flow for more accurate visual odometry)
 
 * [Kitti-Odom-Eval-Python](https://github.com/Huangying-Zhan/kitti-odom-eval) (python code for kitti odometry evaluation)
 
