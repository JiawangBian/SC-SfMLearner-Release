KITIT_VO=/media/bjw/Disk/Dataset/kitti_odom/

POSE_NET=~/Research/SC-Models/cs+k_pose.tar

python test_pose.py $POSE_NET \
--img-height 256 --img-width 832 \
--dataset-dir $KITIT_VO \
--sequences 09