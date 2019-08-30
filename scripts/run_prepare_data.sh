# for kitti raw dataset
DATASET=/media/bjw/Disk/Dataset/kitti_raw/
TRAIN_SET=/media/bjw/Disk/Dataset/kitti_256/
STATIC_FILES=data/static_frames.txt
python data/prepare_train_data.py $DATASET --dataset-format 'kitti_raw' --dump-root $TRAIN_SET --width 832 --height 256 --num-threads 4 --static-frames $STATIC_FILES --with-depth 

# # for cityscapes dataset
# DATASET=/media/bjw/Disk/Dataset/cityscapes/
# TRAIN_SET=/media/bjw/Disk/Dataset/cs_256/
# python data/prepare_train_data.py $DATASET --dataset-format 'cityscapes' --dump-root $TRAIN_SET --width 832 --height 342 --num-threads 4

# # for kitti odometry dataset
# DATASET=/media/bjw/Disk/Dataset/kitti_odom/
# TRAIN_SET=/media/bjw/Disk/Dataset/kitti_vo_256/
# python data/prepare_train_data.py $DATASET --dataset-format 'kitti_odom' --dump-root $TRAIN_SET --width 832 --height 256 --num-threads 4