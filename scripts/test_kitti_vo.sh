DATASET_DIR=/media/bjw/Disk/Dataset/kitti_odom/sequences/
OUTPUT=vo_results/

POSE_NET=checkpoints/resnet50_pose_256/10-24-12:56/exp_pose_model_best.pth.tar

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT