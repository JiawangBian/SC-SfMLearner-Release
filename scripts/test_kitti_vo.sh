DATASET_DIR=/media/bjw/Disk/Dataset/kitti_odom_test/sequences/
OUTPUT_DIR=vo_results/

POSE_NET=checkpoints/resnet50_pose_256/exp_pose_model_best.pth.tar

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'