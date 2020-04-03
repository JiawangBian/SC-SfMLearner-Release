INPUT_DIR=/media/bjw/Disk/Dataset/kitti_odometry/sequences/09/image_2
OUTPUT_DIR=results/
DISP_NET=checkpoints/resnet18_depth_256/dispnet_model_best.pth.tar

python3 run_inference.py --pretrained $DISP_NET --resnet-layers 18 \
--dataset-dir $INPUT_DIR --output-dir $OUTPUT_DIR --output-disp