DISP_NET=checkpoints/resnet18_depth_256/dispnet_model_best.pth.tar
# DISP_NET=checkpoints/resnet50_depth_256/dispnet_model_best.pth.tar

DATA_ROOT=/media/bjw/Disk/Dataset/kitti_depth_test
RESULTS_DIR=results/test

# test
python test_disp.py --resnet-layers 18 --img-height 256 --img-width 832 \
--pretrained-dispnet $DISP_NET --dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR

# evaluate
python eval_depth.py \
--dataset kitti \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth