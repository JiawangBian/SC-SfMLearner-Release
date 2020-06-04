DISPNET=checkpoints/r18_rectified_nyu/dispnet_model_best.pth.tar

DATA_ROOT=/media/bjw/Disk/Dataset/nyu_test
RESULTS_DIR=results/nyu_test/

#  test 256*320 images
python test_disp.py --resnet-layers 18 --img-height 256 --img-width 320 \
--pretrained-dispnet $DISPNET --dataset-dir $DATA_ROOT/color \
--output-dir $RESULTS_DIR

# evaluate
python eval_depth.py \
--dataset nyu \
--pred_depth=$RESULTS_DIR/predictions.npy \
--gt_depth=$DATA_ROOT/depth.npy \
--img_dir $DATA_ROOT/color --vis_dir $RESULTS_DIR