DISP_NET=checkpoints/resnet18_depth_256/10-17-13:19/dispnet_model_best.pth.tar
# DISP_NET=checkpoints/resnet50_depth_256/10-17-14:46/dispnet_model_best.pth.tar

DATA_ROOT=/media/bjw/Disk/Dataset/kitti_raw/
RESULTS_DIR=results/test/
TEST_FILE=kitti_eval/test_files_eigen.txt

# test big images
python test_disp.py --resnet-layers 18 --img-height 256 --img-width 832 \
--pretrained-dispnet $DISP_NET --dataset-dir $DATA_ROOT --dataset-list $TEST_FILE \
--output-dir $RESULTS_DIR


# evaluate
python2 ./kitti_eval/eval_depth.py --kitti_dir=$DATA_ROOT \
--test_file_list $TEST_FILE \
--pred_file=$RESULTS_DIR/predictions.npy