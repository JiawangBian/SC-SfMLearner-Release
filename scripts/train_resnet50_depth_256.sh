DATA_ROOT=/media/bjw/Disk
TRAIN_SET=$DATA_ROOT/Dataset/kitti_256/
python train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--name resnet50_depth_256