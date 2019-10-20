TRAIN_SET=/media/bjw/Disk/Dataset/kitti_256/
python train.py $TRAIN_SET \
--pretrained-disp= \
--pretrained-pose= \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-mask \
--with-ssim \
--with-gt \
--name cs+k_resnet_256