TRAIN_SET=/media/bjw/Disk/Dataset/cs_256/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-mask \
--with-ssim \
--name cs_resnet_256