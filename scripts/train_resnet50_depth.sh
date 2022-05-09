# ----------------------------------------------------------------------------------------------------------------------
# Data paths...
# ----------------------------------------------------------------------------------------------------------------------

# Path where dataset is stored, e.g., DATA_ROOT="/nas/drives/yaak/yaak_dataset"
DATA_ROOT="/nas/drives/yaak/yaak_dataset"

# Path where experimental results will be stored.
EXP_RESULTS_ROOT="/some_path/my_experiments"

# Create the experimental results path.
mkdir -p "$EXP_RESULTS_ROOT"

# Path where video sequences are stored.
VIDEO_DATASET_PATH="$DATA_ROOT/video_data"

# Path where the camera calibration data is stored (e.g., camera intrinsics).
CAMERA_CALIB_PATH="$DATA_ROOT/camera_calibration"

# ----------------------------------------------------------------------------------------------------------------------
# Hyper-parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Common training hyper-parameters.
EPOCHS=25
MAX_TRAIN_ITERATIONS=20
MAX_VAL_ITERATIONS=5
BATCH_SIZE=5
LR=1e-4
MOMENTUM=0.9
BETA=0.999
WEIGHT_DECAY=0

# Weight losses...
PHOTO_LOSS_WEIGHT=1.0
SMOOTH_LOSS_WEIGHT=0.1
GEOM_CONSISTENCY_LOSS_WEIGHT=0.5

# Video clip step, scaling factor, number of scales...
VIDEO_CLIP_STEP=15
FRAME_SCALING_FACTOR=0.25
NUM_SCALES=2
ROTATION_MATRIX_MODE="euler"
PADDING_MODE="zeros"

# Print results frequency...
PRINT_FREQ=25

# ----------------------------------------------------------------------------------------------------------------------
# Camera views used for model training/validation.
# ----------------------------------------------------------------------------------------------------------------------

CAMERA_VIEW_TRAIN="cam_front_center"
CAMERA_VIEW_VAL="cam_front_center"

# ----------------------------------------------------------------------------------------------------------------------
# Test drive IDs used for model training/validation.
#
#
# Test drive ID             Is the video OK?
#
# 2021-08-13--09-27-11      OK
# 2021-09-10--13-58-21      The car is not moving most of the time.
# 2021-10-13--06-30-56      OK
# 2021-11-15--15-13-02      Although the car is moving most of the time, half of the scenes are very dark.
# 2022-01-25--14-54-13      OK
# 2022-01-28--13-15-40      OK
# 2022-03-28--09-40-59      Remove - Reason: Metadata is corrupted.
#
# ----------------------------------------------------------------------------------------------------------------------

TEST_DRIVE_ID_TRAIN="2021-08-13--09-27-11"
TEST_DRIVE_ID_VAL="2021-08-13--09-27-11"

# ----------------------------------------------------------------------------------------------------------------------
# Experiments name, checkpoints path...
# ----------------------------------------------------------------------------------------------------------------------

# Experiment name.
EXPERIMENT_NAME="exp0_epochs_""$EPOCHS""_numScales_$NUM_SCALES""_frameScaleFactor_""$FRAME_SCALING_FACTOR"
EXPERIMENT_NAME+="_videoClipStep_""$VIDEO_CLIP_STEP""_batchSize_""$BATCH_SIZE""_rotMatrix_""$ROTATION_MATRIX_MODE""_pad_""$PADDING_MODE"

# Checkpoints path.
CHECKPOINTS_PATH="$EXP_RESULTS_ROOT/checkpoints"

echo "[ Train the depth estimation model (resnet50_depth_256) on the Yaak dataset ]"
echo "- [ Model validation ] No ground-truth used for model validation (only losses)."
echo "- [ Video dataset ] Path: $VIDEO_DATASET_PATH"
echo "- [ Camera calibration dataset ] Path: $CAMERA_CALIB_PATH"
echo "- [ Checkpoints ] Path: $CHECKPOINTS_PATH"
echo "- [ Camera view(s) | Model training ] $CAMERA_VIEW_TRAIN"
echo "- [ Camera view(s) | Model validation ] $CAMERA_VIEW_VAL"
echo "- [ Test drive ID(s) | Model training ] $TEST_DRIVE_ID_TRAIN"
echo "- [ Test drive ID(s) | Model validation ] $TEST_DRIVE_ID_VAL"
echo " "

# ----------------------------------------------------------------------------------------------------------------------
# Train the network on the yaak dataset.
# ----------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 \
python train_model.py \
--dataset-path "$VIDEO_DATASET_PATH" \
--cam-calib-path "$CAMERA_CALIB_PATH" \
--dataset-name 'yaak' \
--camera-view-train "$CAMERA_VIEW_TRAIN" \
--camera-view-val "$CAMERA_VIEW_VAL" \
--test-drive-id-train "$TEST_DRIVE_ID_TRAIN" \
--test-drive-id-val "$TEST_DRIVE_ID_VAL" \
--checkpoints-path "$CHECKPOINTS_PATH" \
--frame-scaling-factor $FRAME_SCALING_FACTOR \
--workers 0 \
--resnet-layers 50 \
--num-scales $NUM_SCALES \
--lr $LR --momentum $MOMENTUM --beta $BETA --weight-decay $WEIGHT_DECAY \
--batch-size $BATCH_SIZE \
--photo-loss-weight $PHOTO_LOSS_WEIGHT \
--smooth-loss-weight $SMOOTH_LOSS_WEIGHT \
--geometry-consistency-loss-weight $GEOM_CONSISTENCY_LOSS_WEIGHT \
--video-clip-length 3 \
--video-clip-step $VIDEO_CLIP_STEP \
--epochs $EPOCHS \
--max-train-iterations $MAX_TRAIN_ITERATIONS \
--max-val-iterations $MAX_VAL_ITERATIONS \
--rotation-matrix-mode "$ROTATION_MATRIX_MODE" \
--padding-mode "$PADDING_MODE" \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--print-freq $PRINT_FREQ \
--name "$EXPERIMENT_NAME"
