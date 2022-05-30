# ----------------------------------------------------------------------------------------------------------------------
# Increase value of DECORD_EOF_RETRY_MAX
# ----------------------------------------------------------------------------------------------------------------------

# Export decord variable to avoid issues..
export DECORD_EOF_RETRY_MAX=200000480

# ----------------------------------------------------------------------------------------------------------------------
# Data paths...
# ----------------------------------------------------------------------------------------------------------------------

# Path where datasets are stored.
DATA_ROOT="/nas/drives/yaak/yaak_dataset"

# Path where experimental results are stored.
EXP_RESULTS_ROOT="/nas/team-space/arturo/experimental_results/depth_estimation/sc_sfmlearner_yaak_data"

# Create the experimental results path.
mkdir -p "$EXP_RESULTS_ROOT"

# Path where video sequences are stored.
VIDEO_DATASET_PATH="$DATA_ROOT/video_data"

# Path where the camera calibration data is stored (e.g., camera intrinsics).
CAMERA_CALIB_PATH="$DATA_ROOT/camera_calibration"

# ----------------------------------------------------------------------------------------------------------------------
# Pre-trained model parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Path where the pretrained model parameters are stored on disk.
PRETRAINED_MODEL_PATH="/nas/team-space/arturo/pretrained_models/sc_sfmlearner_pretrained_models/resnet50_depth_256"

# Pre-trained Disparity network.
PRETRAINED_MODEL_DISP="$PRETRAINED_MODEL_PATH/dispnet_model_best.pth.tar"

# Pretrained Pose network.
PRETRAINED_MODEL_POSE="$PRETRAINED_MODEL_PATH/exp_pose_model_best.pth.tar"

# ----------------------------------------------------------------------------------------------------------------------
# Hyper-parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Seed number.
SEED_NUMBER=19052022

# Common training hyper-parameters.
EPOCHS=25
MAX_TRAIN_ITERATIONS=25
MAX_VAL_ITERATIONS=5
BATCH_SIZE=1
LR=1e-4
MOMENTUM=0.9
BETA=0.999
WEIGHT_DECAY=0

# Initial model validation iterations...
INITIAL_MODEL_VAL_ITERATIONS=5

# Weight losses...
PHOTO_LOSS_WEIGHT=1.0
SMOOTH_LOSS_WEIGHT=0.1
GEOM_CONSISTENCY_LOSS_WEIGHT=0.5

# Video clip step (i.e., number of frames to skip between consecutive frames).
VIDEO_CLIP_STEP=10

# Number of iterations to oversample each video in the training set.
OVERSAMPLING_ITERATIONS_TRAIN=1

# Number of iterations to oversample each video in the validation set.
OVERSAMPLING_ITERATIONS_VAL=1

# Frame scaling factor use: 0.25, 0.5, 0.75 or 1.0.
FRAME_SCALING_FACTOR=0.5

# Number of scales.
NUM_SCALES=1

# Rotation matrix representation. Use: "euler", "quat".
ROTATION_MATRIX_MODE="euler"

# Padding mode. Use any of: "zeros", "border", "reflection".
# This argument is passed to torch.nn.functional.grid_sample(...)
#
PADDING_MODE="border"

# Print results frequency...
PRINT_FREQ=25

# ----------------------------------------------------------------------------------------------------------------------
# Camera views used for model training/validation.
# ----------------------------------------------------------------------------------------------------------------------

# Camera view used as part of the training set.
CAMERA_VIEW_TRAIN="cam_front_center"

# Camera view as part of the validation set.
CAMERA_VIEW_VAL="cam_front_center"

# ----------------------------------------------------------------------------------------------------------------------
# Test drive IDs used for model training/validation.
#
#
#   Test drive ID             Is the video OK?
#
#   2021-08-13--09-27-11      OK
#   2021-09-10--13-58-21      The car is not moving most of the time.
#   2021-10-13--06-30-56      OK
#   2021-11-15--15-13-02      Although the car is moving most of the time, half of the scenes are very dark.
#   2022-01-25--14-54-13      OK
#   2022-01-28--13-15-40      OK
#   2022-03-28--09-40-59      Remove - Reason: Metadata is corrupted.
#
# Exclude:
#
# 1) No metadata.json:
#
#   2022-02-25--14-42-58
#   2022-03-11--14-36-45
#   2022-03-28--09-40-59
#   2022-04-07--11-15-26
#   2022-04-13--11-10-16
#
# 2) No videos sequences (cam_front_center-force-key.defish.mp4, cam_front_center.defish.mp4):
#
#   2022-04-12--09-58-33
#   2022-04-29--06-48-39
#   2022-05-04--13-16-38
#
# ----------------------------------------------------------------------------------------------------------------------

# Test drive ID used as training set.
# TEST_DRIVE_ID_TRAIN="2021-08-13--09-27-11, 2021-10-13--06-30-56, 2022-01-25--14-54-13, 2022-01-28--13-15-40"
TEST_DRIVE_ID_TRAIN="2021-08-13--09-27-11, 2021-10-13--06-30-56"
# TEST_DRIVE_ID_TRAIN="test_drive_ids/drive_ids_train.txt"

# Test drive ID used as validation set.
# TEST_DRIVE_ID_VAL="2021-08-13--09-27-11, 2021-10-13--06-30-56, 2022-01-25--14-54-13, 2022-01-28--13-15-40"
TEST_DRIVE_ID_VAL="2021-08-13--09-27-11"
# TEST_DRIVE_ID_VAL="test_drive_ids/drive_ids_val.txt"

# ----------------------------------------------------------------------------------------------------------------------
# Parameters to be used to load telemetry data from the JSON files.
# Setting return_telemetry to "False" will ignore the rest of the arguments, and thus, the telemetry data will not be returned.
# ----------------------------------------------------------------------------------------------------------------------

# Telemetry data parameters - Training stage.
TELEMETRY_DATA_PARAMS_TRAIN='{"minimum_speed": 8.0, "camera_view": "cam_rear", "return_telemetry": "True"}'

# Telemetry data parameters - Validation stage.
TELEMETRY_DATA_PARAMS_VAL='{"minimum_speed": 8.0, "camera_view": "cam_rear", "return_telemetry": "False"}'

# ----------------------------------------------------------------------------------------------------------------------
# Experiments name, checkpoints path...
# ----------------------------------------------------------------------------------------------------------------------

# Device ID
GPU_DEVICE_ID=1

# Experiment number.
EXP_NUMBER=0

# Experiment name.
EXPERIMENT_NAME="exp""$EXP_NUMBER"
EXPERIMENT_NAME+="_epochs_""$EPOCHS""_numScales_$NUM_SCALES""_frameScaleFactor_""$FRAME_SCALING_FACTOR"
EXPERIMENT_NAME+="_videoClipStep_""$VIDEO_CLIP_STEP""_batchSize_""$BATCH_SIZE""_rotMatrix_""$ROTATION_MATRIX_MODE"
EXPERIMENT_NAME+="_pad_""$PADDING_MODE""_overSamplingIterTrain_""$OVERSAMPLING_ITERATIONS_TRAIN"
EXPERIMENT_NAME+="_maxTrainIter_""$MAX_TRAIN_ITERATIONS""_maxValIter_""$MAX_VAL_ITERATIONS"

# Checkpoints path.
CHECKPOINTS_PATH="$EXP_RESULTS_ROOT/tests_20.5.2022/checkpoints_r50_depth_trained_on_29_videos_20.5.2022_02"

# Weights and biases project name.
# WANDB_PROJECT_NAME="Exp_""$EXP_NUMBER""_withOverSampling_updateAllParams_singleScale_26videos"
WANDB_PROJECT_NAME="None"

echo "[ Train the depth estimation model (resnet50_depth_256) on the Yaak dataset ]"
echo "- [ Model validation ] No ground-truth used for model validation (only losses)."
echo "- [ Video dataset ] Path: $VIDEO_DATASET_PATH"
echo "- [ Camera calibration dataset ] Path: $CAMERA_CALIB_PATH"
echo "- [ Checkpoints ] Path: $CHECKPOINTS_PATH"
echo "- [ W & B Project name ] $WANDB_PROJECT_NAME"
echo "- [ Camera view(s) | Model training ] $CAMERA_VIEW_TRAIN"
echo "- [ Camera view(s) | Model validation ] $CAMERA_VIEW_VAL"
echo "- [ Test drive ID(s) | Model training ] $TEST_DRIVE_ID_TRAIN"
echo "- [ Test drive ID(s) | Model validation ] $TEST_DRIVE_ID_VAL"
echo " "

# ----------------------------------------------------------------------------------------------------------------------
# Train the network on the Yaak dataset.
# ----------------------------------------------------------------------------------------------------------------------

# Set the variable CUDA_VISIBLE_DEVICES to any GPU device (default is 0).
CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID \
python train_model.py \
--dataset-path "$VIDEO_DATASET_PATH" \
--cam-calib-path "$CAMERA_CALIB_PATH" \
--dataset-name 'yaak' \
--camera-view-train "$CAMERA_VIEW_TRAIN" \
--camera-view-val "$CAMERA_VIEW_VAL" \
--test-drive-id-train "$TEST_DRIVE_ID_TRAIN" \
--test-drive-id-val "$TEST_DRIVE_ID_VAL" \
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
--video-clip-fps 30 \
--oversampling-iterations-train $OVERSAMPLING_ITERATIONS_TRAIN \
--oversampling-iterations-val $OVERSAMPLING_ITERATIONS_VAL \
--telemetry-data-params-train "$TELEMETRY_DATA_PARAMS_TRAIN" \
--telemetry-data-params-val "$TELEMETRY_DATA_PARAMS_VAL" \
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
--pretrained-disp "$PRETRAINED_MODEL_DISP" \
--pretrained-pose "$PRETRAINED_MODEL_POSE" \
--checkpoints-path "$CHECKPOINTS_PATH" \
--seed $SEED_NUMBER \
--experiment-name "$EXPERIMENT_NAME" \
--wandb-project-name "$WANDB_PROJECT_NAME" \
--initial-model-val-iterations $INITIAL_MODEL_VAL_ITERATIONS
# --use-mask-static-objects-train
# --freeze-disp-encoder-parameters