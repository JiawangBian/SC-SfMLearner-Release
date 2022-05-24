### Monocular depth estimation on the Yaak Dataset

#### Depth estimation technique

The monocular depth estimation technique described in [1,2] was adapted to the Yaak dataset.

References:
1. Bian, Jiawang, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. "Unsupervised scale-consistent depth and ego-motion learning from monocular video." Advances in neural information processing systems 32 (2019).
2. Bian, Jia-Wang, Huangying Zhan, Naiyan Wang, Zhichao Li, Le Zhang, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. "Unsupervised scale-consistent depth learning from video." International Journal of Computer Vision 129, no. 9 (2021): 2548-2564.

#### Dependencies

This code was tested on `Python 3.8.13` and `Pytorch 1.11.0` (with `CUDA 11.3` and `CUDNN 8.2.0_0` support).
All the dependencies to run this code are provided in the file `environment.yml`

#### Dataset

##### Description

The Yaak dataset is located in `/nas/drives/yaak/yaak_dataset/video_data/` and it consists videos sequences 
(stored in MP4 format) and telemetry data (i.e., stored in `metadata.log` and `metadata.json`) recorded during each test drive 
(e.g., `test_drive_id`). Thus, data is organized as shown below:

```

    ├── test_drive_id
        ├── cam_front_center-force-key.defish.mp4
        ├── cam_front_center.defish.mp4
        ├── cam_front_center.mp4
        ├── cam_front_left.mp4
        ├── cam_front_right.mp4
        ├── cam_left_backward-force-key.defish.mp4
        ├── cam_left_backward.defish.mp4
        ├── cam_left_backward.mp4
        ├── cam_left_forward.mp4
        ├── cam_rear.defish.mp4
        ├── cam_rear.mp4
        ├── cam_right_backward-force-key.defish.mp4
        ├── cam_right_backward.defish.mp4
        ├── cam_right_backward.mp4
        ├── cam_right_forward.mp4
        ├── metadata.json
        └── metadata.log

```

The telemetry data is stored originally in binary files, i.e., `metadata.log`. Subsequently, it was converted 
to the `JSON` format, resulting in the file `metadata.json`.

##### Data acquisition: Sampling period and frequency

Data samples were recorded at different frequencies (depending on the source of data and sensors used) as shown below:

- GPS: GNSS data every 100ms / 10Hz
- Doors: Vehicle state messages every 100ms / 10hz
- Speed: Vehicle motion every 20ms / 50Hz
- Videos: Camera messages every 33.333ms / 30Hz

##### Notes

- The video sequences filenames with the suffix `-force-key.defish` correspond to undistorted video sequences, with corrected key frames. Such data is used to train the current models.
- Speed data is loaded from the `metadata.json` file. This data is used for removing static scenes from the video sequences (currently, work in progress).

#### Train the model

Before training the model, adjust model hyper-parameters and other options in the bash file `train_resnet50_depth.sh`.
Also, you must specify where the dataset is located and where the experimental results will be stored in this script.
Afterwards, you can run the bash script as `bash scripts/train_resnet50_depth.sh`.

The contents of the bash file are shown below:

```
    # ------------------------------------------------------------------------------------------------------------------
    # Data paths...
    # ------------------------------------------------------------------------------------------------------------------
    
    # Path where dataset is stored. It must be defined by the user.
    # Default dataset location: "/nas/drives/yaak/yaak_dataset".
    DATA_ROOT="/nas/drives/yaak/yaak_dataset"
    
    # Path where experimental results will be stored. It must be defined by the user.
    EXP_RESULTS_ROOT="/some_path/my_experiments"
    
    # Create the experimental results path.
    mkdir -p "$EXP_RESULTS_ROOT"
    
    # Path where video sequences are stored.
    VIDEO_DATASET_PATH="$DATA_ROOT/video_data"
    
    # Path where the camera calibration data is stored (e.g., camera intrinsics).
    CAMERA_CALIB_PATH="$DATA_ROOT/camera_calibration"
    
    # ------------------------------------------------------------------------------------------------------------------
    # Hyper-parameters.
    # ------------------------------------------------------------------------------------------------------------------
    
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
    
    # Video clip step (i.e., number of frames to skip between consecutive frames).
    VIDEO_CLIP_STEP=15
    
    # Frame scaling factor use: 0.25, 0.5, 0.75 or 1.0.
    FRAME_SCALING_FACTOR=0.25
    
    # Number of scales.
    NUM_SCALES=1
    
    # Rotation matrix representation.
    ROTATION_MATRIX_MODE="euler"
    
    # Padding mode.
    PADDING_MODE="zeros"
    
    # Print results frequency...
    PRINT_FREQ=25
    
    # ------------------------------------------------------------------------------------------------------------------
    # Camera views used for model training/validation.
    # ------------------------------------------------------------------------------------------------------------------
    
    # Camera view used as part of the training set.
    CAMERA_VIEW_TRAIN="cam_front_center"
    
    # Camera view as part of the validation set.
    CAMERA_VIEW_VAL="cam_front_center"
    
    # ------------------------------------------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------------------------------------------
    
    # Test drive ID used as training set.
    TEST_DRIVE_ID_TRAIN="2021-08-13--09-27-11"
    
    # Test drive ID used as validation set.
    TEST_DRIVE_ID_VAL="2021-08-13--09-27-11"
    
    # ------------------------------------------------------------------------------------------------------------------
    # Experiments name, checkpoints path...
    # ------------------------------------------------------------------------------------------------------------------
    
    # Experiment number.
    EXP_NUMBER=0
    
    # Experiment name.
    EXPERIMENT_NAME="exp""$EXP_NUMBER""_epochs_""$EPOCHS""_numScales_$NUM_SCALES""_frameScaleFactor_""$FRAME_SCALING_FACTOR"
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
    
    # ------------------------------------------------------------------------------------------------------------------
    # Train the network on the Yaak dataset.
    # ------------------------------------------------------------------------------------------------------------------
    
    # Set the variable CUDA_VISIBLE_DEVICES to any GPU device (default is 0).
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

```

#### Possible issues during training.

If an EOF error related to the `Decord` library (which is used to load video sequences directly into the GPU) occur, you may need to increase the value of the environment variable 
`DECORD_EOF_RETRY_MAX`. For instance: `export DECORD_EOF_RETRY_MAX=200000480`. For more information about the issue, see the link below:

- [Decord issues] (https://github.com/dmlc/decord/issues/156)

#### Tensorboard visualization

To monitor the model training and validation use `Tensorboard`.

```
    tensorboard --logdir="/some_path/my_experiments/" --port=6006 --samples_per_plugin images=10000 
```

Note: Use the flag `--samples_per_plugin images=10000` to display results. Otherwise, information may be incomplete.

      

