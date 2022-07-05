### Monocular depth estimation on the Yaak Dataset

#### Depth estimation technique

The monocular self-supervised depth estimation technique described in [1,2] was adapted to the Yaak dataset.
To estimate the metric scale (e.g., in meters), a speed-based regulariser was added in the loss function (during training) as suggested in [3] (currently, work in progress). 

References:
1. Bian, Jiawang, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. "Unsupervised scale-consistent depth and ego-motion learning from monocular video." Advances in neural information processing systems 32 (2019).
2. Bian, Jia-Wang, Huangying Zhan, Naiyan Wang, Zhichao Li, Le Zhang, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. "Unsupervised scale-consistent depth learning from video." International Journal of Computer Vision 129, no. 9 (2021): 2548-2564.
3. Guizilini, Vitor, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. "3d packing for self-supervised monocular depth estimation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2485-2494. 2020.

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

#### Model training and validation

To train and validate the model, specify all the paths (e.g., dataset path, output path, etc.) and model hyper-parameters in `configs/config_training.toml`.
Afterwards, execute the python script `train_model.py` as shown below:

```
    # Use this line to avoid any EOF decord issues.
    export DECORD_EOF_RETRY_MAX=200000480
    
    # Then execute the Python script.
    CUDA_VISIBLE_DEVICES=0 python train_model.py \
    --config configs/config_training.toml \
    --batch-size 1 \
    --epochs 250 \
    --max-train-iterations 1250 \
    --max-val-iterations 125 \
    --gpu

```

#### Model inference

To carry out model inference, specify all the paths (e.g., dataset path, output path, etc.) and model hyper-parameters in `configs/config_inference.toml`.
Afterwards, execute the Python script `run_inference.py` as shown below:

```

    # Use this line to avoid any EOF decord issues.
    export DECORD_EOF_RETRY_MAX=200000480
    
    # Then execute the Python script.
    CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --config configs/config_inference.toml \
    --batch-size 1 \
    --max-val-iterations -1 \
    --gpu

```

Passing the value `-1` to `max-val-iterations` will perform inference on the whole video.

#### Possible issues during training

If an EOF error related to the `Decord` library (which is used to load video sequences directly into the GPU) occur, you may need to increase the value of the environment variable 
`DECORD_EOF_RETRY_MAX`. For instance: `export DECORD_EOF_RETRY_MAX=200000480`. For more information about the issue, see the link below:

- [Decord issues] (https://github.com/dmlc/decord/issues/156)

#### Tensorboard visualization

To monitor the model training and validation use `Tensorboard`.

```
    tensorboard --logdir="/some_path/my_experiments/" --port=6006 --samples_per_plugin images=10000 
```

Note: Use the flag `--samples_per_plugin images=10000` to display results. Otherwise, information may be incomplete.

      

