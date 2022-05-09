### Monocular depth estimation on the Yaak Dataset.

#### Depth estimation technique

- Bian, Jiawang, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, and Ian Reid. "Unsupervised scale-consistent depth and ego-motion learning from monocular video." Advances in neural information processing systems 32 (2019).

#### Dependencies

All the dependencies to run this code are provided in the file `environment.yml`

#### Dataset

The Yaak dataset is located in: `/nas/drives/yaak/yaak_dataset/video_data/`

#### Train the model.

To run a simple experiment run the bash script: `bash scripts/train_resnet_depth.sh`

Adjust model hyperparameters, input paths and other options in the bash file `train_resnet_depth.sh`.



      

