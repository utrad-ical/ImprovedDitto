# Improving personalized federated learning to optimize site-specific performance in computer-aided detection/diagnosis [SPIE Journal of Medical Imaging]

this is an official implementation of ***Improved Ditto***. [[Link]]()

  - This is code for running federated learning (FL) in a local environment
  - The model used for trainig is Single Shot Multibox Detector(SSD) (ref: https://github.com/qfgaohao/pytorch-ssd)
  - FedAvg is used for medo aggregartion at the server
  - The deep learning library used is PyTorch, and Opacus is used for differential privacy (DP-SGD) function.
  - Since DP-SGD is used, GroupNorm is adopted instead of BatchNorm (to prevent normalization parameters from leaving the institution).

## Development Enviroment
In principle, a Docker image built from nvcr.io/nvidia/pytorch:22.11-py3 is used.

| enviroment | version |
| ---------- | ------- |
| python     | 3.8.10 |
| pytorch    | 1.13.0 (1.13.0a0+936e930) |
| opacus     | 1.4.0 |
| opt-einsum | 3.3.0 |
| CUDA       | 11.8.0 |
| cuDNN      | 8.7.0 |

### How to Build the Docker Image

A countermeasure is implemented to prevent generated files from having root privileges when the image is run (entrypoint.sh is used for this purpose).
  - Reference: https://qiita.com/yohm/items/047b2e68d008ebb0f001

        # cd build_env
        # (sudo) docker build -t 'tag_name:version' .

### Example of Running Training with the Generated Docker Image
        #(sudo) docker run --rm --gpus device=0 --shm-size=2g -it -v /home/username:/home/username -v /data/dir:/data/dir(:ro) -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -e USER_NAME='username' tag_name:version /path/to/bash_file
  - The options -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -e USER_NAME='username' are required.
  - Sometimes shared memory errors occur, so --shm-size=2g is added (2G is empirically set and can be adjusted).
  - Since the source code is divided by training type, it is recommended to create a script file for each training execution. (Example below is for centralized random search; details described later.)

        #!/bin/bash
        cd /path/to/ImprovedDitto/federated

        python fl_training.py /path/to/config/fl_config.yaml


## Centralized Training
use /centralized/centralized_train.py

### How to Run
        cd /path/to/ImprovedDitto/centralized

        python centralized_train.py training_data_path validation_data_path output_path\
        -g [gpu id]\
        -r [learning rate]\
        -b [batch size]\
        -m [max epochs]\
        -e [early stop patience epoch]\
        -a [No. of data augmentation]\
        --time_stamp [time stamp or output folder name for data saving]

## Federated Learning (FedAvg)
### How to run

        cd /path/to/ImprovedDitto/federated

        python fl_training.py [path/of/fl_config.yaml](https://github.com/utrad-ical/ImprovedDitto/blob/main/config/federated/fl_config.yaml.sample)

## Ditto
### How to run
        cd /path/to/ImprovedDitto/federated

        python ditto_train.py [path/of/ditto_config.yaml](https://github.com/utrad-ical/ImprovedDitto/blob/main/config/federated/ditto_config.yaml.sample)

## ImprovedDitto
### How to run
        cd /path/to/ImprovedDitto/federated

        python ditto_train.py [path/of/ImprovedDitto_config.yaml](https://github.com/utrad-ical/ImprovedDitto/blob/main/config/federated/ImprovedDitto_config.yaml.sample)

## parameter setting in yaml
### for FL (FedAvg)
        data_list:
        training_data:
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/GE/training
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/Philips/training
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/Siemens/training
        validation_data:
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/GE/validation
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/Philips/validation
            - /mnt/nas-public/OpenBTAI/OpenBTAI_case_list_20240424/Siemens/validation

        train_parameter:
            learning_rate: 0.01
            batch_size: 12
            max_round: 10
            early_stopping: 0
            data_augmentation_num: 2

        fl_parameter:
            client_names_list:
                - GE
                - Philips
                - Siemens
            client_num: 3
            epoch_per_round: 1 

        dp_parameter:
            do_dp: False (Fixed)
            dp_secure_mode: False
            dp_max_grad: 1
            noise_multiplier: 0.1
            dp_max_physical_batch_size: 32

        other:
            output_path: /output/dir/path/
            gpu_id: 0
            time_stamp: "name of output folder"


### for Ditto and ImprovedDitto
    data_list:
        clientName1: /data/dir/for/client1
        clientName2: /data/dir/for/client2
        clientName3: /data/dir/for/client3
    validation_data:
        clientName1: /validation/data/dir/for/client1
        clientName2: /validation/data/dir/for/client2
        clientName3: /validation/data/dir/for/client3
    train_parameter:
        learning_rate: 0.01
        batch_size: 12
        max_round: 3
        early_stopping: 0 
        data_augmentation_num: 2 

    fl_parameter:
        client_names_list:
            - clientName1
            - clientName2
            - clientName3
        client_num: 3
        epoch_per_round: 1
        mu: 0.1 (for Ditto)
        ratio: 1 (for ImprovedDitto)
        is_ratio: True (ImprovedDitto) / False (Ditto)
        first_except: True (ImprovedDitto) / False (Ditto)

    dp_parameter:
        do_dp: False (fixed)
        dp_secure_mode: False
        dp_max_grad: 1.0 
        noise_multiplier: 0.1 
        dp_max_physical_batch_size: 12

    other:
        is_grad: true
        output_path: /output/dir/path/
        gpu_id: 0
        time_stamp: "name of output folder"


## Citation

if you find ***Improved Ditto*** to be useful in your own research, please consider citing the folloing paper:
        ///