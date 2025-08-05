"""
Created on Mon Jul 10 2023
@author: ayamada
"""

import argparse
from datetime import datetime as dt
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rn
import sys
import tqdm
import time
import yaml

sys.path.append("..")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from FedAvg import fedavg
from client import Client

from pytorch_ssd.vision.ssd.ssd import MatchPrior
from pytorch_ssd.vision.ssd.proposed_ssd import create_proposed_ssd
from pytorch_ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from pytorch_ssd.vision.ssd.config.proposed_ssd_config import  ProposedSSDConfig
from pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss

from utils.create_dataset import make_client_dataloader_list, create_client_names_list, \
    make_output_client_log_file_list, create_ssd_train_dataset, make_client_dataloader_list_for_data_augmentation

# client name list
CLIENTS_NAME = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def fl_training(training_params, fl_params, dp_params, other_config, data_list):
    
    # Fix seed
    seed_num = 45678
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    
    if fl_params["client_names_list"] is not None:
        global CLIENTS_NAME
        CLIENTS_NAME = []
        CLIENTS_NAME = fl_params["client_names_list"]
        
    client_list = create_client_names_list(CLIENTS_NAME, fl_params["client_num"])
    
    device = f"cuda:{other_config['gpu_id']}"
    
    time_stamp = other_config["time_stamp"]
    
    if time_stamp == "":
        time_stamp = dt.now().strftime("%Y%m%d%H%M%S")
    
    output_path = f"{other_config['out_path']}/{time_stamp}"
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    server_log_file_name = f"{output_path}/server_loss_log.csv"
    global_model_file_name = f"{output_path}/server_best_model.pth"
    global_model_last_name = f"{output_path}/server_last_model.pth"
    
    # set SSD config
    config = ProposedSSDConfig()
    
    # use for early stopping
    patience_cnt = 0
    
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.2)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    # create training dataset and dataloader
    if training_params['augmentation_num'] == 0:
        client_training_loader_list, num_classes = \
            make_client_dataloader_list(client_list, data_list["training_data"], 
                                        train_transform, target_transform, training_params)
    elif training_params['augmentation_num'] >= 1:
        client_training_loader_list, num_classes = \
            make_client_dataloader_list_for_data_augmentation(client_list, data_list['training_data'],
                                                              train_transform, test_transform, target_transform,
                                                              training_params, training_params['augmentation_num']+1)
            
    client_validation_loader_list, _ = \
        make_client_dataloader_list(client_list, data_list["validation_data"],
                                    test_transform, target_transform,
                                    training_params, is_test=True)
    
    global_model = create_proposed_ssd(num_classes)
    
    # send to gpus
    global_model.to(device)
    # global model change to train mode
    global_model.train(True)
    
    # make client output file list
    client_output_path = f"{output_path}/client"
    if not os.path.isdir(client_output_path):
        os.makedirs(client_output_path)
        
    client_output_path_list = make_output_client_log_file_list(fl_params["client_num"], client_output_path, client_list)
    
    best_global_validation_loss = float("inf")
    
    patience_cnt = 0
    for round_idx in range(training_params["max_rounds"]):
        
        print(f"start round {round_idx + 1}")
        
        global_model.train()
        
        client_weight_list = []
        client_training_loss_list = []
        client_validation_loss_list = []
        client_training_data_num_list = []
        
        # each client train and validation
        for client_idx, (client_name, output_file) in\
            enumerate(zip(client_list, client_output_path_list)):
            
            client_model = Client(client_name=client_name,
                                  training_loader=client_training_loader_list[client_idx],
                                  validation_loader=client_validation_loader_list[client_idx],
                                  training_params=training_params,
                                  fl_params=fl_params,
                                  dp_params=dp_params,
                                  output_file_name=output_file,
                                  ssd_config=config,
                                  device=device)
            
            client_weight, training_loss, validation_loss =\
                client_model.trainig_and_validation\
                    (model=copy.deepcopy(global_model),
                           round_idx=round_idx)
            
            client_weight_list.append(copy.deepcopy(client_weight))
            client_training_loss_list.append(copy.deepcopy(training_loss))
            client_validation_loss_list.append(copy.deepcopy(validation_loss))
            
            client_training_data_num_list.append(len(client_training_loader_list[client_idx]))
        
        global_model = fedavg(global_model, client_weight_list, client_training_data_num_list)
        
        # calc each loss
        avg_server_training_loss = float(sum(client_training_loss_list)) / len(client_training_loss_list)
        avg_server_validation_loss = float(sum(client_validation_loss_list)) / len(client_validation_loss_list)
        
        saved_str = ""
        if best_global_validation_loss > avg_server_validation_loss:
            best_global_validation_loss = avg_server_validation_loss
            torch.save(global_model.state_dict(), global_model_file_name)
            saved_str = " ==> model saved"
            patience_cnt = 0
        else:
            patience_cnt += 1
        
        msg_str = f"round : {round_idx + 1}\n"\
                  f"server training loss : {avg_server_training_loss:.4f}\n"\
                  f"servetr validation loss : {avg_server_validation_loss:.4f}"
        log_str = f"{round_idx + 1}, {avg_server_training_loss}, {avg_server_validation_loss}"
        
        msg_str += saved_str
        print(msg_str)
        
        with open(server_log_file_name, "a") as fp:
            fp.write(f"{log_str}\n")
        
        if 0 < training_params["early_stopping"] <= patience_cnt:
            break
        
    torch.save(global_model.state_dict(), global_model_last_name)
    print(f'finish train. save last epoch model ==> {global_model_last_name}')


def main():
    parser = argparse.ArgumentParser(
        description="Training of SSD model",
        add_help=True)
    
    parser.add_argument("training_yaml_file",
                        help="training config file path(.yaml or .yml)")
    
    args = parser.parse_args()
    
    with open(args.training_yaml_file) as fp:
        fl_config = yaml.safe_load(fp)
    
    training_params = {"learning_rate"   : fl_config["train_parameter"]["learning_rate"],
                       "batch_size"      : fl_config["train_parameter"]["batch_size"],
                       "max_rounds"      : fl_config["train_parameter"]["max_round"],
                       "early_stopping"  : fl_config["train_parameter"]["early_stopping"],
                       "augmentation_num": fl_config["train_parameter"]["data_augmentation_num"]}
    
    other_config = {"out_path"   : fl_config["other"]["output_path"],
                    "gpu_id"     : fl_config["other"]["gpu_id"],
                    "time_stamp" : fl_config["other"]["time_stamp"]}
    
    dp_params = {"do_dp"                      : fl_config["dp_parameter"]["do_dp"],
                 "dp_secure_rng"              : fl_config["dp_parameter"]["dp_secure_mode"],
                 "dp_max_grad"                : fl_config["dp_parameter"]["dp_max_grad"],
                 "noise_multiplier"           : fl_config["dp_parameter"]["noise_multiplier"],
                 "dp_max_physical_batch_size" : fl_config["dp_parameter"]["dp_max_physical_batch_size"]}
    
    fl_params = {"client_names_list" : fl_config["fl_parameter"]["client_names_list"],
                 "client_num"        : fl_config["fl_parameter"]["client_num"],
                 "epoch_per_round"   : fl_config["fl_parameter"]["epoch_per_round"]}
    
    data_list = {"training_data"   : fl_config["data_list"]["training_data"],
                 "validation_data" : fl_config["data_list"]["validation_data"]}
    
    fl_training(training_params, fl_params, dp_params, other_config, data_list)


if __name__ == "__main__":
    main()