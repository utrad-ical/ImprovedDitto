# -*- coding: utf-8 -*-
"""
Created on Mon March 4th 2024
@author: yamadaaiki
reference (https://proceedings.mlr.press/v139/li21h.html)
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

from FedAvg import fedavg_dp
from client import ClientDitto

from pytorch_ssd.vision.ssd.ssd import MatchPrior
from pytorch_ssd.vision.ssd.proposed_ssd import create_proposed_ssd
from pytorch_ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from pytorch_ssd.vision.ssd.config.proposed_ssd_config import ProposedSSDConfig

from utils.utils import create_client_name_list, make_client_dataloader_dict, make_client_loss_path_dict,\
    make_client_model_path_dict, make_client_val_dataloader_dict, make_grad_loss_path_dict

# client name list
CLIENTS_NAME = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def ditto_train(data_configs, training_params, fl_params, dp_params, other_configs):
    
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
    
    client_name_list = create_client_name_list(CLIENTS_NAME, fl_params["client_num"])
    print(client_name_list)
    
    device = f'cuda:{other_configs["gpu_id"]}'
    
    time_stamp = other_configs["time_stamp"]
    
    if not os.path.isdir(other_configs['output_path']):
        os.makedirs(other_configs["output_path"])
    
    if time_stamp == "":
        time_stamp = dt.now().strftime("%Y%m%d%H%M%S")
    
    output_path = f"{other_configs['output_path']}/{time_stamp}"
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    server_log_file_name    = f"{output_path}/server_loss_log.csv"
    server_model_file_name  = f"{output_path}/server_model_best.pth"
    server_last_model_name = f"{output_path}/server_last_model.pth"
    #roc_curve_file_name = f'{output_path}/server_validation_roc_model_best_{time_stamp}.png'
    
    # setting for SSD
    config = ProposedSSDConfig()
    
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.2)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    client_training_loader_dict, client_data_size_dict, num_classes, class_names =\
        make_client_dataloader_dict(client_name_list,
                                    data_configs['training_data'],
                                    train_transform,
                                    test_transform,
                                    target_transform,
                                    training_params)
    
    client_validation_loader_dict = make_client_val_dataloader_dict(client_name_list,
                                                                    data_configs['validation_data'],
                                                                    test_transform,
                                                                    target_transform,
                                                                    training_params,
                                                                    is_test=True)
    
    output_client_loss_dir = f"{output_path}/client_loss"
    if not os.path.isdir(output_client_loss_dir):
        os.makedirs(output_client_loss_dir)
    
    # dict for each client model and loss path 
    client_loss_path_dict = make_client_loss_path_dict(output_client_loss_dir,
                                                       client_name_list)
    client_model_path_dict = make_client_model_path_dict(output_client_loss_dir,
                                                         client_name_list)
    client_per_val_loss_path_dict = make_client_loss_path_dict(output_client_loss_dir,
                                                               client_name_list,
                                                               True)
    
    if other_configs['is_grad']:
        grad_loss_dict = make_grad_loss_path_dict(fl_params['client_num'],
                                                  output_client_loss_dir,
                                                  client_name_list)
    
    # load network
    global_model = create_proposed_ssd(num_classes)
    
    global_model.to(device)
    global_model.train()
    
    best_server_validation_loss = float('inf')
    
    client_per_model_dict = {}
    
    # fl training start
    for round_idx in range(training_params["max_round"]):
        
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {round_idx+1} round(s) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        is_last = (round_idx == training_params['max_round'] - 1)
        
        if round_idx == 0 and fl_params['first_except']:
            is_first = True
        else:
            is_first = False
        
        global_model.train()
        
        client_model_dict = {}
        client_training_loss_dict = {}
        client_validation_loss_dict = {}
        
        # each client training and validation
        for client_name in client_name_list:
            
            print(f'<{client_name}>')
            
            client = \
                ClientDitto(client_name=client_name,
                            training_loader=client_training_loader_dict[client_name],
                            validation_loader=client_validation_loader_dict[client_name],
                            trainig_params=training_params,
                            fl_params=fl_params,
                            dp_params=dp_params,
                            output_file_name=client_loss_path_dict[client_name],
                            ssd_config=config,
                            device=device,
                            personalized_model_path=client_model_path_dict[client_name],
                            personalized_loss_path=client_per_val_loss_path_dict[client_name],
                            is_ratio=fl_params['is_ratio'])
            
            # personalized model train and validation
            
            personalized_model, _, _ = \
                client.per_training_and_validation(copy.deepcopy(global_model),
                                                    copy.deepcopy(client_per_model_dict.get(client_name, global_model)),
                                                    round_idx,
                                                    is_last,
                                                    grad_loss_dict.get(client_name, None),
                                                    is_first=is_first)
            
            client_per_model_dict[client_name] = copy.deepcopy(personalized_model)
            
            # global model train and validation
            client_model, client_training_loss, client_validation_loss =\
                client.training_and_validation(copy.deepcopy(global_model), round_idx)
            
            client_model_dict[client_name] = copy.deepcopy(client_model)
            client_training_loss_dict[client_name] = copy.deepcopy(client_training_loss)
            client_validation_loss_dict[client_name] = copy.deepcopy(client_validation_loss)
            
        global_model = fedavg_dp(global_model, client_model_dict, client_data_size_dict, client_name_list)
        
        # calculate each loss
        avg_server_training_loss = float(sum(client_training_loss_dict.values())) / len(client_training_loss_dict)
        avg_server_validation_loss = float(sum(client_validation_loss_dict.values()) / len(client_validation_loss_dict))
        
        saved_str = ""
        if best_server_validation_loss > avg_server_validation_loss:
            best_server_validation_loss = avg_server_validation_loss
            torch.save(global_model.state_dict(), server_model_file_name)
            saved_str = " ==> global model saved"
        
        msg_str = f"[server] round : {round_idx + 1} training loss : {avg_server_training_loss:.4f}, server validation loss : {avg_server_validation_loss:.4f}"
        log_str = f"{round_idx+1},{avg_server_training_loss},{avg_server_validation_loss}"
        
        msg_str += saved_str
        print(msg_str)
        
        with open(server_log_file_name, 'a') as fp:
            fp.write(f"{log_str}\n")
            
    print(f"finished training with FL.\nsaved final model :{server_last_model_name}")
    torch.save(global_model.state_dict(), server_last_model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Training of classification model for multi classes with federated learning.",
        add_help=True)
    
    parser.add_argument("training_yaml_file",
                        help="training config file path(.yaml or .yml)")
    
    args = parser.parse_args()
    
    with open(args.training_yaml_file) as fp:
        fl_config = yaml.safe_load(fp)
    
    training_params = fl_config["train_parameter"]
    
    other_config = fl_config["other"]
    
    dp_params = fl_config["dp_parameter"]
    
    fl_params = fl_config["fl_parameter"]
    
    data_list = fl_config["data_list"]
    
    print('start')
    ditto_train(data_list, training_params, fl_params, dp_params, other_config)
    print('fin')


if __name__=="__main__":
    main()