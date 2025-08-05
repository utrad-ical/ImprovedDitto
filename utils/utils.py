# -*- coding: utf-8 -*-
"""
Created on Mon March 4th 2024
@author: yamadaaiki
"""
import sys
sys.path.append('..')

import os

from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset

from torch.utils.data import ConcatDataset, DataLoader


def create_client_name_list(client_name_candidate_list:list, client_num:int)->list:
    
    if 0 < client_num <= len(client_name_candidate_list):
        return client_name_candidate_list[:client_num]
    else:
        raise ValueError()


def make_client_dataloader_dict(client_name_list:list,
                                data_path_dict:dict,
                                train_transform,
                                test_transform,
                                target_transform,
                                training_params,
                                is_test:bool=False):
    
    dataloader_dict = {}
    datasize_dict = {}
    
    for client_name in client_name_list:
        dataset_list = []
        for i in range(training_params.get('data_augmentation_num', 0) + 1):
            if i == 0:
                dataset = VOCDataset(data_path_dict[client_name],
                                     transform=test_transform,
                                     target_transform=target_transform,
                                     is_test=is_test)
            else:
                dataset =VOCDataset(data_path_dict[client_name],
                                    transform=train_transform,
                                    target_transform=target_transform,
                                    is_test=is_test)
                
            num_classes = len(dataset.class_names)
            class_names = dataset.class_names
            
            dataset_list.append(dataset)
        
        if i != 0:
            train_dataset = ConcatDataset(dataset_list)
        else:
            train_dataset = dataset
        
        training_loader = DataLoader(train_dataset,
                                     training_params['batch_size'],
                                     num_workers=2,
                                     shuffle=True)
        
        dataloader_dict[client_name] = training_loader
        datasize_dict[client_name] = len(training_loader.dataset) 
        
    print(datasize_dict)
    
    return dataloader_dict, datasize_dict, num_classes, class_names


def make_client_val_dataloader_dict(client_name_list:list,
                                    data_path_dict:dict,
                                    test_transform,
                                    target_transform,
                                    training_params,
                                    is_test:bool=True):
    
    dataloader_dict = {}
    
    for client_name in client_name_list:
        dataset = VOCDataset(data_path_dict[client_name],
                             transform=test_transform,
                             target_transform=target_transform,
                             is_test=is_test)
        
        data_loader = DataLoader(dataset,
                                 1,
                                 num_workers=2,
                                 shuffle=False)
        
        dataloader_dict[client_name] = data_loader
    
    return dataloader_dict


def make_client_loss_path_dict(output_dir:str, client_name_list:list, is_per:bool=False)->dict:
    
    output_file_dict = {}
    
    for client_name in client_name_list:
        output_file_dict[client_name] = f'{output_dir}/{client_name}/loss_log_client_{client_name}.csv'
        
        if is_per:
            output_file_dict[client_name] = f'{output_dir}/{client_name}/per_loss_log_client_{client_name}.csv'
        
        if not os.path.isdir(f'{output_dir}/{client_name}'):
            os.makedirs(f'{output_dir}/{client_name}')
            
        with open(output_file_dict[client_name], 'w') as fp:
            fp.write(f'round,epoch_index,training loss,train reg loss,train class loss,val loss,val reg loss,val class loss,delta,epsilon\n')
        
    return output_file_dict


def make_client_model_path_dict(output_dir:str, client_name_list:list)->dict:
    
    client_path_dict = {}
    
    for client_name in client_name_list:
        client_path_dict[client_name] = f'{output_dir}/{client_name}/model_client_{client_name}.pth'
        
        if not os.path.isdir(f'{output_dir}/{client_name}'):
            os.makedirs(f'{output_dir}/{client_name}')
            
    return client_path_dict


def make_grad_loss_path_dict(client_num:int, output_dir:str, client_name_list:list)->dict:
    
    if client_num != len(client_name_list):
        raise ValueError(f'the number of client is not matched. {client_num}')
    
    client_grad_loss_path_dict = {}
    
    for client_name in client_name_list:
        
        client_grad_loss_path_dict[client_name] = f'{output_dir}/grad_loss_{client_name}.csv'
        
        with open(f'{output_dir}/grad_loss_{client_name}.csv', 'w') as fp:
            fp.write(f"p.grad.data,p.data-g.data\n")
            
    return client_grad_loss_path_dict


# use for calc cos sim for automaticaly tuning r or lambda.
def make_client_cos_sim_path_dict(client_num:int, output_dir:str, client_name_list:list)->dict:
    
    if client_num != len(client_name_list):
        raise ValueError(f'the number of clients is not matched.')
    
    client_cos_sim_path_dict = {}
    
    for client_name in client_name_list:
        
        client_cos_sim_path_dict[client_name] = f'{output_dir}/cos_sim_{client_name}.csv'
        
        with open(f'{output_dir}/cos_sim_{client_name}.csv', 'w') as fp:
            fp.write(f"global model gradient with all clients,global model gradient with in-client data,personalized model gradient with in-client data\n")
            
    return client_cos_sim_path_dict