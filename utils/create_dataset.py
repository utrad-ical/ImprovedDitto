"""
Created on Tue Jul 4 2023
@author: yamadaaiki
"""

import sys
sys.path.append("..")

from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset
from pytorch_ssd.vision.utils import str2bool, Timer, freeze_net_layers, store_labels
from torch.utils.data import ConcatDataset, DataLoader

from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset


def create_ssd_train_dataset(dataset_path, train_transform, target_transform, test_transform, data_augmentation_num=1):
    
    datasets_list = []
    
    for n in range(data_augmentation_num):
        if n == 0:
            dataset = VOCDataset(dataset_path, transform=test_transform,
                                 target_transform=target_transform)
        else:
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
        num_classes = len(dataset.class_names)
        datasets_list.append(dataset)
        class_names = dataset.class_names
    
    train_dataset = ConcatDataset(datasets_list)
    
    return num_classes, train_dataset, class_names


def make_client_dataloader_list(client_list,
                                dataset_list,
                                transform,
                                target_transform,
                                train_params,
                                is_test=False):
    
    dataloader_list = []
    
    for i in range(len(client_list)):
        
        dataset = VOCDataset(dataset_list[i], transform=transform,
                             target_transform=target_transform, is_test=is_test)
        num_classes = len(dataset.class_names)
        
        train_loader = DataLoader(dataset, train_params["batch_size"],
                                  num_workers=2, shuffle=True)
        dataloader_list.append(train_loader)
        
    return dataloader_list, num_classes


def make_client_dataloader_list_for_data_augmentation(client_list,
                                                      dataset_list,
                                                      train_transform,
                                                      test_transform,
                                                      target_transform,
                                                      train_params,
                                                      data_augmentation_num,
                                                      is_test=False):
    
    dataloader_list = []
    
    for i in range(len(client_list)):
        num_classes, dataset, _ = create_ssd_train_dataset(dataset_list[i], train_transform,
                                                           target_transform, test_transform,
                                                           data_augmentation_num)
        
        train_loader = DataLoader(dataset, train_params['batch_size'],
                                  num_workers=2, shuffle=True)
        dataloader_list.append(train_loader)
    
    return dataloader_list, num_classes


def create_client_names_list(name_list:list, num:int):
    
    if 0 < num <= len(name_list):
        return name_list[:num]
    else:
        raise Exception("the number of clients is not matched.")


def make_output_client_log_file_list(num_client:int, output_dir:str, client_name_list:list):
    
    if num_client != len(client_name_list):
        raise Exception("the number of cilent and client name list's length must be matched.")
    
    output_file_list = []
    
    for client_name in client_name_list:
        output_file_list.append(f"{output_dir}/loss_log_client_{client_name}.csv")
        
        with open(f"{output_dir}/loss_log_client_{client_name}.csv", "w") as fp:
            fp.write(f"round, epoch_idx, training loss, validation loss, delta, epsilon\n")
        
    return output_file_list