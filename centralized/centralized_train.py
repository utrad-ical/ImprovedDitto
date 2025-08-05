"""
Created on Mon Jul 3 2023
@author: ayamada
"""


import argparse
from contextlib import nullcontext
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rn
import itertools
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.append("..")

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from pytorch_ssd.vision.utils import str2bool, Timer, freeze_net_layers, store_labels
from pytorch_ssd.vision.ssd.ssd import MatchPrior
from pytorch_ssd.vision.ssd.proposed_ssd import create_proposed_ssd
from pytorch_ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from pytorch_ssd.vision.ssd.config.proposed_ssd_config import ProposedSSDConfig
from pytorch_ssd.vision.datasets.voc_dataset import VOCDataset
from pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss
from pytorch_ssd.vision.utils.misc import store_labels

from utils.create_dataset import create_ssd_train_dataset


def training_ssd(training_params, dp_params, other_config, is_in_random_serch=False):
    
    # Fix seed
    seed_num = 45678
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    
    device = f'cuda:{other_config["gpu_id"]}'
    
    time_stamp = other_config["time_stamp"]
    
    if not os.path.isdir(other_config["out_path"]):
        os.makedirs(other_config["out_path"])
    
    if time_stamp == "":
        time_stamp = dt.now().strftime("%Y%m%d%H%M%S")
    
    if is_in_random_serch:
        output_path = other_config['out_path']
    else:
        output_path = f"{other_config['out_path']}/{time_stamp}"
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    log_file_name   = f"{output_path}/loss_log_{time_stamp}.csv"
    model_file_name = f"{output_path}/model_best_{time_stamp}.pth"
    model_last_name = f"{output_path}/model_last_{time_stamp}.pth"
    
    config = ProposedSSDConfig()
    
    patience_cnt = 0
    
    train_transform  = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.2)
    test_transform   = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    labe_file = f"{output_path}/voc_model_labels.txt"
    # create training dataset and dataloader
    if training_params['augmentation_num'] == 0:
        dataset = VOCDataset(other_config["trainig_dataset_path"],
                                transform=train_transform,
                                target_transform=target_transform)
        num_classes = len(dataset.class_names)
        store_labels(labe_file, dataset.class_names)
    elif training_params['augmentation_num'] >= 1:
        num_classes, dataset,  class_names= create_ssd_train_dataset(other_config['trainig_dataset_path'],
                                                        train_transform, target_transform,
                                                        test_transform, training_params['augmentation_num']+1)
        store_labels(labe_file, class_names)
    
    
    training_loader = DataLoader(dataset, training_params["batch_size"],
                                 num_workers=2, shuffle=True)
    
    # create validation dataset and dataloader
    validation_dataset = VOCDataset(other_config["validation_dataset_path"], transform=test_transform,
                                    target_transform=target_transform, is_test=True)
    validation_loader = DataLoader(validation_dataset, training_params["batch_size"],
                                   num_workers=2, shuffle=False)
    
    model = create_proposed_ssd(num_classes)
    
    best_validation_loss = float('inf')
    
    criterion = MultiboxLoss(config.priors, iou_threshold=0.2, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=training_params["learning_rate"])
    
    model.to(device)
    model.train(True)
    
    for epoch_idx in tqdm.tqdm(range(training_params["max_epoch"])):
        
        model.train()
        
        if dp_params["do_dp"]:
            privacy_engine = PrivacyEngine(secure_mode=dp_params["dp_secure_mode"])
            
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        validation_running_loss = 0.0
        validation_running_regression_loss = 0.0
        validation_running_classification_loss = 0.0
        
        if dp_params["do_dp"]:
            
            epsilon = 0
            training_delta = 1.0 / len(dataset) # 確認が必要
            model, optimizer, training_loader = \
                privacy_engine.make_private(module=model,
                                            optimizer=optimizer,
                                            data_loader=training_loader,
                                            max_grad_norm=dp_params["dp_max_grad_norm"],
                                            noise_multiplier=dp_params["dp_noise_multiplier"])
        else:
            training_delta = 0
        
        model.train()
        
        with BatchMemoryManager(data_loader=training_loader,
                                max_physical_batch_size=dp_params["dp_max_physical_batch_size"],
                                optimizer=optimizer)\
                                    if dp_params["do_dp"] else nullcontext(training_loader) as new_data_loader:
            
            for batch_idx, (data, boxes, labels) in tqdm.tqdm(enumerate(new_data_loader)):
                
                data, boxes, labels = data.to(device), boxes.to(device), labels.to(device)
                
                optimizer.zero_grad()
                confidence, locations = model(data)
                regression_loss, classification_loss = \
                    criterion(confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_regression_loss += regression_loss.item()
                running_classification_loss += classification_loss.item()
                
        avg_running_loss = running_loss / (batch_idx + 1)
        avg_running_regression_loss = running_regression_loss / (batch_idx + 1)
        avg_running_classification_loss = running_classification_loss / (batch_idx + 1)
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, boxes, labels) in tqdm.tqdm(enumerate(validation_loader)):
                data, boxes, labels = data.to(device), boxes.to(device), labels.to(device)
                confidence, locations = model(data)
                validation_regression_loss, validation_classification_loss =\
                    criterion(confidence, locations, labels, boxes)
                loss = validation_regression_loss + validation_classification_loss
                
                validation_running_loss += loss.item()
                validation_running_classification_loss += validation_classification_loss.item()
                validation_running_regression_loss += validation_regression_loss.item()
        
        avg_validation_runnig_loss = validation_running_loss / (batch_idx + 1)
        avg_validation_running_regression_loss = validation_running_regression_loss / (batch_idx + 1)
        avg_validation_running_classification_loss = validation_running_classification_loss / (batch_idx + 1)
        
        saved_str = ""
        
        if best_validation_loss > avg_validation_runnig_loss:
            best_validation_loss = avg_validation_runnig_loss
            torch.save(model.state_dict(), model_file_name)
            saved_str = " ==> model saved"
            patience_cnt = 0
        else:
            patience_cnt += 1
            if 0 < training_params["early_stopping"] <= patience_cnt:
                saved_str = " ==> stopped"
        
        msg_str = f"epoch {epoch_idx + 1}\n" +\
                  f"training loss : {avg_running_loss:.4f}\n" +\
                  f"training regression loss:{avg_running_regression_loss:.4f}\n" +\
                  f"training classification loss:{avg_running_classification_loss:.4f}\n" + \
                  f"validation loss:{avg_validation_runnig_loss:.4f}\n" +\
                  f"validation regression loss:{avg_validation_running_regression_loss:.4f}\n" +\
                  f"validation classification loss:{avg_validation_running_classification_loss:.4f}"
        log_str = f"{epoch_idx + 1}, {avg_running_loss}, {avg_running_regression_loss}," +\
                  f"{avg_running_classification_loss}, {avg_validation_runnig_loss}," + \
                  f"{avg_validation_running_regression_loss}," + \
                  f"{avg_validation_running_classification_loss}"
        
        if dp_params["do_dp"]:
            epsilon = privacy_engine.get_epsilon(training_delta)
            msg_str += f",\ndelta:{training_delta},\nepsilon:{epsilon:.4f}"
            log_str += f",{training_delta},{epsilon}"
        
        msg_str += saved_str
        print(msg_str)
        
        with open(log_file_name, "a") as fp:
            fp.write(f"{log_str}\n")
        
        if 0 < training_params["early_stopping"] <= patience_cnt:
            break
        
    torch.save(model.state_dict(), model_last_name)
    print(f'finished training. Save last epoch model.')


def main():
    parser = argparse.ArgumentParser(
        description=" Training of SSD model",
        add_help=True)
    
    parser.add_argument("training_datasets", help="Dataset directory path")
    parser.add_argument("validation_datasets", help="validation dataset directory path")
    parser.add_argument("output_path", help="output path or dir.")
    parser.add_argument("-g", "--gpu_id", help="GPU IDs",
                        type=str, default="0")
    parser.add_argument("-r", "--learning_rate", help="Learning rate",
                        type=float, default=0.001)
    parser.add_argument("-b", "--batch_size", help="batch size",
                        type=int, default=4)
    parser.add_argument("-m", "--max_epochs", 
                        help="maximum of the number of epochs",
                        type=int, default=500)
    parser.add_argument("-e", "--early_stopping",
                        help="patience of early stopping",
                        type=int, default=0)
    parser.add_argument('-a', '--augmentation_num',
                        help='Number of augmentation',
                        type=int, default=1)
    parser.add_argument("--time_stamp", help="time stamp or folder name for saved data",
                        type=str, default="")
    
    # for differential privacy
    parser.add_argument('--do_dp', help="whether do differential privacy training or not.",
                        action="store_true")
    parser.add_argument('--dp_secure_mode', help='Using secure mode for opacus.',
                        action='store_true')
    parser.add_argument('-mgd', '--dp_max_grad', help='dp parameter for max grad clipping',
                        type=float, default=1.0)
    parser.add_argument('-sig', '--noise_multiplier', help='dp parameter for epsilon',
                        type=float, default=0.1)
    parser.add_argument('-m_bth', '--max_physical_batch_size',type=int, default=4)
    
    
    args = parser.parse_args()
    
    training_params = {"batch_size" : args.batch_size,
                       "learning_rate" : args.learning_rate,
                       "max_epoch" : args.max_epochs,
                       "early_stopping" : args.early_stopping,
                       "augmentation_num": args.augmentation_num}
    
    dp_params = {"do_dp"                      : args.do_dp,
                 "dp_secure_mode"             : args.dp_secure_mode,
                 "dp_max_grad_norm"           : args.dp_max_grad,
                 "dp_noise_multiplier"        : args.noise_multiplier,
                 "dp_max_physical_batch_size" : args.max_physical_batch_size}
    
    other_config = {"gpu_id"                  : args.gpu_id,
                    "time_stamp"              : args.time_stamp,
                    "out_path"                : args.output_path,
                    "trainig_dataset_path"    : args.training_datasets,
                    "validation_dataset_path" : args.validation_datasets}
    
    print("start ssd training")
    training_ssd(training_params=training_params,
                 dp_params=dp_params,
                 other_config=other_config)
    print("fin")


if __name__ == "__main__":
    main()