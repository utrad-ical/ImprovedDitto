"""
Created on Mon Jul 10 2023
@author: ayamada
"""
import sys
sys.path.append("..")

import copy
from contextlib import nullcontext
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from FedAvg import fedavg

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from pytorch_ssd.vision.nn.multibox_loss import MultiboxLoss

from personalized_optim import PerturbedGradientDescent, ProposedPerturbedGradientDescent


class Client():
    def __init__(self, client_name, training_loader, validation_loader,
                 training_params, fl_params, dp_params, output_file_name,
                 ssd_config, device):
        
        self.client_name = client_name
        
        self.training_loader   = training_loader
        self.validation_loader = validation_loader
        
        self.training_params = training_params
        self.fl_params = fl_params
        self.dp_params = dp_params
        
        self.output_file_name = output_file_name
        self.device = device
        
        self.ssd_config = ssd_config
        
        if dp_params["do_dp"]:
            self.privacy_engine = PrivacyEngine(secure_mode=dp_params["dp_secure_rng"])
        
        
    def trainig_and_validation(self, model, round_idx):
        
        info_str = f"client name : {self.client_name} - {round_idx + 1} round"
        print(info_str)
        
        criterion = MultiboxLoss(self.ssd_config.priors,
                                    iou_threshold=0.2,
                                    neg_pos_ratio=0.3, center_variance=0.1,
                                    size_variance=0.2, device=self.device)
        
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=self.training_params["learning_rate"])
        
        # training mode
        model.train()
        
        if self.dp_params["do_dp"]:
            
            epsilon = 0
            training_delta = 1 / len(self.training_loader)
            model, optimizer, self.training_loader =\
                self.privacy_engine.make_private(module=model,
                                                 optimizer=optimizer,
                                                 data_loader=self.training_loader,
                                                 max_grad_norm=self.dp_params["dp_max_grad"],
                                                 noise_multiplier=self.dp_params["noise_multiplier"])
        else:
            training_delta = 0
            
        model.train()
        zero_batch_cnt = 0
        
        for epoch_idx in range(self.fl_params["epoch_per_round"]):
            
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            
            model.train()
            
            with BatchMemoryManager(data_loader=self.training_loader,
                                    max_physical_batch_size=self.dp_params["dp_max_physical_batch_size"],
                                    optimizer=optimizer)\
                if self.dp_params["do_dp"] else nullcontext(self.training_loader) as new_data_loader:
                
                for batch_idx, (data, boxes, labels) in enumerate(new_data_loader):
                    
                    if len(data) > 0:
                        
                        data = data.to(self.device)
                        boxes = boxes.to(self.device)
                        labels = labels.to(self.device)
                        
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
                    
                    else:
                        zero_batch_cnt += 1
            
            avg_running_loss = running_loss / (batch_idx + 1)
            avg_regression_loss = running_regression_loss / (batch_idx + 1)
            avg_classification_loss = running_classification_loss / (batch_idx + 1)
            
            model.eval()
            
            validation_running_loss = 0.0
            validation_running_regression_loss = 0.0
            validation_running_classification_loss = 0.0
            
            for batch_idx, (data, boxes, labels) in enumerate(self.validation_loader):
                
                data = data.to(self.device)
                boxes = boxes.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    confidence, locations = model(data)
                    validation_regression_loss, validation_classification_loss = \
                        criterion(confidence, locations, labels, boxes)
                    loss = validation_regression_loss + validation_classification_loss
                    
                    validation_running_loss += loss.item()
                    validation_running_regression_loss += validation_regression_loss.item()
                    validation_running_classification_loss += validation_classification_loss.item()
                    
            avg_validation_running_loss = validation_running_loss / (batch_idx + 1)
            
            avg_validation_running_regression_loss = \
                validation_running_regression_loss / (batch_idx + 1)
            
            avg_validation_running_classification_loss = \
                validation_running_classification_loss / (batch_idx +1)
                
            msg_str = f"round : {round_idx+1}, epoch : {epoch_idx + 1},\n"\
                      f"training loss : {avg_running_loss:.4f},\n"\
                      f"training regression loss : {avg_regression_loss:.4f},\n"\
                      f"training classification loss : {avg_classification_loss:.4f},\n"\
                      f"validation loss : {avg_validation_running_loss:.4f},\n"\
                      f"validation regression loss : {avg_validation_running_regression_loss:.4f}\n"\
                      f"validation classification loss : {avg_validation_running_classification_loss:.4f}"
            
            log_str = f"{round_idx+1}, {epoch_idx+1}, {avg_running_loss},"\
                        + f"{avg_regression_loss}, {avg_classification_loss},"\
                        + f"{avg_validation_running_loss}, {avg_validation_running_regression_loss},"\
                        + f"{avg_validation_running_classification_loss}"
            
            if self.dp_params["do_dp"]:
                epsilon = self.privacy_engine.get_epsilon(training_delta)
                msg_str += f",\ndelta : {training_delta},\nepsilon : {epsilon}"
                log_str += f",{training_delta}, {epsilon}"
            
            print(msg_str)
            
            with open(self.output_file_name, "a") as fp:
                fp.write(f"{log_str}\n")
        
        return model, avg_running_loss, avg_validation_running_loss


class ClientDitto():
    def __init__(self, client_name:str,
                 training_loader,
                 validation_loader,
                 trainig_params:dict,
                 fl_params:dict,
                 dp_params:dict,
                 output_file_name:str,
                 ssd_config,
                 device:str,
                 personalized_model_path:str,
                 personalized_loss_path:str,
                 is_ratio:bool):
        
        self.client_name = client_name
        
        self.training_loader = training_loader
        self.val_loader = validation_loader
        
        self.training_params = trainig_params
        self.fl_params = fl_params
        self.dp_params = dp_params
        
        self.loss_file = output_file_name
        self.ssd_config = ssd_config
        self.device = device
        
        self.personalized_model_path = personalized_model_path
        self.personalized_loss_path = personalized_loss_path
        
        self.is_ratio = is_ratio
        
        if dp_params['do_dp']:
            self.privacy_engine = PrivacyEngine(secure_mode=dp_params['dp_secure_rng'])
        else:
            self.privacy_engine = None
    
    def training_and_validation(self, model, round_idx):
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.training_params['learning_rate'])
        criterion = MultiboxLoss(self.ssd_config.priors,
                                 iou_threshold=0.2,
                                 neg_pos_ratio=0.3,
                                 center_variance=0.1,
                                 size_variance=0.2,
                                 device=self.device)
        
        model.train()
        
        if self.privacy_engine is not None:
            
            epsilon = 0.0
            training_delta = 1.0 / len(self.training_loader)
            model, optimizer, self.training_loader =\
                self.privacy_engine.make_private(module=model,
                                                 optimizer=optimizer,
                                                 data_loader=self.training_loader,
                                                 max_grad_norm=self.dp_params['dp_max_grad'],
                                                 noise_multiplier=self.dp_params['noise_multiplier'])
        else:
            training_delta = 0.0
        
        model.train()
        zero_batch_cnt = 0
        
        for epoch_idx in range(self.fl_params['epoch_per_round']):
            
            train_loss = 0.0
            train_reg_loss = 0.0
            train_class_loss = 0.0
            
            model.train()
            
            with BatchMemoryManager(data_loader=self.training_loader,
                                    max_physical_batch_size=self.dp_params['dp_max_physical_batch_size'],
                                    optimizer=optimizer)\
                                        if self.dp_params['do_dp'] else nullcontext(self.training_loader) as new_data_loader:
                
                for batch_idx, (data, boxes, labels) in enumerate(new_data_loader):
                    
                    if len(data) > 0:
                        
                        data   = data.to(self.device)
                        boxes  = boxes.to(self.device)
                        labels = labels.to(self.device)
                        
                        optimizer.zero_grad()
                        conf, loc = model(data)
                        reg_loss, class_loss = criterion(conf, loc, labels, boxes)
                        
                        loss = reg_loss + class_loss
                        loss.backward()
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_reg_loss += reg_loss.item()
                        train_class_loss += class_loss.item()
                        
                    else:
                        zero_batch_cnt += 1
                        
            avg_train_loss = train_loss / (batch_idx + 1 - zero_batch_cnt)
            avg_train_reg_loss = train_reg_loss / (batch_idx + 1 - zero_batch_cnt)
            avg_train_class_loss = train_class_loss / (batch_idx + 1 - zero_batch_cnt)
            
            # validation
            model.eval()
            
            val_loss, val_reg_loss, val_class_loss = 0.0, 0.0, 0.0
            
            for batch_idx, (data, boxes, labels) in enumerate(self.val_loader):
                
                data   = data.to(self.device)
                boxes  = boxes.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    conf, loc = model(data)
                    reg_loss, class_loss = criterion(conf, loc, labels, boxes)
                    loss = reg_loss + class_loss
                    
                    val_loss += loss.item()
                    val_reg_loss += reg_loss.item()
                    val_class_loss += class_loss.item()
                    
            avg_val_loss = val_loss / (batch_idx + 1)
            avg_val_reg_loss = val_reg_loss / (batch_idx + 1)
            avg_val_class_loss = val_class_loss / (batch_idx + 1)
            
            msg_str = f'[global train] round{round_idx+1}-epoch{epoch_idx+1}, train loss : {avg_train_loss:.4f}, val loss : {avg_val_loss:.4f}'
            
            log_str = f'{round_idx+1},{epoch_idx+1},{avg_train_loss},{avg_train_reg_loss},{avg_train_class_loss},'\
                        + f'{avg_val_loss},{avg_val_reg_loss},{avg_val_class_loss}'
            
            if self.dp_params['do_dp']:
                epsilon = self.privacy_engine.get_epsilon(training_delta)
                msg_str += f', delta : {training_delta}, epsilon : {epsilon}'
                log_str += f',{training_delta},{epsilon}'
                
            print(msg_str)
            
            with open(self.loss_file, 'a') as fp:
                fp.write(f'{log_str}\n')
            
        return model, avg_train_loss, avg_val_loss
    
    def per_training_and_validation(self, global_model, model_per, round_idx:int, is_last:bool=False, 
                                    grad_loss_path:str=None, is_first:bool=False):
        
        model_per.to(self.device)
        
        if self.is_ratio:
            optimizer = ProposedPerturbedGradientDescent(model_per.parameters(),
                                                lr=self.training_params['learning_rate'],
                                                ratio=self.fl_params['ratio'])
        else:
            optimizer = PerturbedGradientDescent(model_per.parameters(),
                                                lr=self.training_params['learning_rate'],
                                                mu=self.fl_params['mu'])
        
        
        criterion = MultiboxLoss(self.ssd_config.priors,
                                 iou_threshold=0.2,
                                 neg_pos_ratio=0.3,
                                 center_variance=0.1,
                                 size_variance=0.2,
                                 device=self.device)
        
        model_per.train()
        
        if self.privacy_engine is not None:
            
            epsilon = 0.0
            training_delta = 1.0 / len(self.training_loader)
            model_per, optimizer, self.training_loader =\
                self.privacy_engine.make_private(module=model_per,
                                                 optimizer=optimizer,
                                                 data_loader=self.training_loader,
                                                 max_grad_norm=self.dp_params['dp_max_grad'],
                                                 noise_multiplier=self.dp_params['noise_multiplier'])
        else:
            training_delta = 0.0
        
        model_per.train()
        zero_batch_cnt = 0
        
        for epoch_idx in range(self.fl_params['epoch_per_round']):
            
            train_loss = 0.0
            train_reg_loss = 0.0
            train_class_loss = 0.0
            
            model_per.train()
            
            with BatchMemoryManager(data_loader=self.training_loader,
                                    max_physical_batch_size=self.dp_params['dp_max_physical_batch_size'],
                                    optimizer=optimizer)\
                                        if self.dp_params['do_dp'] else nullcontext(self.training_loader) as new_data_loader:
                
                for batch_idx, (data, boxes, labels) in enumerate(new_data_loader):
                    
                    if len(data) > 0:
                        
                        data   = data.to(self.device)
                        boxes  = boxes.to(self.device)
                        labels = labels.to(self.device)
                        
                        optimizer.zero_grad()
                        conf, loc = model_per(data)
                        reg_loss, class_loss = criterion(conf, loc, labels, boxes)
                        
                        loss = reg_loss + class_loss
                        loss.backward()
                        
                        optimizer.step(global_model.parameters(), self.device, grad_loss_path, is_first=is_first)
                        
                        train_loss += loss.item()
                        train_reg_loss += reg_loss.item()
                        train_class_loss += class_loss.item()
                        
                    else:
                        zero_batch_cnt += 1
                        
            avg_train_loss = train_loss / (batch_idx + 1 - zero_batch_cnt)
            avg_train_reg_loss = train_reg_loss / (batch_idx + 1 - zero_batch_cnt)
            avg_train_class_loss = train_class_loss / (batch_idx + 1 - zero_batch_cnt)
            
            # validation
            model_per.eval()
            
            val_loss, val_reg_loss, val_class_loss = 0.0, 0.0, 0.0
            
            for batch_idx, (data, boxes, labels) in enumerate(self.val_loader):
                
                data   = data.to(self.device)
                boxes  = boxes.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    conf, loc = model_per(data)
                    reg_loss, class_loss = criterion(conf, loc, labels, boxes)
                    loss = reg_loss + class_loss
                    
                    val_loss += loss.item()
                    val_reg_loss += reg_loss.item()
                    val_class_loss += class_loss.item()
                    
            avg_val_loss = val_loss / (batch_idx + 1)
            avg_val_reg_loss = val_reg_loss / (batch_idx + 1)
            avg_val_class_loss = val_class_loss / (batch_idx + 1)
            
            msg_str = f'[personalized train] round{round_idx+1}-epoch{epoch_idx+1}, train loss : {avg_train_loss:.4f}, val loss : {avg_val_loss:.4f}'
            
            log_str = f'{round_idx+1},{epoch_idx+1},{avg_train_loss},{avg_train_reg_loss},{avg_train_class_loss},'\
                        + f'{avg_val_loss},{avg_val_reg_loss},{avg_val_class_loss}'
            
            if self.dp_params['do_dp']:
                epsilon = self.privacy_engine.get_epsilon(training_delta)
                msg_str += f', delta : {training_delta}, epsilon : {epsilon}'
                log_str += f',{training_delta},{epsilon}'
                
            print(msg_str)
            
            with open(self.personalized_loss_path, 'a') as fp:
                fp.write(f'{log_str}\n')
                
            if is_last:
                print(f'saved client personalized model. ==> {self.personalized_model_path}')
                torch.save(model_per.state_dict(), self.personalized_model_path)
            
        return model_per.to(self.device), avg_train_loss, avg_val_loss

