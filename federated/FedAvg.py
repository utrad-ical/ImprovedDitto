import torch
import copy


def fedavg(global_model, client_model_list, data_num_list):
    
    # korede iino?
    global_model_ = copy.deepcopy(global_model)
    
    total_data_num = float(sum(data_num_list))
    
    if total_data_num <= 0:
        raise Exception(f"Total data count less than 0 is not correct.")
    
    # Fills self tensor with zeros.
    for param in global_model_.parameters():
        param.data.zero_()
    
    for n in range(len(client_model_list)):
        for server_param, client_param \
            in zip(global_model_.parameters(), client_model_list[n].parameters()):
            
            server_param.data += client_param.clone() * (data_num_list[n] / total_data_num)
            
    return global_model_


def fedavg_dp(global_model:torch.nn.Module, client_model_dict:dict, data_size_dict:dict, client_name_list:list):
    
    global_model_ = copy.deepcopy(global_model)
    
    total_data_num = float(sum(data_size_dict.values()))
    
    if total_data_num <= 0:
        raise ValueError()
    
    for param in global_model_.parameters():
        param.data.zero_()
        
    for client_name in client_name_list:
        for server_param, client_param\
            in zip(global_model_.parameters(), client_model_dict[client_name].parameters()):
                
                server_param.data += client_param.clone() * (float(data_size_dict[client_name]) / total_data_num)
    
    return global_model_