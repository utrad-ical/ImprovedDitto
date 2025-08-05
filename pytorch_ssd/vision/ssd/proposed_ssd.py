
from torch.nn import Sequential, ModuleList
from ..nn.scaled_l2_norm import ScaledL2Norm
import torch.nn as nn

from .ssd import ProposedSSD
from .predictor import Predictor
from .config.proposed_ssd_config import ProposedSSDConfig
config = ProposedSSDConfig()

def create_proposed_ssd(num_classes, is_test=False, in_channels=3):
    base_net = ModuleList([
        # block 1
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # block 2
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # block 3
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # block 4
        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding='same', stride=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding='same', stride=1),
        nn.ReLU(inplace=False)
    ])
    
    # not use
    source_layer_indexes = []
    
    extras = ModuleList([
        Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='valid', stride=2),
            nn.ReLU(inplace=False)),
        Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding='same', stride=1),
            nn.ReLU(inplace=False),
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='valid', stride=2),
            nn.ReLU(inplace=False)),
        Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding='same', stride=1),
            nn.ReLU(inplace=False),
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='valid', stride=2),
            nn.ReLU(inplace=False))
    ])
    
    regression_headers = ModuleList([
        nn.Conv2d(in_channels=128,  out_channels=4 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256,  out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512,  out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256,  out_channels=4 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256,  out_channels=4 * 4, kernel_size=3, padding=1), ]) # TODO: change to kernel_size=1, padding=0?
    
    classification_headers = ModuleList([
        nn.Conv2d(in_channels=128,  out_channels=4 * num_classes, kernel_size=3, padding=1), # 16384 128*128
        nn.Conv2d(in_channels=256,  out_channels=6 * num_classes, kernel_size=3, padding=1), # 4096 64*64
        nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1), # 441 21*21
        nn.Conv2d(in_channels=512,  out_channels=6 * num_classes, kernel_size=3, padding=1), # 121 11*11
        nn.Conv2d(in_channels=256,  out_channels=4 * num_classes, kernel_size=3, padding=1), # 36 6*6
        nn.Conv2d(in_channels=256,  out_channels=4 * num_classes, kernel_size=3, padding=1), # 9 3*3
        # TODO: change to kernel_size=1, padding=0?
        ])
    
    return ProposedSSD(num_classes, base_net, source_layer_indexes,
                       extras, classification_headers, regression_headers,
                       is_test=is_test, config=config)


def create_proposed_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    
    return predictor