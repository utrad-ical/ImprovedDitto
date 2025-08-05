# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 2022
@author: yamadaaiki
"""

import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy import ndimage
import shutil
import SimpleITK as sitk
import skimage.io
import xml.etree.ElementTree as ET
import xml.dom.minidom as md

import mhd_io


def create_data_for_object_detection(src_list_file_name, src_base_path, dst_base_path, dst_list_file_name):
    
    #target_size = np.array(target_size)
    
    dst_img_path = os.path.join(dst_base_path, "JPEGImages")
    dst_annotation_path = os.path.join(dst_base_path, "Annotations")
    dst_img_set_path = os.path.join(dst_base_path, "ImageSets", "Main")
    
    print(dst_annotation_path)
    
    os.makedirs(dst_img_path, 0o777, True)
    os.makedirs(dst_annotation_path, 0o777, True)
    os.makedirs(dst_img_set_path, 0o777, True)
    
    dst_img_set_file_name = os.path.join(dst_img_set_path, dst_list_file_name)
    
    structure_8 = np.ones((3,3), dtype=np.uint8)
    structure_26 = np.ones((3,3,3), dtype=np.uint8)
    
    with open(dst_img_set_file_name, "w", newline="\r\n") as fout:
        
        with open(src_list_file_name, "r") as fin:
        
            reader = csv.reader(fin)
        
            for items in reader:
                
                src_vol_file_name = os.path.join(src_base_path, items[0], "iso_normalized_volume.mhd")
                normalized_vol = skimage.io.imread(src_vol_file_name, plugin='simpleitk')
                normalized_vol_max = np.max(normalized_vol)
                
                src_label_file_name = os.path.join(src_base_path, items[0], "iso_meta_mask.mhd")
                labels = skimage.io.imread(src_label_file_name, plugin='simpleitk')
                label_num = np.max(labels)
                
                base_file_name = os.path.splitext(os.path.basename(items[1]))[0]
                
                for n in range (normalized_vol.shape[0]):
                    
                    if np.sum(labels[n]) > 0:
                        
                        #mask_img = (src_mask[n] > 0).astype(np.uint8)
                        
                        #labels, label_num = ndimage.label(mask_img, structure_8)
                        
                        # #src_img_file_name = items[0]
                        # #print(items[0])
                        
                        dst_img_file_name = os.path.join(dst_img_path, base_file_name + f"_{n:03d}.jpg")
                        dst_xml_file_name = os.path.join(dst_annotation_path, base_file_name + f"_{n:03d}.xml")
                        
                        normalized_img = normalized_vol[n,:,:]
                        normalized_img[normalized_img < 0] = 0
                        normalized_img = (normalized_img * 255.0 / normalized_vol_max).astype(np.uint8)
                        imageio.imwrite(dst_img_file_name, normalized_img)
                        
                        # write xml file
                        xml_root = ET.Element("annotation")
                        
                        a1 = ET.Element("folder")
                        a1.text = "JPEGImages"
                        xml_root.append(a1)
                        
                        a2 = ET.Element("filename")
                        #a2.text = os.path.basename(items[0])
                        xml_root.append(a2)
                        
                        # Add information of image size
                        b1 = ET.Element("source")
                        xml_root.append(b1)
                        b2 = ET.SubElement(b1, "database")
                        b2.text = "Unknown"
                        
                        # Add information of image size
                        c1 = ET.Element("size")
                        xml_root.append(c1)
                        c2 = ET.SubElement(c1, "width")
                        c2.text = str(normalized_vol.shape[2])
                        
                        c2 = ET.SubElement(c1, "height")
                        c2.text = str(normalized_vol.shape[1])
                        
                        c2 = ET.SubElement(c1, "depth")
                        c2.text = "3"
                        
                        d1 = ET.Element("segmented")
                        d1.text = "0"
                        xml_root.append(d1)
                        
                        # Add infoemation of metastasis
                        #histogram = ndimage.histogram(labels[n], 1, label_num, label_num)
                        b = ndimage.find_objects(labels[n])
                        
                        cnt = 0
                        
                        for idx, pos in enumerate(b):
                            
                            if pos is None:
                                continue                            
                            
                            e1 = ET.Element("object")
                            xml_root.append(e1)         
                            
                            e2 = ET.SubElement(e1, "name")
                            e2.text = "TP"
                            e3 = ET.SubElement(e1, "pose")
                            e3.text = "Unspecified"  
                            
                            e3 = ET.SubElement(e1, "truncated")
                            e3.text = "0"  
                            
                            e3 = ET.SubElement(e1, "difficult")
                            e3.text = "0"
                            
                            f1 = ET.Element("bndbox")
                            e1.append(f1)         
                            
                            f2 = ET.SubElement(f1, "xmin")
                            f2.text = str(pos[1].start)
                            
                            f3 = ET.SubElement(f1, "ymin")
                            f3.text = str(pos[0].start)
                            
                            f4 = ET.SubElement(f1, "xmax")
                            f4.text = str(pos[1].stop-1)
                            
                            f5 = ET.SubElement(f1, "ymax")
                            f5.text = str(pos[0].stop-1)
                            
                            cnt += 1
                            
                        print(items[1], n, label_num, cnt)
                        
                        # 文字列パースを介してminidomへ移す
                        xml_document = md.parseString(ET.tostring(xml_root, 'utf-8'))
                        
                        with open (dst_xml_file_name, "w") as fxml :
                            xml_document.writexml(fxml, encoding='utf-8', newl='\n', indent='', addindent='  ')
                
                #break    


def main():
    
    set_list = [["training", "trainval.txt", "training_case_list_mhd.csv"],
                ["test", "test.txt", "test_case_list_mhd.csv"]]
    
    src_base_path = "preprocessed"
    
    for items in set_list:
        create_data_for_object_detection(items[2], src_base_path, items[0], items[1])


if __name__ == '__main__':
    main()
