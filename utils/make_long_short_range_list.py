# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26th 2023
@author: yamadaaiki
"""

import argparse
import xml.etree.ElementTree as ET


def make_file_selected_above_lower(input_file:str,
                                   output_dir:str,
                                   parent_dir_path:str):
    
    short_output_file = f"{output_dir}/short_list.txt"
    long_output_file = f"{output_dir}/long_diameter_list.txt"
    
    # open file
    
    path_list = []
    
    with open(input_file, "r") as fp:
        
        for path in fp:
            path_list.append(path.rstrip('\n'))
            
    short_range_list = []
    long_range_list  = []
    
    for path in path_list:
        
        xml_path = f"{parent_dir_path}/{path}.xml"
        
        object = ET.parse(xml_path).findall("object")
        
        for object in object:
            bbox = object.find("bndbox")
            
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            x_range = x2 - x1 + 1
            y_range = y2 - y1 + 1
            
            short_range = min(x_range, y_range)
            long_range = max(x_range, y_range)
            
            print(f"long range : {long_range} / short range : {short_range}")
            
            short_range_list.append(short_range)
            long_range_list.append(long_range)
            
    # print(short_range_list)
    # print(long_range_list)
    print(f"the number of metastatic brain list : {len(short_range_list)}")
    print(f"the number of metastatic brain long list : {len(long_range_list)}")
    
    short_count_list = []
    long_count_list  = []
    
    for idx in range(255):
        short_count_list.append([idx, short_range_list.count(idx)])
        long_count_list.append([idx, long_range_list.count(idx)])
    
    with open(short_output_file, "w") as fp:
        fp.writelines(f"{d[0]},{d[1]}\n" for d in short_count_list)
    
    with open(long_output_file, "w") as fp:
        fp.writelines(f"{d[0]},{d[1]}\n" for d in long_count_list)


def main():
    parser = argparse.ArgumentParser(
        description="Determine the object limits and retrieve the file.")
    parser.add_argument("input_file", help="input file name (.txt)")
    parser.add_argument("output_file_dir", help="output dir")
    parser.add_argument("parent_dir_path", help="parent dir path")
    
    args = parser.parse_args()
    
    print("start.")
    make_file_selected_above_lower(args.input_file,
                                   args.output_file_dir,
                                   args.parent_dir_path)
    print("fin.")
    

if __name__ == "__main__":
    main()