import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET


def make_file_selected_above_lower(input_file:str,
                                   output_file:str,
                                   parent_dir_path:str):
    
    # open file
    
    path_list = []
    
    with open(input_file, "r") as fp:
        
        for path in fp:
            path_list.append(path.rstrip('\n'))
            
    new_range_list = []
    
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
            
            min_range = min(x_range, y_range)
            
            print(x_range, y_range)
            
            new_range_list.append(min_range)
            
    print(new_range_list)
    print(f"the number of metastatic brain list : {len(new_range_list)}")
    
    count_list = []
    
    for idx in range(255):
        count_list.append([idx, new_range_list.count(idx)])
    
    with open(output_file, "w") as fp:
        fp.writelines(f"{d[0]},{d[1]}\n" for d in count_list)
    

def main():
    parser = argparse.ArgumentParser(
        description="Determine the object limits and retrieve the file.")
    parser.add_argument("input_file", help="input file name (.txt)")
    parser.add_argument("output_file_path", help="output file name (.txt)")
    parser.add_argument("parent_dir_path", help="parent dir path")
    
    args = parser.parse_args()
    
    print("start.")
    make_file_selected_above_lower(args.input_file,
                                   args.output_file_path,
                                   args.parent_dir_path)
    print("fin.")
    

if __name__ == "__main__":
    main()