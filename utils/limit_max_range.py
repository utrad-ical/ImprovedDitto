import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET


def make_file_selected_above_lower(input_file:str,
                                   output_file:str,
                                   parent_dir_path:str,
                                   lower_limit:float=10):
    
    # open file
    
    path_list = []
    
    with open(input_file, "r") as fp:
        
        for path in fp:
            path_list.append(path.rstrip('\n'))
            
    
    new_path_list = []
    
    for path in path_list:
        
        xml_path = f"{parent_dir_path}/{path}.xml"
        
        object = ET.parse(xml_path).findall("object")
        
        minimum_range = 100000
        
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
            
            if min_range < minimum_range:
                minimum_range = min_range
        
        if minimum_range >= lower_limit:
            print(f"mim range = {minimum_range}")
            new_path_list.append(path)
            
    print(new_path_list)
    print(f"the number of data list : {len(new_path_list)}")
    
    with open(output_file, "w") as fp:
        fp.writelines([d+"\n" for d in new_path_list])
    

def main():
    parser = argparse.ArgumentParser(
        description="Determine the object limits and retrieve the file.")
    parser.add_argument("input_file", help="input file name (.txt)")
    parser.add_argument("output_file_path", help="output file name (.txt)")
    parser.add_argument("parent_dir_path", help="parent dir path")
    parser.add_argument("-p", "--pixel_lower_limit",
                        help="lower limit of pixels", type=float)
    
    args = parser.parse_args()
    
    print("start.")
    make_file_selected_above_lower(args.input_file,
                                   args.output_file_path,
                                   args.parent_dir_path,
                                   args.pixel_lower_limit)
    print("fin.")
    

if __name__ == "__main__":
    main()