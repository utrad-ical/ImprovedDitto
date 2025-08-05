import argparse
import glob
import os
import sys


def create_trainval_file(input_folder_path, output_file_path):
    
    if not os.path.isdir(input_folder_path):
        raise Exception(f"folder : {input_folder_path} does's not exist.")
    
    path_list = []
    
    path_list = glob.glob(f"{input_folder_path}/*")
    
    for i, path in enumerate(path_list):
        buf = os.path.dirname(path)
        buf_extention = os.path.splitext(os.path.basename(path))[0]
        path_list[i] = f"{buf_extention}"
        
        print(path_list[i])
    
    with open(output_file_path, "w") as fp:
        fp.writelines(f"{path}\n" for path in path_list)
    

def main():
    
    parser = argparse.ArgumentParser(description="create trainval txt file for SSD.",
                                     add_help=True)
    parser.add_argument("input_folder_path",
                        help="folder path for file path")
    parser.add_argument("output_file_name",
                        help="trainval.txt file name and path")
    args = parser.parse_args()
    
    create_trainval_file(args.input_folder_path, args.output_file_name)


if __name__ == "__main__":
    main()