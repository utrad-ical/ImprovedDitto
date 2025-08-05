"""
Created on Thr Jan 18 2022
@author: ynomura
"""

import numpy as np
import os
import SimpleITK as sitk


def get_voxel_spacing_from_mhd(mhd_file_name):

    reader = sitk.ImageFileReader()
    reader.SetFileName(mhd_file_name)
    reader.ReadImageInformation()
    
    return reader.GetSpacing()


def get_met_type_from_numpy_type(dtype):
    if dtype == np.int8:
        return 'MET_CHAR'
    elif dtype == np.uint8:
        return 'MET_UCHAR'
    elif dtype == np.int16:
        return 'MET_SHORT'
    elif dtype == np.uint16:
        return 'MET_USHORT'
    elif dtype == np.int32:
        return 'MET_INT'
    elif dtype == np.uint32:
        return 'MET_UINT'
    elif dtype == np.float32:
        return 'MET_FLOAT'
    elif dtype == np.float64:
        return 'MET_DOUBLE'
    return 'MET_OTHER'


def export_mhd(raw_file_name, volume, voxel_spacing):

    # export raw data
    volume.tofile(raw_file_name)
    
    # set file name of raw data
    base_dir_pair = os.path.split(raw_file_name)
    mhd_file_name = os.path.join(base_dir_pair[0],
                                    os.path.splitext(base_dir_pair[1])[0] + ".mhd")
    
    if volume.ndim == 3:
        size_str = f'DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}'
        spacing_str = f'ElementSpacing = {voxel_spacing[0]} {voxel_spacing[1]} {voxel_spacing[2]}'
    else:
        size_str = f'DimSize = {volume.shape[1]} {volume.shape[0]}'
        spacing_str = f'ElementSpacing = {voxel_spacing[0]} {voxel_spacing[1]}'
        
    element_type_str = "ElementType = " + get_met_type_from_numpy_type(volume.dtype)
    
    with open(mhd_file_name, mode="w") as fp:
        fp.write("ObjectType = Image\r\n")
        fp.write(f"NDims = {volume.ndim}\r\n")
        fp.write(f"{size_str}\r\n")
        fp.write(f"{element_type_str}\r\n")
        fp.write(f"{spacing_str}\r\n")
        fp.write("ElementByteOrderMSB = False\r\n")
        fp.write(f"ElementDataFile = {base_dir_pair[1]}\r\n")
