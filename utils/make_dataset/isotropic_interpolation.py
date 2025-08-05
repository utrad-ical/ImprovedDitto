"""
Created on Thu May 26 09:19:40 2022
@author: yamadaaiki
"""

import argparse
import numpy as np
import skimage.io
from scipy import ndimage

import SimpleITK as sitk
import skimage.measure

from mhd_io import export_raw_data, get_voxel_spacing_from_mhd


def main():
    parser = argparse.ArgumentParser(description='interpolation')
    parser.add_argument('in_mhd_file_name', help='Input volume (mhd) file name')
    parser.add_argument('out_raw_file_name', help='Output volume (raw) file name')
    parser.add_argument('-v', '--iso_voxel_size', help='Output voxel size',
                        type=float, default=1.0)
    parser.add_argument('-o', '--order',
                        help='Order of interpolation (1: trilinear, 3: tricubic)',
                        type=int, default=1)
    args = parser.parse_args()
    
    # Get voxel size from mhd file
    org_voxel_size = np.array(get_voxel_spacing_from_mhd(args.in_mhd_file_name))[::-1]
    
    # Load volume data via SimpleITK
    in_volume = skimage.io.imread(args.in_mhd_file_name, plugin='simpleitk')
    
    if(args.order == 3):
        zoom_ratio = org_voxel_size / (0.5 * args.iso_voxel_size)
        tmp_volume = ndimage.zoom(in_volume, zoom_ratio, order=3)
        tmp_volume = np.pad(tmp_volume,
                            ((0, tmp_volume.shape[0] % 2),
                             (0, tmp_volume.shape[1] % 2),
                             (0, tmp_volume.shape[2] % 2)),
                            'edge')
        out_volume = skimage.measure.block_reduce(tmp_volume,
                                                  (2,2,2),
                                                  np.average).astype(np.int16)
    else:
        structure = np.ones([3,3,3])
        zoom_ratio = org_voxel_size / args.iso_voxel_size
        tmp_volume = (in_volume > 0).astype(np.uint8) * 255
        out_volume = ndimage.zoom(tmp_volume, zoom_ratio, order=1)
        out_volume = (out_volume >= 128).astype(np.uint8)    
        out_volume, _ = ndimage.label(out_volume, structure=structure)
        out_volume = out_volume.astype(np.uint8)
    
    print(in_volume.shape, out_volume.shape)
    iso_voxel_size = (args.iso_voxel_size, args.iso_voxel_size, args.iso_voxel_size)
    export_raw_data(args.out_raw_file_name, out_volume, iso_voxel_size)


if __name__ == '__main__':
    main()