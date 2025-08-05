# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1st 2023
@author: yamadaaiki
"""


import argparse


def make_site_order_num_list(list_length, institution_num, each_institution_epoch):
    
    list_temp1 = []
    list_temp2 = []
    list_temp = []
    
    repetitions_num = (list_length // (institution_num * each_institution_epoch))
    repetitions_remainder = list_length % (institution_num * each_institution_epoch)
    
    for i in range(institution_num):
        list_temp = [i] * each_institution_epoch
        list_temp1 = list_temp1 + list_temp
    
    list_temp2 = list_temp1 * repetitions_num
    
    if not repetitions_remainder == 0:
        list_temp2 = list_temp2 + list_temp1[:repetitions_remainder]
        
    return list_temp2


def main():
    
    parser = argparse.ArgumentParser(
                description='Create institution order num list',
                add_help=True)
    parser.add_argument('-m', '--max_epochs', help='maximum of the number of epochs',
                        type=int, default=50)
    parser.add_argument('-n', '--institution_num', help='the number of institutions for cyclical waight transfer',
                    type=int, default=4, choices=range(1, 21))
    parser.add_argument('-q', '--each_institution_epoch', help='Number of epochs for each insitution',
                    type=int, default=2)
    
    args = parser.parse_args()
    
    institution_order_list = []
    
    institution_order_list = make_site_order_num_list(args.max_epochs, args.institution_num, args.each_institution_epoch)
    
    print(" list len : %d, list : %s" % (len(institution_order_list), institution_order_list))
    
    print('start')
    print(f"linst len: {len(institution_order_list)}, list: {institution_order_list}")
    print('fin')


if __name__ == "__main__":
    main()