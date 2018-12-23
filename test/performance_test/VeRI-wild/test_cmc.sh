#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_1000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_1000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_2000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_2000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_3000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_3000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_5000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_5000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_8000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_8000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 

CUDA_VISIBLE_DEVICES=0 python cmc_pytorch_CVPR.py --image_dir /media/p100/Extreme\ SSD/Dataset_CAR_LOU/images/ --ext '.jpg' --list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_10000.txt --query_list_file /media/p100/Extreme\ SSD/Dataset_CAR_LOU/train_test_split/test_10000_query.txt --save_dir runs/Triplet_Soft_Net_0.33/checkpoint.pth.tar 



