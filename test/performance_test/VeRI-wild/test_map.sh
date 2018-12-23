#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_mAP.py --image_dir /media/p100/Dataset_CAR_LOU/images/ --test_ext 'jpg' --test_list_file /media/p100/Dataset_CAR_LOU/train_test_split/test_1000.txt --query_list_file /media/p100/Dataset_CAR_LOU/train_test_split/test_1000_query.txt --batch_size 3 --feat_size 1024 --ID_net VGGM --resume runs/Soft_trip_Net_0.33_10/checkpoint.pth.tar 



