#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data  --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A1 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A2 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A3 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A4 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A5_1 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch A5_2_

