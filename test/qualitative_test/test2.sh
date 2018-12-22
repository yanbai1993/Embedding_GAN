#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data  --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch B1 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch B2 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch B3 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch B4 
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot test_data --name neg_all_epoch --model test --phase test --no_dropout --display_id 0 --dataset_mode single --which_epoch B5 


