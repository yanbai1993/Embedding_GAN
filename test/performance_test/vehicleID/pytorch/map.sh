#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python mAP2.py --list_file /data/CarReID/train_test_split/convert_view_test_800.txt --image_dir /data/CarReID/cropped/ --net runs/Soft_trip_Net_0.33_10/checkpoint.pth.tar
