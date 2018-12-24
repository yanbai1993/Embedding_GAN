#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python mAP_vehicleID.py --list_file test_list/convert_view_test_800.txt --image_dir CarReID/cropped/ 
