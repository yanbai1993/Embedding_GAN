#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python cmc_pytorch.py --list_file convert_view_test_800.txt --image_dir CarReID/cropped/ --save cmc.txt
