#!/bin/bash

# Run inference for ForestNet
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s train
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s val
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s test

# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s train
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s val
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s test

# python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s train
 python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s val
 python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s test

# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s train
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s val
 python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s test

# Run inference for BigEarthNet
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s train
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s val
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s test

#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s train
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s val
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s test

#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s train
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s val
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s test

#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s train
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s val
python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s test

# Run inference with vanilla ViT
# python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s train
 python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s val
 python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s test

# python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s train
 python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s val
 python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s test

#python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s train
python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s val
python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s test

#python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s train
python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s val
python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s test
