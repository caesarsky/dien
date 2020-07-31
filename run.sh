#!/bin/sh
pushd ./script
python data_prepare.py
python generate_voc.py
python train.py train DIEN