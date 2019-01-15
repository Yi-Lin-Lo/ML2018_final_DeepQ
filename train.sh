#!/bin/bash
# Usage: bash train.sh [image_dir_path] [train.csv]

# The script create models (model_1.h5~ model_13.h5) to the directory "model."  (The folder "model" will created by the script) 
# You should give [image_dir_path] [train.csv] as script input parameters.
# To run this script, you can type like this:   bash train.sh ../final/images ../final/train.csv

# This procedure takes about 12 hours on geForce 1080 Ti (10Gb). Please wait patiently.

mkdir model
python3 train_1.py $1 $2 model_1.h5 #224
python3 train_2.py $1 $2 model_2.h5 #224
python3 train_3.py $1 $2 model_3.h5 #299
python3 train_3.py $1 $2 model_4.h5 #299

python3 train_5.py $1 $2 model_5.h5   #224
python3 train_6.py $1 $2 model_6.h5   #224
python3 train_7.py $1 $2 model_7.h5   #224
python3 train_8.py $1 $2 model_8.h5   #299
python3 train_9.py  $1 $2 model_9.h5  #299
python3 train_10.py $1 $2 model_10.h5 #224

python3 train_11.py $1 $2 model_11.h5 #299
python3 train_12.py $1 $2 model_12.h5 #299

python3 train_13.py $1 $2 model_13.h5 #299

