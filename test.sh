#!/bin/bash
# Usage: bash test.sh [image_dir_path] [test.csv]

# You should run train.sh first !!!!!!!!!!!

# The script will generate ans.csv by models in folder "model," and output ans.csv to folder "single_ans"
# The script requires 2 parameter.  you can type like this:  bash test.sh ../final/image ../final/test.csv
# This procedure takes about 10 hours on geForce 1080 Ti (10Gb), and take about 15 GB memory. Please wait patiently.

# After the script done, you should run ensemble.sh !!!!!!!!
  
mkdir single_ans

python3 test_ten_crop_224.py $1 $2 ./model/model_1.h5 ./single_ans/ans_1.csv
python3 test_ten_crop_224.py $1 $2 ./model/model_2.h5 ./single_ans/ans_2.csv
python3 test_ten_crop_299.py $1 $2 ./model/model_3.h5 ./single_ans/ans_3.csv
python3 test_ten_crop_299.py $1 $2 ./model/model_4.h5 ./single_ans/ans_4.csv

python3 test_ten_crop_224.py $1 $2 ./model/model_5.h5 ./single_ans/ans_5.csv
python3 test_ten_crop_224.py $1 $2 ./model/model_6.h5 ./single_ans/ans_6.csv
python3 test_ten_crop_224.py $1 $2 ./model/model_7.h5 ./single_ans/ans_7.csv


python3 test_ten_crop_299.py $1 $2 ./model/model_8.h5 ./single_ans/ans_8.csv
python3 test_ten_crop_299.py $1 $2 ./model/model_9.h5 ./single_ans/ans_9.csv

python3 test_ten_crop_224.py $1 $2 ./model/model_10.h5 ./single_ans/ans_10.csv


python3 test_ten_crop_299.py $1 $2 ./model/model_11.h5 ./single_ans/ans_11.csv
python3 test_ten_crop_299.py $1 $2 ./model/model_12.h5 ./single_ans/ans_12.csv
python3 test_ten_crop_299.py $1 $2 ./model/model_13.h5 ./single_ans/ans_13.csv


