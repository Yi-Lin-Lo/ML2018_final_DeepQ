#!/bin/bash
# Usage: bash ensemble.sh

# Please run test.sh first !!!!!! Be sure that you have answers in folder "single_ans."


# This script ensemble all answers in folder "single_ans", and tries to give different weights to each ans.csv.
# It will generate 10 answer files, and output them to folder "ensemble_ans." (ensemble_1.csv to ensemble_10.csv)

mkdir ensemble_ans
python3 ans_ensemble.py ./single_ans ./ensemble_ans
