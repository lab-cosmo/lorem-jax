#!/bin/bash
export DATASETS="."

# prepare data
python prepare.py

# run training
cd my_experiment
DATASETS=.. lorem-train
