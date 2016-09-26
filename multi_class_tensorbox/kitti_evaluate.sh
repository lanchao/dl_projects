#!/usr/bin/env bash

echo model path: $1
echo expname: $2
echo gpu: $3

python kitti_evaluation.py --weights $1 --expname $2 --evaluate --no-precision --gpu $3

cd /home/chaolan/kitti_benchmark/
./evaluate_object $2
cd -

python kitti_evaluation.py --weights $1 --expname $2 --no-evaluate --precision --gpu $3