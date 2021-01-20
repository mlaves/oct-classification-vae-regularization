#!/bin/sh

rm -rf __pycache__/
BS=64
python -u train.py --bs ${BS} --epoch 100 --model vae --ls 512 | tee `date '+%Y-%m-%d_%H-%M-%S'`_vae_2000.log
python -u train.py --bs ${BS} --epoch 50 --model resnet | tee `date '+%Y-%m-%d_%H-%M-%S'`_resnet_2000.log
