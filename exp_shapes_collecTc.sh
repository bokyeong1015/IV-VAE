#!/bin/bash

StartTime=$(date +%s)

python main.py --dataset shapes \
--gpu 1 --expNum 1 --useSepaUnit 2 --unsupLearn 2 \
--tcvae_flag 1 --alpha_h 0 --beta_h 1 --gamma_h_z 1 --gamma_h_y 2 \
--indepLs_flag 1 --lamb_indep 0.5 \
--lamb_cls 51 --temperature 0.75 \
--batchSize 1024 --batchSize_eval 1000 --learnRate 0.001 \
--nEpoch 100 --continueFlag 0 --dataSeed 12 --rngNum 1 --labels_per_class 4096 \
--modelNum 2 --n_class 3 --latent-dim_z 6 --imSz 64

EndTime=$(date +%s)

echo "*** $(($EndTime - $StartTime)) sec elapsed"
