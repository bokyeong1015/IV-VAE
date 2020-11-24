#!/bin/bash

StartTime=$(date +%s)

python main.py --dataset mnist \
--gpu 1 --expNum 1 --useSepaUnit 1 --unsupLearn 2 \
--tcvae_flag 1 --alpha_z 0 --beta_z 32 --gamma_z 1 --alpha_y 0 --beta_y 2 --gamma_y 0 \
--indepLs_flag 1 --lamb_indep 8 \
--lamb_cls 5.1 --temperature 0.67 \
--batchSize 256 --batchSize_eval 1000 --learnRate 0.001 \
--nEpoch 200 --continueFlag 0 --dataSeed 12 --rngNum 1 --labels_per_class 100 \
--modelNum 1 --n_class 10 --latent-dim_z 10 --imSz 32

EndTime=$(date +%s)

echo "*** $(($EndTime - $StartTime)) sec elapsed"
