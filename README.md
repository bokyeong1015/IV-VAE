# Independent Vector VAE
PyTorch implementation for paper

* "[Semi-supervised Disentanglement with Independent Vector Variational Autoencoders](https://arxiv.org/abs/2003.06581)," arXiv preprint arXiv:2003.06581 (2020)

## Tested Environment
* {python 2.7, pytorch 0.4.1} OR
* {python 3.6, pytorch 1.1.0}

## Dataset
* Use the following script (~570MB storage needed):
```
bash ./download_data.sh
```
* Alternative: download the data via [this GoogleDrive link](https://drive.google.com/drive/folders/1s5mqyESXE2jUip2iEo1ypIDLJZCCuDtv?usp=sharing) → put them in `data/`
* Note: the images were resized and divided into training, validation, and test sets as follows.
  - **Fashion-MNIST**, **MNIST**: 32×32, # (train, valid, test) imgs = (50K, 10K, 10K)
  - **dSprites**: 64×64, # (train, valid, test) imgs = (614K, 61K, 61K)


## Code
* For Fashion-MNIST and MNIST, run the following scripts. Both use the separate TC setup.
```
bash ./exp_fashMni.sh
bash ./exp_mnist.sh
```
* For dSprites under the separate TC setup, run:
```
bash ./exp_shapes_sepaTc.sh
```
* For dSprites under the collective TC setup, run:
```
bash ./exp_shapes_collecTc.sh
```

## Acknowledgment
* This repository borrows heavily from [beta-TCVAE](https://github.com/rtqichen/beta-tcvae). We thank them for open-sourcing their codes.


## Bibtex
```
@article{kim2020semi,
  title={Semi-supervised Disentanglement with Independent Vector Variational Autoencoders},
  author={Kim, Bo-Kyeong and Park, Sungjin and Kim, Geonmin and Lee, Soo-Young},
  journal={arXiv preprint arXiv:2003.06581},
  year={2020}
}
```  