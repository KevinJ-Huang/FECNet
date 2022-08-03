# Exposure-Consistency Representation Learning for Exposure Correction (ACMMM 2022)

Jie Huang+, Yajing Liu+, Feng Zhao*, Keyu Yan, Jinghao Zhang, Yukun Huang, Man Zhou*, Zhiwei Xiong

*Equal Corresponding Authors

+Equal Contributions

University of Science and Technology of China (USTC)

## Introduction

This repository is the **official implementation** of the paper, "Deep Fourier-based Exposure Correction with Spatial-Frequency Interaction", where more implementation details are presented.

### 0. Hyper-Parameters setting

Overall, most parameters can be set in options/train/train_Enhance.yml 

### 1. Dataset Preparation

Create a .txt file to put the path of the dataset using 

```python
python create_txt.py
```

### 2. Training

```python
python train.py --opt options/train/train_Enhance.yml
```


### 3. Inference

set is_training in "options/train/train_Enhance.yml" as False
set the val:filelist as the validation set. 

then
```python
python train.py --opt options/train/train_Enhance.yml
```

## Dataset 
MSEC dataset (please refer to https://github.com/mahmoudnafifi/Exposure_Correction)

SICE dataset (I have uploaded it to https://share.weiyun.com/C2aJ1Cti)

## Ours Results

coming soon


## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (hj0117@mail.ustc.edu.cn).

## Cite

```
```
