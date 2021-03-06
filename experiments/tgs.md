# 0
## condition
UNet feature 24 depth 3
val_split:0.2
lr 0.01
cyclic 5 epoch 0.5 - 1.0
batch_size 32
SGD
cross entropy
data clean

## result
54 epoch (1h 7m)
val iou 0.68
train iou 0.88
val loss 0.28
train loss 0.022

## memo
overfitted

# 1
## condition
RUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch: 0.5 - 1.0
SGD
cross entropy
data clean
augmentation: hflip

## result
400 epoch(4h 41m)
### peak at epoch 313
val iou 0.7897
train iou 0.883
val loss 0.18
train loss 0.021

### end at epoch 399
val iou 0.763
train iou 0.893
val loss 0.19
train loss 0.018


# 2
## condition
DUNet feature 8 depth 3
val_split:0.2
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.5 - 1.0
SGD
cross entropy
data clean

## result
val iou |0.77|0.74|
train iou |0.87|0.86|
val loss |0.027|0.025|
train loss |0.13|0.14|

# memo
デーコーダはパラメータが多いとoverfitを抑制できる

# 3
## condition
RUNet feature 8 depth 3
val_split:0.2
batch_size: 64
lr: 0.01
cyclic: 5 epoch 0.5 - 1.0
SGD
lovasz
data clean
augmentation: hflip

## result
epoch 375 (3h 55m)
val iou 0.76
train iou 0.83
val loss 0.11 
train loss 0.04

## memo
lovasz lossはoverfitを抑制できる

# 4
## condition
RUNet feature 16 depth 3
epoch 800
val_split:0.15
batch_size: 32
lr: 0.01
cyclic: 5 epoch 0.2 - 0.8
SGD
lovasz
data clean
augmentation: hflip

## result
epoch 375 (3h 55m)
val iou 0.69
train iou 0.84
val loss 0.11 
train loss 0.05
## memo
encoderのチャネル数が多すぎるとoverfitする？

# 5
## condition
RUNet feature 8 depth 3
val_split:0.2
batch_size: 64
lr: 0.01
cyclic: 5 epoch 0.2 - 0.8
SGD
lovasz
data clean
augmentation: hflip

## result
epoch 800 (9h 1m)
val iou 0.74
train iou 0.86
val loss 0.11 
train loss 0.05
## memo
encoderのパラメータが多すぎるとoverfitする
