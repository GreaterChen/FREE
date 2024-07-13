# FREE
This repository contains the reference code for the paper "**FREE: Feature Refinement for Generalized Zero-Shot Learning**" accepted to ICCV 2021. [[arXiv]](https://arxiv.org/pdf/2107.13807.pdf)[[Paper]](https://github.com/shiming-chen/FREE)

![](images/pipeline.png)


## 0. CLB tips
运行： 
```bash
python train_free.py
```
跑CUB没问题，但是在ZDFY上发现：
```python
def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    #return (KLD)
    return (BCE + KLD)
```
其中各个变量尺寸：
```python
recon x: torch.Size([64,2048])
x: torch.Size([64,2048])
mean: torch.Size([64,768])
log_var: torch.Size([64,768])
```

这个代码在`optimizer.step()`的时候报错
```python
RuntimeError: CUDA error: device-side assert triggered CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1. Compile with TORCH_USE_CUDA_DSA to enable device-side assertions。
```
参考[这篇文章](https://blog.csdn.net/By_Z0la/article/details/134642425)，具体来说是因为下面这一行有问题。
```python
BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
```
我将其改为了
```python
MSE = torch.nn.functional.mse_loss(recon_x, x.detach(), reduction='sum') / x.size(0)
```


其次，需要注意config中--nz需要等于--attSize

## 1. Preparing Dataset and Model
Datasets can be download from [Xian et al. (CVPR2017)](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and take them into dir `data`.
## Requirements
The code implementation of **FREE** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run in Python 3.8.8.

## 2. Runing
Before running commands, you can set the hyperparameters in config.py. Please run the following commands and testing **FREE** on different datasets: 
```
$ python ./image-scripts/run-cub.py       #CUB
$ python ./image-scripts/run-sun.py       #SUN
$ python ./image-scripts/run-flo.py       #FLO
$ python ./image-scripts/run-awa1.py      #AWA1
$ python ./image-scripts/run-awa2.py      #AWA2
```

**Note**: All of above results are run on a server with one GPU (Nvidia 1080Ti).


## 3. Citation
If this work is helpful for you, please cite our paper.

```
@InProceedings{Chen_2021_ICCV,
    author    = {Chen, Shiming and Wang, Wenjie and Xia, Beihao and Peng, Qinmu and You, Xinge and Zheng, Feng and Shao, Ling},
    title     = {FREE: Feature Refinement for Generalized Zero-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
    pages     = {122-131}
}
```

## 4. Ackowledgement
We thank the following repos providing helpful components in our work.
1. [TF-VAEGAN](https://github.com/akshitac8/tfvaegan)
2. [cycle-CLSWGAN](https://github.com/rfelixmg/frwgan-eccv18)


