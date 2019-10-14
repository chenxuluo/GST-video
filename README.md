# Grouped Spatial-Temporal Aggretation for Efficient Action Recognition

Pytorch implementation of paper Grouped Spatial-Temporal Aggretation for Efficient Action Recognition. [arxiv](https://arxiv.org/abs/1909.13130)


#### Prerequisites
* PyTorch 1.0 or higher
* python 3.5 or higher

### Data preparation
Please refer to [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch) for data preparation on Something-Something.

### Training
* For GST-Large:
`python3 main.py --root_path /path/to/video/folder --dataset somethingv1 --checkpoint_dir /path/for/saving/checkpoints/  --type GST --arch resnet50 --num_segments 8 --beta 1`
* For GST:
` python3 main.py --root_path /path/to/video/folder --dataset somethingv1 --checkpoint_dir /path/for/saving/checkpoints/  --type GST --arch resnet50 --num_segments 8 --beta 2 --alpha 4`
* For more details, please type `python3 main.py -h`

### Pretrained Models
  |                              | Something-v1 | Something-v2 |
  |------------------------------| -------------| -------------|
  |GST(alpha=4, 8 frames)        | 47.0     |  [61.6](https://drive.google.com/file/d/18xiD9C2GS0YcdrAzTUbW7ylv6XbrWkdS/view?usp=sharing)    |
  |GST(alpha=4,16 frames)        | 48.6     |  [62.6](https://drive.google.com/file/d/1llsMnRFEaKLPcCNG4uk2ls9K8iuLVGFi/view?usp=sharing)    |
  |GST-Large(alpha=4,8 frames)   | [47.7](https://drive.google.com/file/d/1TvgmZqv-20P77jKmqWNoJA7tewqJC_nK/view?usp=sharing)     |  [62.0](https://drive.google.com/file/d/1Ymc_NK5WK47Z4qAoVEJI3wBGlbuwaFdw/view?usp=sharing)    |
  
 * results are reported based on center crop  and 1 clip sampling. 




### Reference
If you find our work useful in your research, please consider citing our paper
```
@inproceedings{luo2019grouped,
  title={Grouped Spatial-Temporal Aggretation for Efficient Action Recognition},
  author={Luo, Chenxu and Yuille, Alan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
} 
```
or
```
@article{luo2019grouped,
  title={Grouped Spatial-Temporal Aggregation for Efficient Action Recognition},
  author={Luo, Chenxu and Yuille, Alan},
  journal={arXiv preprint arXiv:1909.13130},
  year={2019}
}
```

#### Acknowledge
This codebase is build upon [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch) and [TSN-pytorch](https://github.com/yjxiong/tsn-pytorch)



