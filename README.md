# Final-Year-Thesis-Retinal-Vessel-Segmentation-
This project is based on my final year thesis i.e. Retinal Vessel Segmentation using pytorch framework. This repository includes dataset preparation, data processing, training, testing, and visualization.

## Table of Content
1. [Requirements](#requirements)

## Requirements: <a name="requirements"></a>
It is highly recommended to first try using your current python environment. However, the following environment is successfully able to run the code.

```
Libraries                 Version

1.  python                 >= 3.5
2.  torch                  1.8.1
3.  torchvision            0.9.1
4.  cudnn
5.  tensorboardX           2.2
6.  argparse               1.1
7.  pandas                 1.1.5
8.  matplotlib             3.2.2
9.  opencv                 4.1.2
10. numpy                  1.19.5
```


## 1. Dataset Preparation
* A total of 5 different publicly available retinal image datasets were used for experimentation purpose. Please download the datasets from the following official address: [DRIVE](https://drive.grand-challenge.org/), [STARE]( https://cecas.clemson.edu/~ahoover/stare/), [CHASE_DB1]( https://blogs.kingston.ac.uk/retinal/chasedb1/), [DIARET_DB1](http://www.it.lut.fi/project/imageret/diaretdb1/), [HRF]( https://www5.cs.fau.de/research/data/fundus-images/). These datasets consist of the retinal images, its manual segmentation, mask. We also used some proprietary dataset.


## 2. Training
* Update the config.py file according to your requirement and then run train.py file. 


## 3. Testing
