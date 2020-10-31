## HLNet: A Unified Framework for Real-Time Segmentation and Facial Skin Tones Evaluation

## Abstract:
Real-time semantic segmentation plays a crucial role in industrial applications, such as
autonomous driving, the beauty industry, and so on. It is a challenging problem to balance the
relationship between speed and segmentation performance. To address such a complex task, this
paper introduces an efficient convolutional neural network (CNN) architecture named HLNet for
devices with limited resources. Based on high-quality design modules, HLNet better integrates
high-dimensional and low-dimensional information while obtaining sufficient receptive fields, which
achieves remarkable results on three benchmark datasets. To our knowledge, the accuracy of skin
tone classification is usually unsatisfactory due to the influence of external environmental factors such
as illumination and background impurities. Therefore, we use HLNet to obtain accurate face regions,
and further use color moment algorithm to extract its color features. Specifically, for a 224 × 224
input, using our HLNet, we achieve 78.39% mean IoU on Figaro1k dataset at over 17 FPS in the case
of the CPU environment. We further use the masked color moment for skin tone grade evaluation
and approximate 80% classification accuracy demonstrate the feasibility of the proposed method.  

## The latest open source code:
https://github.com/JACKYLUO1991/FaceParsing.

## This work is accepted by Symmetry-Basel!

## Please cited:
```
@article{luo2019real,
  title={Real-time Segmentation and Facial Skin Tones Grading},
  author={Luo, Ling and Xue, Dingyu and Feng, Xinglong and Yu, Yichun and Wang, Peng},
  journal={arXiv preprint arXiv:1912.12888},
  year={2019}
}
```

