# 重大修订 (Major Revisions)

需要特别指出的是，我对论文进行了重大修改，一旦论文被录取，我会给出最新的评估指标(以前的指标评估略高，由于我的疏忽将部分用于测试代码可行性的训练数据集当当成了测试集用于指标测试，导致了精度的问题)。

It must be pointed out that I have made major revisions to the paper. Once the paper is accepted, I will give the latest evaluation indicators (the previous indicators were slightly higher. Due to my negligence, some training data used to test the feasibility of the code was used as test data, which eventually led to accuracy issues).

## Abstract

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

## 最新的工作 (The latest open source code)
https://github.com/JACKYLUO1991/FaceParsing.

## Please cited

```
@article{luo2019real,
  title={Real-time Segmentation and Facial Skin Tones Grading},
  author={Luo, Ling and Xue, Dingyu and Feng, Xinglong and Yu, Yichun and Wang, Peng},
  journal={arXiv preprint arXiv:1912.12888},
  year={2019}
}
```

