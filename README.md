# train PyTorch:MNIST, inference OpenVINO:C++

## Introduction

MNIST を PyTorch で学習、OpenVINO C++ で推論するサンプルです。


## Do it.


```
% cd tool
% python convert_csv.py

% cd python
% sh 00train.sh

% mkdir build
% cd build
% cmake ../cpp
% make


% ./classify -i ../data/mnist_test.csv -m ../python/result 
model: ../python/result/keypoints.xml
MNIST : digit
count: 9617 / 10000
accuracy: 0.9617
```

## environment

- MacOS Big Sur (Intel and Apple Silion)
- official: OpenVINO 2021.2 (Python3.7)
- build myself: [Apple Silicon(M1): build PyTorch-master, OpenVINO-2021.2, OpenCV-4.5.1-openvino](https://qiita.com/kose3/items/2863791559cad7a3932f)
