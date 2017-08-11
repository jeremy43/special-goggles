# Flow-Guided-Feature-Aggregation for Video Recognition


The major contributors of this repository include [Xizhou Zhu](https://github.com/einsiedler0408), [Yuwen Xiong](https://github.com/Orpine), [Yuqing Zhu](), [Shuhao Fu](https://github.com/howardmumu) [Jifeng Dai](https://github.com/daijifeng001), [Lu Yuan](http://www.lyuan.org/), and  [Yichen Wei](https://github.com/YichenWei).

## Introduction


**Flow-Guided Feature Aggregation** is initially described in a [CVPR 2017 paper](https://arxiv.org/abs/1611.07715). It provides an accurate and end-to-end framework for video recognition (e.g., object detection and semantic segmentation in videos). It is worth noting that:

* Flow-Guided Feature Aggregation significantly improves the video recognition accuracy by applying the heavy-weight image recognition network (e.g., ResNet-101) on an interval of frames, and aggregating nearby features along the motion paths to produce a more accurate prediction with the leverage of temporal coherence on feature level.
* The entire system is end-to-end trained for the task of video recognition, which is vital for improving the recognition accuracy. Directly adopting state-of-the-art flow estimation methods without end-to-end training would deliver noticable worse results.
* Flow-Guided Feature Aggregation overperforms the strong single-frame baselines in ImageNet VID, especially for more challenging fast moving objects

***Click image to watch our demo video***

[![Demo Video on YouTube](https://media.giphy.com/media/7D9tmDgzB10HK/giphy.gif)](https://www.youtube.com/watch?v=R2h3DbTPvVg)

## Disclaimer

This is an official implementation for [Flow-Guided Feature Aggregation for Video Recognition](https://arxiv.org/abs/1611.07715) (FGFA) based on MXNet. It is worth noticing that:

  * The original implementation is based on our internal Caffe version on Windows. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.
  * The code is tested on official [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) with the extra operators for Deep Feature Flow.
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) model and [Flying Chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) pre-trained [FlowNet](https://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz) model using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository used code from [MXNet rcnn example](https://github.com/dmlc/mxnet/tree/master/example/rcnn) and [mx-rfcn](https://github.com/giorking/mx-rfcn).




## License

© Microsoft, 2017. Licensed under an Apache-2.0 license.

## Citing Flow-Guided Feature Aggregation

If you find Flow-Guided Feature Aggregation useful in your research, please consider citing:
```
@inproceedings{zhu17dff,
    Author = {Xizhou Zhu, Yuwen Xiong, Yuqing Zhu, Shuhao Fu, Jifeng Dai, Lu Yuan, Yichen Wei},
    Title = {Flow-Guided Feature Aggregation for Video Object Detection},
    Conference = {CVPR},
    Year = {2017}
}

@inproceedings{dai16rfcn,
    Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
    Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
    Conference = {NIPS},
    Year = {2016}
}
```

## Main Results


|                                 | <sub>training data</sub>     | <sub>testing data</sub> | <sub>mAP@0.5</sub> | <sub>time/image</br> (Tesla K40)</sub> | <sub>time/image</br>(Maxwell Titan X)</sub> |
|---------------------------------|-------------------|--------------|---------|---------|--------|
| <sub>Frame baseline</br>(R-FCN, ResNet-v1-101)</sub>                    | <sub>ImageNet DET train + VID train</sub> | <sub>ImageNet VID validation</sub> | 74.1    | 0.271s    | 0.133s |
| <sub>Flow-Guided Feature Aggregationw</br>(R-FCN, ResNet-v1-101, FlowNet)</sub>           | <sub>ImageNet DET train + VID train</sub> | <sub>ImageNet VID validation</sub> | 73.0    | 0.073s    | 0.034s |

*Running time is counted on a single GPU (mini-batch size is 1 in inference, key-frame duration length for Flow-Guided-Feature-Aggregation is 10).*

*The runtime of the light-weight FlowNet seems to be a bit slower on MXNet than that on Caffe.*

## Evaluation Results


|                                 | <sub>small</sub>     | <sub>middle</sub> | <sub>large</sub> | <sub>all area</sub> |
|---------------------------------|----------------------|-------------------|------------------|---------------------|
| <sub>mAP(%)(slow)</sub>         | 0.3803 | 0.4886 | 0.8732   | 0.8258    |
| <sub>mAP(%)(medium)</sub>       | 0.3475 | 0.5292 | 0.8732   | 0.8258    |
| <sub>mAP(%)(fast)</sub>         | 0.2439 | 0.4097 | 0.6341   | 0.4839    |
| <sub>mAP(%)(all motion)</sub>   | 0.2367 | 0.4858 | 0.8340   | '0.7293'  |

*Detection accuracy of small (area < 50^2 pixels), medium (50^2 ≤ area ≤ 150^2 pixels), and large (area > 150^2 pixels) object instances with respect to slow (motion iou < 0.7), medium (0.7 ≤ motion iou ≤ 0.9), and fast (motion iou > 0.9) object instances.
*The last row is the accuracy of object instances with different areas without considering instance motions, and the last column is the accuracy of object instances with different motions without considering instance areas.
*The last cell is the mAP of all object instances.


## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install Cython
	pip install opencv-python==3.2.0.6
	pip install easydict==1.6
	```
3. For Windows users, Visual Studio 2015 is needed to compile cython module.


## Requirements: Hardware

Any NVIDIA GPUs with at least 6GB memory should be OK

## Installation

1. Clone the Flow-Guided Feature Aggregation repository
~~~
git clone https://github.com/msracver/Flow-Guided-Feature-Aggregation.git
~~~
2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.
3. Copy operators in `./rfcn/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` and recompile MXNet.
4. Please install MXNet following the official guide of MXNet. For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.


## Demo


1. To run the demo with our trained model (on ImageNet DET + VID train), please download the model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMPLjGGCvAeciQflg), and put it under folder `model/`.

	Make sure it looks like this:
	```
	./model/rfcn_vid-0000.params
	./model/rfcn_fgfa_flownet_vid-0000.params
	```
2. Run (inference batch size = 1)
	```
	python ./rfcn/demo.py
	python ./fgfa_rfcn/demo.py
	```
	or run (inference batch size = 10)
	```
	python ./rfcn/demo_batch.py
	python ./fgfa_rfcn/demo_batch.py
	```

## Preparation for Training & Testing

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset, and make sure it looks like this:

	```
	./data/ILSVRC2015/
	./data/ILSVRC2015/Annotations/DET
	./data/ILSVRC2015/Annotations/VID
	./data/ILSVRC2015/Data/DET
	./data/ILSVRC2015/Data/VID
	./data/ILSVRC2015/ImageSets
	```

2. Please download ImageNet pre-trained ResNet-v1-101 model and Flying-Chairs pre-trained FlowNet model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMOBdCBiNaKbcjPrA), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	./model/pretrained_model/flownet-0000.params
	```

## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/{rfcn/fgfa_rfcn}/cfgs`.

2. Two config files have been provided so far, namely, Frame baseline with R-FCN and Flow-Guided-Feature-Aggregation with R-FCN for ImageNet VID. We use 4 GPUs to train models on ImageNet VID.

3. To perform experiments, run the python script with the corresponding config file as input. For example, to train and test Flow-Guided Feature Aggregation with R-FCN, use the following command
    ```
    python experiments/fgfa_rfcn/fgfa_rfcn_end2end_train_test.py --cfg experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml
    ```
	A cache folder would be created automatically to save the model and the log under `output/fgfa_rfcn/imagenet_vid/`.
    
4. Please find more details in config files and in our code.

## Misc.

Code has been tested under:

- Ubuntu 14.04 with a Maxwell Titan X GPU and Intel Xeon CPU E5-2620 v2 @ 2.10GHz
- Windows Server 2012 R2 with 8 K40 GPUs and Intel Xeon CPU E5-2650 v2 @ 2.60GHz
- Windows Server 2012 R2 with 4 Pascal Titan X GPUs and Intel Xeon CPU E5-2650 v4 @ 2.30GHz

## FAQ

Q: It says `AttributeError: 'module' object has no attribute 'MultiProposal'`.

A: This is because either
 - you forget to copy the operators to your MXNet folder
 - or you copy to the wrong path
 - or you forget to re-compile and install
 - or you install the wrong MXNet

    Please print `mxnet.__path__` to make sure you use correct MXNet

<br/><br/>
Q: I encounter `segment fault` at the beginning.

A: A compatibility issue has been identified between MXNet and opencv-python 3.0+. We suggest that you always `import cv2` first before `import mxnet` in the entry script. 

<br/><br/>
Q: I find the training speed becomes slower when training for a long time.

A: It has been identified that MXNet on Windows has this problem. So we recommend to run this program on Linux. You could also stop it and resume the training process to regain the training speed if you encounter this problem.

<br/><br/>
Q: Can you share your caffe implementation?

A: Due to several reasons (code is based on a old, internal Caffe, port to public Caffe needs extra work, time limit, etc.). We do not plan to release our Caffe code. Since a warping layer is easy to implement, anyone who wish to do it is welcome to make a pull request.
