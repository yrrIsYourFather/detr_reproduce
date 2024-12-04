# DETR 测试结果复现报告

## 前期准备

1. 下载复现代码至主文件夹。

   ```
   git clone https://github.com/yrrIsYourFather/detr_reproduce.git
   ```

2. 配置环境。

   ```
   conda create -n detr
   conda activate detr
   conda install -c pytorch pytorch torchvision
   conda install cython scipy
   pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
   pip install git+https://github.com/cocodataset/panopticapi.git
   ```

3. 数据集准备：在 [COCO 数据集官网](https://cocodataset.org/#download) 下载 coco2017 和 coco-panoptic 数据集至服务器 data 盘，文件目录结构如下：

   ```
   data2
   - coco2017
       - annotations/  				# annotation json files
       - train2017/    				# train images
       - val2017/      				# val images
   - coco-panoptic
       - annotations/  				# annotation json files
       - panoptic_train2017/    		# train panoptic annotations
       - panoptic_val2017/      		# val panoptic annotations
   ```

   在主文件夹下建立数据集软链接。

   ```
   ln -s data2/coco2017/ home/ruiran/detr/datasets/coco
   ln -s data2/coco-panoptic home/ruiran/detr/datasets/coco-panoptic
   ```

4. 作者开源的训练代码可以不做修改直接跑通。但考虑到训练时长和训练资源问题（8 张 V100 并行训练 300 轮次大约需要 6 天），我放弃训练，而是使用作者训练好的模型复现测试结果。

## 复现测试结果

这一部分我主要在作者源代码 `main.py` 文件上进行修改，并参考了[这一网页](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918)，整合得到测试结果复现代码 `reproduce.py`。下面我将分别阐释 DETR 在目标检测和全景分割两个任务（分别对应 coco2017 和 coco-panoptic 数据集）的复现方式及结果。

### 目标检测任务复现

==阐释各个模型之间的区别==

#### 复现指令

==需要阐释指令中各个参数的意义，==

1. **[DETR-Resnet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)** 

   ```
   python reproduce.py --no_aux_loss --eval \
   	--batch_size 2 \
   	--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
       --coco_path home/ruiran/detr/datasets/coco
   ```

2. **[DETR-Resnet50-DC5](https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth)**

   复现 DETR-DC5 模型时发现 AP 明显比报告的低20个点。然后发现README下面这段文字，将batch_size更新为1。好像效果还是不行。。。

   ```
   python main.py --no_aux_loss --eval \
       --batch_size 1 --dilation \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
       --coco_path home/ruiran/detr/datasets/coco
   ```

   在这一模型的复现过程中，我注意到 `batch_size` 的设置对于复现结果的显著影响。

   We provide results for all DETR detection models in this gist. Note that numbers vary depending on batch size (number of images) per GPU. Non-DC5 models were trained with batch size 2, and DC5 with 1, so DC5 models show a significant drop in AP if evaluated with more than 1 image per GPU. 
   关于batch_size问题的一些讨论：https://github.com/facebookresearch/detr/issues/217#issuecomment-684087741

   

   能够成功复现，与我的唯一区别在于 `--dilation`。查看 Args 部分，该参数的作用是 "If true, we replace stride with dilation in the last convolutional block (DC5)"。

   ==注意batchsize问题，需要说明==

3. **[DETR-Resnet101](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth)** 

   的复现用上面的 DETR 指令也不太行，会报错，官方指令如下，指定了`--backbone`，否则默认为 resnet50，某些架构不同。

   ```
   python main.py --no_aux_loss --eval \
       --backbone resnet101 \
       --batch_size 2 \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth \
       --coco_path home/ruiran/detr/datasets/coco
   ```

4. **[DETR-Resnet101-DC5](https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth)**

   ```
   python main.py --no_aux_loss --eval \
       --backbone resnet101 \
       --batch_size 1 --dilation \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
       --coco_path home/ruiran/detr/datasets/coco
   ```

#### 复现结果



### 全景分割任务复现

#### 复现指令

1. **[DETR-Resnet50](https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth)**
2. **[DETR-Resnet50-DC5](https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth)**
3. **[DETR-Resnet101](https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth)**

#### 复现结果

==需要复现日志==

## 局限性

根据论文中的测试结果，DETR 在 small objects 层面不如同层次的 Faster R-CNN

小目标检测（SOD）:[2020-2023年Transformer在小目标检测领域应用综述 - 知乎](https://zhuanlan.zhihu.com/p/656402058)

[Small object detection by DETR via information augmentation and adaptive feature fusion](https://x.sci-hub.org.cn/target?link=https://dl.acm.org/doi/abs/10.1145/3664524.3675362) 专门提到 DETR 在小目标检测上的改进方法