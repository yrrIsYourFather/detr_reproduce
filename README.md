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
   ln -s data2/coco2017/ /home/ruiran/detr/datasets/coco
   ln -s data2/coco-panoptic /home/ruiran/detr/datasets/coco-panoptic
   ```

4. 作者开源的训练代码可以不做修改直接跑通。但考虑到训练时长和训练资源问题（8 张 V100 并行训练 300 轮次大约需要 6 天），我放弃训练，而是使用作者训练好的模型复现测试结果。

## 复现测试结果

这一部分我主要在作者源代码 `main.py` 文件上进行修改，并参考了[这一网页](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918)，整合得到测试结果复现代码 `reproduce.py`。下面我将分别阐释 DETR 在目标检测和全景分割两个任务（分别对应 coco2017 和 coco-panoptic 数据集）的复现方式及结果。

### 目标检测任务复现

==阐释各个模型之间的区别，各个模型的训练是如何设置的？==

==DC5模型是什么？可以修正翻译==

#### 复现指令

==需要阐释指令中各个参数的意义，==

能够成功复现，与我的唯一区别在于 `--dilation`。查看 Args 部分，该参数的作用是 "If true, we replace stride with dilation in the last convolutional block (DC5)"。

复现用上面的 DETR 指令也不太行，会报错，官方指令如下，指定了`--backbone`，否则默认为 resnet50，某些架构不同。

1. **[DETR-Resnet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)** 

   ```
   python reproduce.py --no_aux_loss --eval \
   	--batch_size 2 \
   	--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
       --coco_path /home/ruiran/detr/datasets/coco
   ```

2. **[DETR-Resnet50-DC5](https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth)**

   ```
   python reproduce.py --no_aux_loss --eval \
       --batch_size 1 --dilation \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
       --coco_path /home/ruiran/detr/datasets/coco
   ```

   在这一模型的复现过程中，我注意到 `batch_size` 的设置对于复现结果的显著影响。`batch_size=2` 的结果在各项指标上均比 `batch_size=1` 下降了 6~7 个点，例如 `mAP` 由 43.2 降低至 36.0 。我将 DETR-Resnet50 测试阶段的 `batch_size` 同样调整至 1，测试结果却并没有出现类似的大幅波动。我对此感到疑惑，并在作者源代码仓库 [issue#217](https://github.com/facebookresearch/detr/issues/217#issuecomment-684087741) 中找到前人对这一问题的探讨，我了解到：这一问题与模型的训练 `batch_size` 设置紧密相关。DETR-Resnet-DC5 模型训练过程中设置 `batch_size=1`==为什么设为1？==，导致模型从未见过填充（padding），因此在测试集 `batch_size=2` 存在 padding 的情况下性能明显不足；而 DETR-Resnet50 模型训练过程中 `batch_size=4`，不存在这一问题，且模型对不同数量的 padding 具有较好的鲁棒性。

3. **[DETR-Resnet101](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth)** 

   ```
   python reproduce.py --no_aux_loss --eval \
       --backbone resnet101 \
       --batch_size 2 \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth \
       --coco_path /home/ruiran/detr/datasets/coco
   ```

4. **[DETR-Resnet101-DC5](https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth)**

   ```
   python reproduce.py --no_aux_loss --eval \
       --backbone resnet101 \
       --batch_size 1 --dilation \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
       --coco_path /home/ruiran/detr/datasets/coco
   ```

#### 复现结果

| 模型名             |  AP  | AP$_{50}$ | AP$_{75}$ | AP$_\text{S}$ | AP$_\text{M}$ | AP$_\text{L}$ |
| :----------------- | :--: | :-------: | :-------: | :-----------: | :-----------: | :-----------: |
| DETR               |      |           |           |               |               |               |
| DETR-DC5           |      |           |           |               |               |               |
| DETR-Resnet101     |      |           |           |               |               |               |
| DETR-Resnet101-DC5 |      |           |           |               |               |               |

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