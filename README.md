# detr
 数学建模大作业：detr复现

参照 official README 下载 COCO2017 数据集，配置环境。

建立 COCO 数据集软链接。

训练代码可以不做修改直接跑通，但考虑到一个 epoch （好像是7900轮次）都需要很长时间，我们放弃训练，而是使用作者预训练好的模型复现测试结果。
A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on a single machine with 8 V100 cards. To ease reproduction of our results we provide results and training logs for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

使用预训练模型进行测试数据复现。官方复现网页为 https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918。

- 复现 DETR 模型时出现了报错，详见消息记录 12.4 15：08 图片。修改了 coco_eval.py 文件中的细节，即可正常跑通。

- 复现 DETR-DC5 模型时发现 AP 明显比报告的低20个点。然后发现README下面这段文字，将batch_size更新为1。好像效果还是不行。。。
  
  We provide results for all DETR detection models in this gist. Note that numbers vary depending on batch size (number of images) per GPU. Non-DC5 models were trained with batch size 2, and DC5 with 1, so DC5 models show a significant drop in AP if evaluated with more than 1 image per GPU. 
  关于batch_size问题的一些讨论：https://github.com/facebookresearch/detr/issues/217#issuecomment-684087741
  
  我的指令为：
  
  ```
  python main.py --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --coco_path /path/to/coco
  ```
  
  我查看了官方复现网页，他用的指令为：
  
  ```
  python main.py --no_aux_loss --eval \
      --batch_size 1 --dilation \
      --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth \
      --coco_path /path/to/coco
  ```
  
  能够成功复现，与我的唯一区别在于 `--dilation`。查看 Args 部分，该参数的作用是 "If true, we replace stride with dilation in the last convolutional block (DC5)"。
  
- DETR-R101 的复现用上面的 DETR 指令也不太行，会报错，官方指令如下，指定了`--backbone`，否则默认为 resnet50，某些架构不同。

  ```
  python main.py --batch_size 2 --no_aux_loss --eval \
      --backbone resnet101 \
      --resume https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth \
      --coco_path /path/to/coco
  ```

- DETR-R101-DC5 的指令类似：

  ```
  python main.py --no_aux_loss --eval \
      --backbone resnet101 \
      --batch_size 1 --dilation \
      --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
      --coco_path /path/to/coco
  ```

按照以上指令，能够复现。