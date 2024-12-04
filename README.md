# detr
 数学建模大作业：detr复现

参照 official README 下载 COCO2017 数据集，配置环境。

建立 COCO 数据集软链接。

训练代码可以不做修改直接跑通，但考虑到一个 epoch （好像是7900轮次）都需要很长时间，我们放弃训练，而是使用作者预训练好的模型复现测试结果。
A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on a single machine with 8 V100 cards. To ease reproduction of our results we provide results and training logs for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.

使用 https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth 的预训练模型进行测试数据复现。

出现了报错，详见消息记录 12.4 15：08 图片。修改了 coco_eval.py 文件中的细节，即可正常跑通。