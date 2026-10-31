# 猫品种识别

输入一张猫的照片，程序识别一个品种。项目使用 Oxford-IIIT Pet 中的 12 个猫品种，提供命令行预测和本地桌面界面。

网络用 PyTorch 的基础层手写 ResNet18。`model.py` 中可以直接看到两层卷积的残差块、跳跃连接、尺寸变化时的投影分支，以及四个残差阶段。训练分为两组：`scratch` 从随机参数开始，`finetune` 在同一份手写网络中加载公开的 ImageNet 参数后微调。

## 在本机打开

双击 `启动演示.cmd`，点击“选择图片”，再点击“识别”，界面直接显示“这是一只××猫”。也可以点击“下一张例图”体验识别。界面可以切换两组模型，也可以查看训练曲线、混淆矩阵和错分图片。例图来自独立测试集，界面同时显示原始标注。

程序覆盖的品种为：阿比西尼亚猫、孟加拉猫、伯曼猫、孟买猫、英国短毛猫、埃及猫、缅因猫、波斯猫、布偶猫、俄罗斯蓝猫、暹罗猫和斯芬克斯猫。输入以一只猫为主体的清晰照片更合适。界面的百分比是模型的分类分数。

## 文件安排

| 文件或目录 | 用途 |
|---|---|
| `data.py` | 下载数据、筛选猫、记录坏图和重复图片、划分数据、提供共用预处理 |
| `model.py` | 手写 BasicBlock 和 ResNet18 |
| `train.py` | 训练、验证、保存最佳参数和独立测试 |
| `predict.py` | 加载模型，对一张图片预测 |
| `analyze.py` | 用 NumPy、Pandas 和 Matplotlib 生成统计表及结果图 |
| `demo.py` | 本地桌面演示 |
| `artifacts/data/` | 样本清单、清理记录和数据统计 |
| `artifacts/scratch/` | 从零训练的参数、配置、曲线数据和测试预测 |
| `artifacts/finetune/` | 微调实验的参数、配置、曲线数据和测试预测 |
| `artifacts/figures/` | 类别分布、曲线、混淆矩阵和错分样例 |
| `examples/` | 演示图片与原始标签 |

## 安装环境

使用 Python 3.12。Windows PowerShell 中，在项目目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CUDA 12.8 版本用于支持相应驱动的 NVIDIA 显卡。只使用 CPU 时，把第一条安装 PyTorch 的命令中的 `cu128` 改为 `cpu`。数据分析和预测共用这套环境。

## 重新整理和训练

下载与解压：

```powershell
.\.venv\Scripts\python.exe data.py --data-root data --out-dir artifacts/data --download --mirror fastai
```

`--mirror fastai` 使用 fast.ai 提供的 Oxford-IIIT Pet 镜像。也可以选择 `--mirror official`。首次下载包含猫和狗，整理时按官方的猫标注筛选；原始文件保留在数据目录中。

训练两组模型：

```powershell
.\.venv\Scripts\python.exe train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/scratch --run scratch --epochs 60 --patience 18
.\.venv\Scripts\python.exe train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/finetune --run finetune --epochs 30 --patience 12
```

两组使用同一份样本清单。官方 `trainval` 中每个品种留出约 20% 作验证；官方 `test` 单独保留。训练集更新参数，验证集选择保存哪一轮，最后用保存的模型评价测试集。`best.pt` 同时保存类别顺序和图片尺寸，预测时据此重建网络。

每轮结果写入 `history.csv`。`config.json` 记录这次训练的设置和运行环境；`metrics.json` 保存独立测试指标；`predictions.csv` 保存每张测试图片的真实类别和预测类别。

## 单图预测与结果分析

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint artifacts/finetune/best.pt --image "自己的猫.jpg"
.\.venv\Scripts\python.exe analyze.py --manifest artifacts/data/manifest.csv --data-root data --out-dir artifacts/figures --run-dir artifacts/scratch --run-dir artifacts/finetune
.\.venv\Scripts\python.exe demo.py
```

训练中的随机裁剪、翻转和颜色变化用于增加图片变化。验证、测试和单图预测使用同一套确定性预处理。混淆矩阵的行是真实品种，列是预测品种；对角线表示分对的图片。按行看，可以找到某个品种最容易被认成哪一类。

## 参考资料

- 数据：[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)，以及 [fast.ai 数据下载说明](https://docs.fast.ai/data.external.html)。
- 网络：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。
- 预训练参数：[PyTorch 官方 ResNet18 参数](https://download.pytorch.org/models/resnet18-f37072fd.pth)。
- 阅读过的训练流程参考：[花卉分类](https://github.com/dm9717/flower-classification)、[垃圾分类](https://github.com/AidinHamedi/Pytorch-Garbage-Classification)、[植物病害分类](https://github.com/abdullahselek/plant-disease-classification-pytorch)。本项目重新编写网络、数据整理、训练、分析和演示代码。

原始数据的使用要求随下载包中的说明保留。项目中的样例用于课程演示，图片来源见 `examples/examples.csv`。
