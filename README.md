# 看图识猫

选择一张猫的照片，程序显示一个品种，例如“这是一只布偶猫”。项目用PyTorch基础层手写ResNet18和ResNet50，使用NumPy、Pandas、Matplotlib整理数据和展示实际实验结果。

## 打开程序

在本机双击“启动演示.cmd”，点击“选择图片”，选择JPG、PNG或WebP，再点击“识别”。“下一张例图”会加载随项目保存的测试图片。例图包含正确与错分案例，并显示原始标注；演示例图经过挑选，整体成绩以全部1,183张测试图片为准。

程序默认使用ResNet50 优化，测试准确率88.84%。默认选择记录在artifacts/final_selection.json，规则是比较验证准确率；分数相同时保留较早的实验。界面一次只显示一个品种。

支持的12类：阿比西尼亚猫、孟加拉猫、伯曼猫、孟买猫、英国短毛猫、埃及猫、缅因猫、波斯猫、布偶猫、俄罗斯蓝猫、暹罗猫、斯芬克斯猫。照片以一只猫为主体，品种范围以这12类为准。界面百分比表示模型给出的分类分数。

## 数据与实际结果

Oxford-IIIT Pet原始猫图共2,371张，分为950训练、238验证、1,183测试。额外下载Kaggle第2版的18,738张候选照片：解码、尺寸、SHA256和pHash清理后保留17,933张；再用原ResNet50做标签一致性筛选，保留来源标签与预测一致且分数≥0.70的6,870张。扩充训练共使用7,820张真实照片。所有新增图片只进训练集，原验证与测试清单保持原样。

| 实验 | 最佳验证准确率 | 测试准确率 | 宏平均F1 |
|---|---:|---:|---:|
| ResNet18 从零训练 | 56.72% | 51.73% | 0.5165 |
| ResNet18 预训练微调 | 87.82% | 84.78% | 0.8494 |
| ResNet50 优化 | 94.12% | 88.84% | 0.8904 |
| ResNet50 扩充训练 | 94.12% | 88.84% | 0.8904 |

扩充组实际运行5轮，最佳权重为第0轮；本次扩充没有超过原优化组的验证准确率，最终保留ResNet50 优化。

每组先根据验证集保存最佳权重，训练结束后加载该权重记录测试结果。扩充组从optimized继续训练，并把初始权重作为第0轮候选；如果最佳轮次是0，表示新训练没有超过起始权重。

## 安装（换电脑时）

安装Python 3.12，在项目文件夹打开PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

上面安装CPU版即可识别。使用支持CUDA 12.8的NVIDIA显卡时，将安装地址最后的cpu改为cu128。本次实际训练为Python 3.12.14、torch 2.9.1+cu128，设备RTX 5080 Laptop GPU；完整版本在environment-lock.txt。

## 单图预测

```powershell
.\.venv\Scripts\python.exe -X utf8 predict.py --checkpoint artifacts/optimized/best.pt --image "自己的猫.jpg"
```

源码包不附大权重。使用源码包时，把完整成果包中artifacts/optimized/best.pt放回同一目录，或按下面的命令重新训练。

## 文件安排

| 文件 | 内容 |
|---|---|
| data.py | 原数据下载、猫图筛选、清单和共同预处理 |
| model.py | 手写BasicBlock、Bottleneck、ResNet18和ResNet50 |
| train.py | 训练、验证选模、模型保存、独立测试 |
| download_extra.py / expand_data.py | 候选照片区间下载和重复、尺寸检查 |
| filter_extra.py | 新增训练照片的标签一致性筛选 |
| predict.py / demo.py | 单图预测与桌面界面 |
| analyze.py / finalize_results.py | 实验图表、验证选模、演示例图与本说明 |
| make_report.py | 从真实记录生成项目说明PDF及文本副本 |
| artifacts/ | 数据清单、清理记录、配置、逐轮指标、预测、图表 |
| examples/ | 来自官方测试集的演示图片和原始标签 |
| docs/ | 数据来源、课程方法、开发记录与考核原文 |
| 上传到Gitee.cmd | 用户主动运行后，将已有Git提交上传到空仓库 |

## 重新整理和训练

以下命令在项目目录执行。完整成果包不附原始训练图片，重新训练前需下载数据。

```powershell
.\.venv\Scripts\python.exe -X utf8 data.py --data-root data --out-dir artifacts/data --download --mirror fastai
.\.venv\Scripts\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/scratch-new --run scratch --epochs 60 --patience 18 --batch-size 64 --image-size 160
.\.venv\Scripts\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/finetune-new --run finetune --epochs 30 --patience 12 --batch-size 64 --image-size 160
.\.venv\Scripts\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/optimized-new --run finetune --architecture resnet50 --epochs 35 --warmup-epochs 2 --batch-size 32 --image-size 224 --label-smoothing 0.05 --tta
.\.venv\Scripts\python.exe -X utf8 download_extra.py --out-dir extra-download --workers 8
.\.venv\Scripts\python.exe -X utf8 expand_data.py --base-manifest artifacts/data/manifest.csv --data-root data --extra-root extra-download/selected-images --out-dir artifacts/data-expanded
.\.venv\Scripts\python.exe -X utf8 filter_extra.py --manifest artifacts/data-expanded/manifest.csv --data-root data --checkpoint artifacts/optimized-new/best.pt --out-dir artifacts/data-curated --threshold 0.7
.\.venv\Scripts\python.exe -X utf8 train.py --data-root data --manifest artifacts/data-curated/manifest.csv --out-dir artifacts/expanded-new --run finetune --architecture resnet50 --initial-checkpoint artifacts/optimized-new/best.pt --epochs 8 --warmup-epochs 0 --batch-size 64 --image-size 224 --patience 5 --label-smoothing 0.05 --tta --balance --base-weight 8 --backbone-lr 5e-6 --head-lr 5e-5 --freeze-bn
.\.venv\Scripts\python.exe -X utf8 analyze.py --manifest artifacts/data-curated/manifest.csv --data-root data --out-dir artifacts/figures-reproduced --run-dir artifacts/scratch-new --run-dir artifacts/finetune-new --run-dir artifacts/optimized-new --run-dir artifacts/expanded-new
```

下载额外数据的索引为docs/extra-selected-entries.json和extra-selected-spans.json，完整参数见download_extra.py --help。保留旧实验，使用新out-dir记录复现实验。复现受随机过程、设备和软件版本影响，比较时查看完整配置。

重新生成项目说明时，先安装文档依赖，再运行脚本（Windows系统需要微软雅黑字体）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-extra.txt
.\.venv\Scripts\python.exe -X utf8 make_report.py
```

## 资料和开发记录

- Oxford-IIIT Pet：https://www.robots.ox.ac.uk/~vgg/data/pets/
- Kaggle Cat Breeds Dataset v2：https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset
- ResNet论文：https://arxiv.org/abs/1512.03385
- 官方权重：https://download.pytorch.org/models/resnet18-f37072fd.pth 和 https://download.pytorch.org/models/resnet50-11ad3fa6.pth
- 课程与参考项目阅读摘要见docs；网络、数据流程和演示根据猫识别选题编写。

训练配置在每组config.json，逐轮记录在history.csv，测试指标在metrics.json，逐图预测在predictions.csv。本地Git保留实际开发阶段。上传、分发由用户执行，具体步骤见独立的上传操作说明。
