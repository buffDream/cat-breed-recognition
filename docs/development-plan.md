# 猫品种识别开发安排

目标是做一个能在本地运行的猫品种分类项目。使用 Oxford-IIIT Pet 的猫图片，网络用 PyTorch 基础层手写 ResNet18。训练、预测、分析和演示入口分别放在独立文件中。

## 实验设置

- 12 个猫品种，按官方 species 标注筛选。官方 trainval 中每类按固定种子 42 留出约 20% 作验证，官方 test 作独立测试。
- 图片读入时检查解码，按 SHA-256 清理完全重复图片，保留清理记录。
- 输入 160×160 RGB，训练使用随机裁剪、水平翻转和轻微颜色变化；验证和预测共用确定性预处理。使用 ImageNet 标准均值和标准差。
- 手写标准 ResNet18，包括 BasicBlock、四个残差阶段和投影分支；不调用 torchvision 的模型构建函数。
- 从零训练作基线；再在同一个手写网络加载公开 ImageNet 权重微调。保存两组配置、逐轮记录、验证最佳权重和测试预测。
- 测试集不参与选轮次或调参。所有成绩取实际运行结果。
- NumPy 处理图片数组与混淆矩阵，Pandas 管理样本表和结果表，Matplotlib 画数据分布、曲线、混淆矩阵和错分图片。
- 本地桌面演示支持选图、例图、单一品种识别结果和模型切换。录制实际运行过程。
- 文档使用 humanizer-zh 润色，直接写做法和结果；不编造个人经历，不使用用户指出的防御性套话。
- 只做本地项目与 Git 历史；上传和分发由用户后续处理。

## 文件与接口

1. `data.py`：`BREEDS` 为按字母序排列的 `(英文名, 中文名)` 列表；`prepare_dataset(data_root, output_root, seed=42, val_fraction=0.2)` 写 `output_root/manifest.csv`、`cleaning_log.csv`、`dataset_summary.json`，返回 DataFrame。清单列：image_id、relative_path、class_index、breed、breed_zh、split、width、height、sha256。`CatDataset(frame, data_root, transform)` 返回 image tensor 和整数标签。`make_transforms(image_size=160, training=False)` 提供共用预处理。
2. `analyze.py`：`plot_data_summary(manifest_path, data_root, out_dir)` 画分布与样例；`plot_run_results(run_dir, data_root, manifest_path, out_dir)` 画训练曲线、混淆矩阵、每类指标与错分例图；命令行支持整理全部结果。
3. `model.py`：`BasicBlock`、`ResNet18(num_classes=12)`，以及 `create_model(num_classes=12, pretrained=False)`。预训练参数来自 PyTorch 官方 ResNet18 地址，先加载 1000 类参数，再替换输出层。
4. `train.py`：参数 `--data-root --manifest --out-dir --run scratch/finetune --epochs --batch-size --image-size --seed --patience --workers --device`。每个 run 目录保存 history.csv、best.pt、metrics.json、predictions.csv、config.json。history 列：epoch、train_loss、train_accuracy、val_loss、val_accuracy、lr、elapsed_seconds。predictions 列：image_id、relative_path、true_index、pred_index、true_breed、pred_breed、confidence、correct。metrics 包含 test_accuracy、macro_precision、macro_recall、macro_f1、num_test、best_epoch、best_val_accuracy、training_seconds。
5. `predict.py`：`Predictor(checkpoint_path, device='auto')`，`predict(image_path)` 返回仅包含一个结果的 dict：breed、breed_zh、probability。checkpoint 包含 state_dict、class_names、class_names_zh、image_size、run、best_epoch、best_val_accuracy、pretrained。
6. `demo.py`：桌面界面，直接复用 Predictor。`examples/` 存精选的独立测试图片及标签表；同时包含正确与错分例子。
7. `README.md`、项目说明 PDF、运行视频、环境记录和本地 Git 提交记录。

## 推进顺序

- [ ] 建立隔离环境和本地仓库，保存当前安排。
- [ ] 整理猫数据，生成分析图和清理记录。
- [ ] 编写手写网络、训练与预测；开发中运行必要的维度、梯度和小批训练检查。
- [ ] 完成两组训练，保存独立测试结果与图表。
- [ ] 完成桌面演示、说明文档和实际运行视频，整理本地交付包。

具体实现由开发过程决定。对照实验的设置、数据清理数量和实际成绩在运行记录中保存，文档据此撰写。

## 识别效果优化

用户要求提高原微调模型的识别准确率。保留原始 train/val/test 清单，不更换测试图片。新增 `artifacts/optimized/` 保存优化实验，原两组结果保留。

- 在基础层代码中增加 Bottleneck 和标准 ResNet50，阶段块数为 [3, 4, 6, 3]，兼容 PyTorch 官方 ResNet50 V2 参数 `resnet50-11ad3fa6.pth`。不调用 torchvision 模型构建函数。
- 优化组输入 224×224，先训练分类层两轮，再用较小的主干学习率微调整网。AdamW，主干学习率 5e-5、分类层 5e-4、权重衰减 1e-4、标签平滑 0.05，最多 35 轮、验证早停 12 轮。
- 224 输入的训练裁剪范围为 0.75~1.0，保留水平翻转和轻微颜色变化；验证短边缩放到 232 后中心裁剪 224。160 输入保留原配方。验证/测试/预测统一使用相应确定性预处理。
- 优化组对原图与水平翻转图各预测一次，平均分数后只输出一个品种。这个设置同时用于验证、测试和桌面预测。
- `train.py` 增加 `--architecture resnet18/resnet50`（默认 resnet18）、`--warmup-epochs`、`--label-smoothing`、`--tta`；原 scratch/finetune 参数兼容。优化组仍用 `--run finetune`，以独立 out-dir 区分。
- checkpoint 保存 architecture、tta、warmup_epochs、label_smoothing 等新增配置。旧 checkpoint 缺少 architecture 时按 resnet18 加载，缺少 tta 时按 False。
- `Predictor.predict` 保持单结果 dict：breed、breed_zh、probability。
- 验证集用于选最佳轮次；优化组设置确定后才生成独立测试结果。不把数据增强数量当成独立照片数量。
