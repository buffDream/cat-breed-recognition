"""按验证结果选演示模型，整理实际指标、例图与使用说明。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from data import BREEDS

ROOT = Path(__file__).resolve().parent
LABELS = {'scratch':'ResNet18 从零训练','finetune':'ResNet18 预训练微调',
          'optimized':'ResNet50 优化','expanded':'ResNet50 扩充训练'}


def main():
    runs = [r for r in LABELS if (ROOT/'artifacts'/r/'metrics.json').is_file()]
    metrics = {r:json.loads((ROOT/'artifacts'/r/'metrics.json').read_text(encoding='utf-8')) for r in runs}
    selected = max(runs, key=lambda r:metrics[r]['best_val_accuracy'])
    checkpoint = ROOT/'artifacts'/selected/'best.pt'
    with checkpoint.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    selection = {'run':selected, 'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                 'selection_rule':'highest validation accuracy; ties keep earlier experiment',
                 'best_val_accuracy':metrics[selected]['best_val_accuracy'],
                 'test_accuracy':metrics[selected]['test_accuracy'],
                 'sha256':digest}
    (ROOT/'artifacts/final_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for run in runs:
        folder=ROOT/'artifacts'/run
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        history=pd.read_csv(folder/'history.csv')
        rows.append({'run':run,'method':LABELS[run],'train_images':config['split_counts']['train'],
                     'actual_epochs':len(history),**metrics[run]})
    pd.DataFrame(rows).to_csv(ROOT/'artifacts/实验结果汇总.csv',index=False,encoding='utf-8-sig')
    original=pd.read_csv(ROOT/'artifacts/data/manifest.csv')
    curated=pd.read_csv(ROOT/'artifacts/data-curated/manifest.csv')
    for split in ('val','test'):
        expected=original.loc[original.split.eq(split)].reset_index(drop=True)
        actual=curated.loc[curated.split.eq(split),original.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,actual)
    cfg=json.loads((ROOT/'artifacts'/selected/'config.json').read_text(encoding='utf-8'))
    data_root=Path(cfg['data_root'])
    if not data_root.is_absolute():
        data_root=ROOT/data_root
    preds=pd.read_csv(ROOT/'artifacts'/selected/'predictions.csv')
    correct=preds[preds.true_index.eq(preds.pred_index)]
    examples=[]
    for i,(_,zh) in enumerate(BREEDS):
        group=correct[correct.true_index.eq(i)].sort_values('confidence',ascending=False)
        row=group.iloc[0] if len(group) else preds[preds.true_index.eq(i)].iloc[0]
        examples.append(row.to_dict())
    errors=preds[preds.true_index.ne(preds.pred_index)].sort_values('image_id').head(2).to_dict('records')
    ordered=examples[:3]+errors[:1]+examples[3:]+errors[1:]
    target=ROOT/'examples';target.mkdir(exist_ok=True)
    records=[]
    for row in ordered:
        source=data_root/row['relative_path']
        filename=row['image_id']+source.suffix
        shutil.copy2(source,target/filename)
        records.append({'file':filename,'image_id':row['image_id'],'breed':row['true_breed'],
                        'breed_zh':BREEDS[int(row['true_index'])][1], 'split':'official test',
                        'predicted_breed':row['pred_breed'],'correct':row['true_index']==row['pred_index'],
                        'source_url':'https://www.robots.ox.ac.uk/~vgg/data/pets/'})
    pd.DataFrame(records).to_csv(target/'examples.csv',index=False,encoding='utf-8-sig')
    table='\n'.join(f"| {LABELS[r]} | {metrics[r]['best_val_accuracy']:.2%} | {metrics[r]['test_accuracy']:.2%} | {metrics[r]['macro_f1']:.4f} |" for r in runs)
    outcome=''
    if 'expanded' in runs:
        length=len(pd.read_csv(ROOT/'artifacts/expanded/history.csv'))
        outcome=f"扩充组实际运行{length}轮，最佳权重为第{metrics['expanded']['best_epoch']}轮；本次扩充没有超过原优化组的验证准确率，最终保留{LABELS[selected]}。"
    content=f'''# 看图识猫

选择一张猫的照片，程序显示一个品种，例如“这是一只布偶猫”。项目用PyTorch基础层手写ResNet18和ResNet50，使用NumPy、Pandas、Matplotlib整理数据和展示实际实验结果。

## 打开程序

在本机双击“启动演示.cmd”，点击“选择图片”，选择JPG、PNG或WebP，再点击“识别”。“下一张例图”会加载随项目保存的测试图片。例图包含正确与错分案例，并显示原始标注；演示例图经过挑选，整体成绩以全部1,183张测试图片为准。

程序默认使用{LABELS[selected]}，测试准确率{metrics[selected]['test_accuracy']:.2%}。默认选择记录在artifacts/final_selection.json，规则是比较验证准确率；分数相同时保留较早的实验。界面一次只显示一个品种。

支持的12类：{'、'.join(zh for _,zh in BREEDS)}。照片以一只猫为主体，品种范围以这12类为准。界面百分比表示模型给出的分类分数。

## 数据与实际结果

Oxford-IIIT Pet原始猫图共2,371张，分为950训练、238验证、1,183测试。额外下载Kaggle第2版的18,738张候选照片：解码、尺寸、SHA256和pHash清理后保留17,933张；再用原ResNet50做标签一致性筛选，保留来源标签与预测一致且分数≥0.70的6,870张。扩充训练共使用7,820张真实照片。所有新增图片只进训练集，原验证与测试清单保持原样。

| 实验 | 最佳验证准确率 | 测试准确率 | 宏平均F1 |
|---|---:|---:|---:|
{table}

{outcome}

每组先根据验证集保存最佳权重，训练结束后加载该权重记录测试结果。扩充组从optimized继续训练，并把初始权重作为第0轮候选；如果最佳轮次是0，表示新训练没有超过起始权重。

## 安装（换电脑时）

安装Python 3.12，在项目文件夹打开PowerShell，执行：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

上面安装CPU版即可识别。使用支持CUDA 12.8的NVIDIA显卡时，将安装地址最后的cpu改为cu128。本次实际训练为Python 3.12.14、torch 2.9.1+cu128，设备RTX 5080 Laptop GPU；完整版本在environment-lock.txt。

## 单图预测

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 predict.py --checkpoint artifacts/{selected}/best.pt --image "自己的猫.jpg"
```

源码包不附大权重。使用源码包时，把完整成果包中artifacts/{selected}/best.pt放回同一目录，或按下面的命令重新训练。

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
| docs/ | 数据来源、下载索引与实验整理记录 |
| 上传到Gitee.cmd | 用户主动运行后，将已有Git提交上传到空仓库 |

## 重新整理和训练

以下命令在项目目录执行。完整成果包不附原始训练图片，重新训练前需下载数据。

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 data.py --data-root data --out-dir artifacts/data --download --mirror fastai
.\\.venv\\Scripts\\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/scratch-new --run scratch --epochs 60 --patience 18 --batch-size 64 --image-size 160
.\\.venv\\Scripts\\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/finetune-new --run finetune --epochs 30 --patience 12 --batch-size 64 --image-size 160
.\\.venv\\Scripts\\python.exe -X utf8 train.py --data-root data --manifest artifacts/data/manifest.csv --out-dir artifacts/optimized-new --run finetune --architecture resnet50 --epochs 35 --warmup-epochs 2 --batch-size 32 --image-size 224 --label-smoothing 0.05 --tta
.\\.venv\\Scripts\\python.exe -X utf8 download_extra.py --out-dir extra-download --workers 8
.\\.venv\\Scripts\\python.exe -X utf8 expand_data.py --base-manifest artifacts/data/manifest.csv --data-root data --extra-root extra-download/selected-images --out-dir artifacts/data-expanded
.\\.venv\\Scripts\\python.exe -X utf8 filter_extra.py --manifest artifacts/data-expanded/manifest.csv --data-root data --checkpoint artifacts/optimized-new/best.pt --out-dir artifacts/data-curated --threshold 0.7
.\\.venv\\Scripts\\python.exe -X utf8 train.py --data-root data --manifest artifacts/data-curated/manifest.csv --out-dir artifacts/expanded-new --run finetune --architecture resnet50 --initial-checkpoint artifacts/optimized-new/best.pt --epochs 8 --warmup-epochs 0 --batch-size 64 --image-size 224 --patience 5 --label-smoothing 0.05 --tta --balance --base-weight 8 --backbone-lr 5e-6 --head-lr 5e-5 --freeze-bn
.\\.venv\\Scripts\\python.exe -X utf8 analyze.py --manifest artifacts/data-curated/manifest.csv --data-root data --out-dir artifacts/figures-reproduced --run-dir artifacts/scratch-new --run-dir artifacts/finetune-new --run-dir artifacts/optimized-new --run-dir artifacts/expanded-new
```

下载额外数据的索引为docs/extra-selected-entries.json和extra-selected-spans.json，完整参数见download_extra.py --help。保留旧实验，使用新out-dir记录复现实验。复现受随机过程、设备和软件版本影响，比较时查看完整配置。

重新生成项目说明时，先安装文档依赖，再运行脚本（Windows系统需要微软雅黑字体）：

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-extra.txt
.\\.venv\\Scripts\\python.exe -X utf8 make_report.py --output ..\\项目说明.pdf
```

## 资料和开发记录

- Oxford-IIIT Pet：https://www.robots.ox.ac.uk/~vgg/data/pets/
- Kaggle Cat Breeds Dataset v2：https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset
- ResNet论文：https://arxiv.org/abs/1512.03385
- 官方权重：https://download.pytorch.org/models/resnet18-f37072fd.pth 和 https://download.pytorch.org/models/resnet50-11ad3fa6.pth
- 训练流程参考阅读：[花卉分类](https://github.com/dm9717/flower-classification)、[垃圾分类](https://github.com/AidinHamedi/Pytorch-Garbage-Classification)、[植物病害分类](https://github.com/abdullahselek/plant-disease-classification-pytorch)。网络、数据整理和演示根据猫识别选题编写。

考核原文、课程阅读摘要、上传步骤和交接记录保存在完整交付材料外层的“开发过程记录”，与代码仓库分开保存。

训练配置在每组config.json，逐轮记录在history.csv，测试指标在metrics.json，逐图预测在predictions.csv。本地Git保留实际开发阶段。上传、分发由用户执行，具体步骤见独立的上传操作说明。
'''
    (ROOT/'README.md').write_text(content,encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
