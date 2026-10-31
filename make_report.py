"""根据实际训练记录生成项目说明。"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line

ROOT = Path(__file__).resolve().parent
LABELS = {"scratch": "ResNet18 从零训练", "finetune": "ResNet18 微调",
          "optimized": "ResNet50 优化", "expanded": "ResNet50 扩充数据"}
INK = colors.HexColor("#203c36")
GREEN = colors.HexColor("#287c67")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "项目说明.pdf")
    args = parser.parse_args()
    runs = [r for r in LABELS if (ROOT / "artifacts" / r / "metrics.json").exists()]
    if not runs:
        raise FileNotFoundError("请先完成实际训练。")
    selection = load(ROOT / "artifacts" / "final_selection.json")
    main_run = selection["run"]
    metrics = {r: load(ROOT / "artifacts" / r / "metrics.json") for r in runs}
    configs = {r: load(ROOT / "artifacts" / r / "config.json") for r in runs}
    data_dir = ROOT / "artifacts" / ("data-curated" if (ROOT / "artifacts" / "data-curated" / "manifest.csv").is_file() else "data")
    manifest = pd.read_csv(data_dir / "manifest.csv")
    base = load(ROOT / "artifacts" / "data" / "dataset_summary.json")
    names = dict(zip(base["class_names"], base["class_names_zh"]))
    history = pd.read_csv(ROOT / "artifacts" / main_run / "history.csv")
    predictions = pd.read_csv(ROOT / "artifacts" / main_run / "predictions.csv")
    mistakes = predictions[predictions.true_index != predictions.pred_index]
    confusions = mistakes.groupby(["true_breed", "pred_breed"]).size().sort_values(ascending=False).head(5)
    counts = manifest.groupby(["breed", "split"]).size().unstack(fill_value=0)
    font_root = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("YaHei", str(font_root / "msyh.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("YaHeiBold", str(font_root / "msyhbd.ttc"), subfontIndex=0))
    pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold")
    body = ParagraphStyle("body", fontName="YaHei", fontSize=10.2, leading=17, textColor=INK,
                          spaceAfter=8, wordWrap="CJK")
    heading = ParagraphStyle("heading", parent=body, fontName="YaHeiBold", fontSize=17, leading=25, spaceAfter=14)
    small = ParagraphStyle("small", parent=body, fontSize=8.3, leading=13, spaceAfter=5)
    title = ParagraphStyle("title", parent=heading, fontSize=26, leading=34)
    story, text = [], []

    def p(value, style=body):
        text.append(value)
        story.append(Paragraph(escape(value).replace("\n", "<br/>"), style))

    def table(rows, widths=None):
        text.extend(["\t".join(map(str, row)) for row in rows])
        cells = [[Paragraph(escape(str(cell)), small) for cell in row] for row in rows]
        result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eee9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .7, GREEN),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#d8e0dc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([result, Spacer(1, 9)])

    def picture(path, width, max_height):
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image(str(path))
        ratio = min(width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        image.hAlign = "CENTER"
        story.extend([image, Spacer(1, 7)])

    p("猫品种识别", title)
    p("基于手写 ResNet 的图像分类项目", heading)
    p("项目接收一张猫的照片，输出一个品种，例如“这是一只英国短毛猫”。本地界面可以选图识别，也可以查看训练曲线和错分图片。识别范围是下面 12 个品种。")
    p("数据整理", heading)
    split_counts = manifest.split.value_counts().to_dict()
    n_train, n_val, n_test = (int(split_counts.get(x, 0)) for x in ("train", "val", "test"))
    p(f"最初使用 Oxford-IIIT Pet 的 {base['num_images']:,} 张猫图片，其中 950 张训练、238 张验证、1,183 张测试。扩充实验使用 {n_train:,} 张训练图片，验证与测试仍分别为 {n_val:,} 张和 {n_test:,} 张。训练集负责更新参数，验证集负责选模型，测试集用于记录最终识别成绩。")
    p("整理时检查图片能否解码，记录图片尺寸、类别和来源，再用文件哈希去重。扩充数据还使用感知哈希排除与验证、测试图片相近的照片，清理过程保存在 CSV 中。", body)
    rows = [["品种", "训练", "验证", "测试"]]
    for breed, zh in names.items():
        row = counts.loc[breed]
        rows.append([zh, int(row.get("train", 0)), int(row.get("val", 0)), int(row.get("test", 0))])
    table(rows, [230, 75, 75, 75])
    p("新增数据来自 Kaggle 的 Cat Breeds Dataset 第2版，目录标签由领养广告发布者填写。18,738张候选照片经文件、尺寸和去重检查后保留17,933张；抽样发现占位图和标签不符，再用原ResNet50筛选：来源标签与模型预测一致，且分数至少0.70，最终接纳6,870张。加上950张原训练照片，共7,820张。该筛选规则只用于新增训练图片，未改动验证和测试标签。", small)
    story.append(PageBreak())

    p("网络怎样识别图片", heading)
    p("图片先转成 RGB 张量，再经过卷积层提取局部特征。后面的层逐步组合这些特征，最后输出 12 个分类分数，取分数最高的一个品种。NumPy 处理数组，Pandas 管理样本和实验表，Matplotlib 绘制统计图。")
    p("残差块保留一条直接传递输入的支路，再把卷积得到的结果与它相加。可以把这个过程理解成：保留原来的信息，再学习需要补充的变化。两条支路的尺寸和通道数要一致，发生变化时用 1×1 卷积匹配。")
    diagram = Drawing(450, 112)
    diagram.add(Rect(1, 20, 448, 82, fillColor=colors.HexColor("#f1f5f2"), strokeColor=colors.HexColor("#d5e1d9")))
    for x, label in [(18, "输入 x"), (117, "卷积处理 F(x)"), (302, "相加"), (374, "ReLU")]:
        diagram.add(Rect(x, 38, 68 if x != 117 else 132, 35, fillColor=colors.white, strokeColor=GREEN, rx=4))
        diagram.add(String(x + (34 if x != 117 else 66), 50, label, fontName="YaHei", fontSize=10, textAnchor="middle", fillColor=INK))
    for x1, x2 in [(86, 117), (249, 302), (370, 374)]:
        diagram.add(Line(x1, 56, x2, 56, strokeColor=GREEN))
    diagram.add(Line(50, 73, 50, 91, strokeColor=GREEN))
    diagram.add(Line(50, 91, 336, 91, strokeColor=GREEN))
    diagram.add(Line(336, 91, 336, 73, strokeColor=GREEN))
    diagram.add(String(185, 94, "跳跃连接；需要时做尺寸匹配", fontName="YaHei", fontSize=9, textAnchor="middle", fillColor=INK))
    story.extend([diagram, Spacer(1, 10)])
    p("实现中保留了手写 ResNet18 作为起点，优化组增加手写 ResNet50。ResNet50 的瓶颈块使用 1×1、3×3、1×1 三次卷积，四个阶段分别放 3、4、6、3 个块。网络由基础层组装，预训练参数从 PyTorch 官方地址加载。")
    table([["ResNet50 阶段", "输出形状（单张 224×224 图片）"],
           ["首层卷积与池化", "64 × 56 × 56"], ["残差阶段 1", "256 × 56 × 56"],
           ["残差阶段 2", "512 × 28 × 28"], ["残差阶段 3", "1024 × 14 × 14"],
           ["残差阶段 4", "2048 × 7 × 7"], ["全局平均池化与分类层", "2048 个特征 → 12 个分数"]], [205, 250])
    p("优化组把输入从 160×160 提高到 224×224，减少训练裁剪幅度，并给特征主干和分类层设置不同学习率。预测时对原图和水平翻转图各算一次，合并分数后仍只输出一个品种。")
    story.append(PageBreak())

    p("训练与实际结果", heading)
    rows = [["训练设置", "最佳轮次", "验证准确率", "测试准确率", "宏平均 F1"]]
    for run in runs:
        m = metrics[run]
        rows.append([LABELS[run], m["best_epoch"], f"{m['best_val_accuracy']:.2%}",
                     f"{m['test_accuracy']:.2%}", f"{m['macro_f1']:.4f}"])
    table(rows, [151, 54, 83, 83, 84])
    m = metrics[main_run]
    correct = int((predictions.true_index == predictions.pred_index).sum())
    p(f"当前演示使用“{LABELS[main_run]}”的验证最佳参数，在同一组 {n_test:,} 张测试图片中识别正确 {correct:,} 张，准确率为 {m['test_accuracy']:.2%}。各实验先按验证准确率选择轮次，最后再在同一测试集记录结果。最终版本也依据验证准确率选择，若相同则保留较早的模型。所有分数来自保存的逐图预测表。")
    cfg = configs[main_run]
    p(f"选中模型的训练集有 {cfg['split_counts']['train']:,} 张图片，最多设置 {cfg['epochs']} 轮，实际记录 {len(history)} 轮，批量大小为 {cfg['batch_size']}，优化器为 AdamW。运行设备是 {cfg.get('gpu', 'CPU')}。当前模型在第 {m['best_epoch']} 轮被保存；第 0 轮表示继续训练前的初始模型在验证集上更好。")
    if "expanded" in metrics:
        expanded_history = pd.read_csv(ROOT / "artifacts" / "expanded" / "history.csv")
        p(f"扩充组实际训练 {len(expanded_history)} 轮，最佳为第 {metrics['expanded']['best_epoch']} 轮。第0轮是起始权重，扩充后没有超过原模型的验证准确率，因此演示保留优化组。", small)
    picture(ROOT / "artifacts" / "figures" / "comparison.png", 455, 180)
    picture(ROOT / "artifacts" / "figures" / main_run / "training_curves.png", 455, 170)
    p("准确率统计分对的图片比例。宏平均 F1 先分别计算每个品种的分数，再取平均，可以同时查看各品种的识别表现。", small)
    story.append(PageBreak())

    p("哪些猫容易被认错", heading)
    p("混淆矩阵的行是真实品种，列是预测品种。对角线越深，表示这一类分对的图片越多；其余格子记录具体错分方向。")
    picture(ROOT / "artifacts" / "figures" / main_run / "confusion_matrix.png", 390, 360)
    rows = [["真实品种", "预测品种", "错分数量"]]
    for (true, pred), count in confusions.items():
        rows.append([names[true], names[pred], int(count)])
    table(rows, [180, 180, 95])
    p(f"测试中共出现 {len(mistakes)} 张错分图片。每张图片的预测类别和分数保存在 predictions.csv，选出的错分图片见 artifacts/figures/{main_run}/misclassified.png。可以把毛色、脸型、长短毛和拍摄角度放在一起看，分析模型把注意力放在了哪些外观差异上。")
    story.append(PageBreak())

    p("实现中的体会与使用方法", heading)
    original_history = pd.read_csv(ROOT / "artifacts" / "scratch" / "history.csv")
    last = original_history.iloc[-1]
    p(f"从零训练组的最后一轮训练准确率达到 {last.train_accuracy:.1%}，验证准确率为 {last.val_accuracy:.1%}。这个差距很直观：把训练照片分对，与识别新照片是两个阶段。项目因此同时记录训练和验证曲线，并始终保存验证表现最好的一轮。")
    p("手写残差块最容易出问题的地方是维度。支路相加前，通道数、宽度和高度都需要对齐。把各阶段形状写成表，比只看一长串层定义更容易理解网络。")
    p("扩充数据时，照片数量和标签质量要一起看。抽样中的部分图片只具备相似毛色，也有等待补图的占位图片。增加标签一致性筛选后，额外照片从17,933张减到6,870张。文件哈希和感知哈希负责去重，模型筛选负责减少标签冲突；逐图记录保留了每一步的去留原因。")
    p("本地使用", heading)
    p("双击“启动演示.cmd”，选择照片，再点击“识别”。程序显示一个结果，例如“这是一只波斯猫”。“实验结果”页可以查看曲线、混淆矩阵和错分图片。输入以一只猫为主体的清晰照片，识别范围为项目列出的 12 个品种。")
    p(f"命令行预测：\npython -X utf8 predict.py --checkpoint artifacts/{main_run}/best.pt --image 自己的猫.jpg", small)
    p("完整安装、下载、整理和训练命令见 README.md。config.json 记录训练设置，history.csv 记录每一轮，metrics.json 与 predictions.csv 记录独立测试结果。源码按开发阶段保留了本地 Git 提交。")
    p("资料来源", heading)
    for value in [
        "Oxford-IIIT Pet：robots.ox.ac.uk/~vgg/data/pets/；Parkhi 等，Cats and Dogs，CVPR 2012。",
        "扩充数据：kaggle.com/datasets/ma7555/cat-breeds-dataset，版本 2；来源记录见docs/extra-download-source.json。",
        "网络：He 等，Deep Residual Learning for Image Recognition，arXiv:1512.03385。",
        "预训练参数：PyTorch 官方 ResNet18 与 ResNet50 V2。训练流程参考阅读了考核提供的花卉、垃圾、植物病害项目。",
    ]:
        p(value, small)

    def page_frame(canvas, doc):
        canvas.setStrokeColor(GREEN)
        canvas.line(18 * mm, 283 * mm, 192 * mm, 283 * mm)
        canvas.setFont("YaHei", 8)
        canvas.setFillColor(colors.HexColor("#68766f"))
        canvas.drawString(18 * mm, 12 * mm, "猫品种识别 · 项目说明")
        canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))

    doc = SimpleDocTemplate(str(args.output), pagesize=(210 * mm, 297 * mm),
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm,
                            title="猫品种识别项目说明", author="猫品种识别项目")
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    args.output.with_suffix(".txt").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
