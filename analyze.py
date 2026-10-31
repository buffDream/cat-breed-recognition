"""用实际样本、训练记录和测试预测生成分析图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from data import BREEDS


def _set_style():
    # Windows 常见中文字体；其他系统优先选择可用的中文字体。
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    else:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 10


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _image_array(path, size=(240, 240)):
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        return np.asarray(image)


def plot_data_summary(manifest_path, data_root, out_dir):
    """画各划分的真实数量，以及每类一张训练图片。"""
    _set_style()
    out_dir, data_root = Path(out_dir), Path(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    names = [name for name, _ in BREEDS]
    labels = [name for _, name in BREEDS]
    counts = frame.groupby(["breed", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=names, columns=["train", "val", "test"], fill_value=0)
    counts.to_csv(out_dir / "class_distribution.csv", encoding="utf-8-sig", index_label="breed")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    for offset, split, label, color in [
        (-1, "train", "训练集", "#2e7797"),
        (0, "val", "验证集", "#e2a344"),
        (1, "test", "测试集", "#6c9c70"),
    ]:
        ax.bar(x + offset * 0.25, counts[split].to_numpy(), width=0.24, label=label, color=color)
    ax.set(xticks=x, xticklabels=labels, ylabel="图片数量", title="12 个猫品种的样本分布")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir / "class_distribution.png")

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    selected = []
    for index, ax in enumerate(axes.flat):
        candidates = frame[(frame.class_index == index) & (frame.split == "train")]
        if candidates.empty:
            candidates = frame[frame.class_index == index]
        ax.axis("off")
        if candidates.empty:
            ax.set_title(labels[index] + "（无样本）")
            continue
        row = candidates.sort_values("image_id").iloc[0]
        ax.imshow(_image_array(data_root / row.relative_path))
        ax.set_title(f"{labels[index]}\n{row.image_id}", fontsize=10)
        selected.append(row.to_dict())
    fig.suptitle("各品种图片示例（来自训练集）", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir / "sample_grid.png")
    pd.DataFrame(selected).to_csv(out_dir / "sample_grid.csv", index=False, encoding="utf-8-sig")
    return counts


def confusion_from_predictions(predictions, num_classes=12):
    """行是真实标签，列是模型预测标签。"""
    true = predictions["true_index"].to_numpy(dtype=np.int64)
    pred = predictions["pred_index"].to_numpy(dtype=np.int64)
    if np.any((true < 0) | (true >= num_classes) | (pred < 0) | (pred >= num_classes)):
        raise ValueError("预测表中存在超出类别范围的标签。")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def metrics_from_confusion(matrix):
    """分母为零时记为 0；每类 F1 是精确率和召回率的调和平均。"""
    matrix = np.asarray(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct, dtype=float), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    return pd.DataFrame({
        "class_index": np.arange(len(BREEDS)),
        "breed": [name for name, _ in BREEDS],
        "breed_zh": [name for _, name in BREEDS],
        "precision": precision, "recall": recall, "f1": f1,
        "support": support, "predicted_count": predicted,
    })


def _plot_training_curves(history, out_dir, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, label in [(axes[0], "loss", "交叉熵损失"), (axes[1], "accuracy", "准确率")]:
        ax.plot(history.epoch, history[f"train_{column}"], label="训练集", color="#2e7797")
        ax.plot(history.epoch, history[f"val_{column}"], label="验证集", color="#dc8b2f")
        ax.set(xlabel="训练轮次", ylabel=label)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    axes[1].set_ylim(0, 1.02)
    fig.suptitle(f"{run_name}：训练与验证曲线")
    fig.tight_layout()
    _save(fig, out_dir / "training_curves.png")


def _plot_confusion(matrix, out_dir, run_name):
    labels = [name for _, name in BREEDS]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    maximum = matrix.max(initial=0)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > maximum * 0.55 else "#263745")
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="预测品种（列）", ylabel="真实品种（行）", title=f"{run_name}：独立测试集混淆矩阵",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.75, label="图片数量")
    fig.tight_layout()
    _save(fig, out_dir / "confusion_matrix.png")


def _plot_per_class(metrics, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    for offset, metric, label, color in [
        (-1, "precision", "精确率", "#2e7797"),
        (0, "recall", "召回率", "#e2a344"),
        (1, "f1", "F1", "#6c9c70"),
    ]:
        ax.bar(x + offset * 0.25, metrics[metric], width=0.24, label=label, color=color)
    ax.set(xticks=x, xticklabels=metrics.breed_zh, ylim=(0, 1.05), ylabel="分数", title="各品种的测试集指标")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir / "per_class_metrics.png")


def _plot_mistakes(predictions, data_root, out_dir):
    mistakes = predictions[predictions.true_index != predictions.pred_index].copy()
    mistakes = mistakes.sort_values(["confidence", "image_id"], ascending=[False, True])
    mistakes.to_csv(out_dir / "misclassified.csv", index=False, encoding="utf-8-sig")
    if mistakes.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "这次测试没有错分图片。", ha="center", va="center", fontsize=16)
        ax.axis("off")
        _save(fig, out_dir / "misclassified.png")
        return
    # 每个真实品种先取一张，再补充置信度较高的错分图片。
    selected = mistakes.drop_duplicates("true_index").head(12)
    if len(selected) < min(12, len(mistakes)):
        selected = pd.concat([selected, mistakes.drop(index=selected.index)]).head(12)
    selected.to_csv(out_dir / "misclassified_selected.csv", index=False, encoding="utf-8-sig")
    nrows = (len(selected) + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(12, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        ax.imshow(_image_array(data_root / row.relative_path))
        true_zh = BREEDS[int(row.true_index)][1]
        pred_zh = BREEDS[int(row.pred_index)][1]
        ax.set_title(f"真实：{true_zh}\n预测：{pred_zh}（{row.confidence:.1%}）", fontsize=10)
    fig.suptitle("错分示例（优先展示不同品种）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir / "misclassified.png")


def plot_run_results(run_dir, data_root, manifest_path, out_dir):
    """读取一次实际运行的 CSV，输出曲线、混淆矩阵与错分示例。"""
    _set_style()
    run_dir, out_dir, data_root = Path(run_dir), Path(out_dir), Path(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(run_dir / "history.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    manifest = pd.read_csv(manifest_path)
    test = manifest[manifest.split == "test"]
    if predictions.image_id.duplicated().any():
        raise ValueError(f"{run_dir} 的预测表含有重复 image_id。")
    if set(predictions.image_id) != set(test.image_id):
        raise ValueError(f"{run_dir} 的预测样本与清单中的独立测试集不一致。")
    expected = test.set_index("image_id").loc[predictions.image_id]
    if not np.array_equal(expected.class_index.to_numpy(), predictions.true_index.to_numpy()):
        raise ValueError(f"{run_dir} 的真实标签与测试清单不一致。")
    if not np.array_equal(expected.relative_path.to_numpy(), predictions.relative_path.to_numpy()):
        raise ValueError(f"{run_dir} 的图片路径与测试清单不一致。")
    matrix = confusion_from_predictions(predictions)
    metrics = metrics_from_confusion(matrix)
    metrics.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix, index=[b[0] for b in BREEDS], columns=[b[0] for b in BREEDS]).to_csv(
        out_dir / "confusion_matrix.csv", index_label="true_breed", encoding="utf-8-sig"
    )
    _plot_training_curves(history, out_dir, run_dir.name)
    _plot_confusion(matrix, out_dir, run_dir.name)
    _plot_per_class(metrics, out_dir)
    _plot_mistakes(predictions, data_root, out_dir)
    return metrics


def compare_runs(run_dirs, out_dir):
    """把真实测试指标放在同一张图上，不用测试结果挑选权重。"""
    _set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in map(Path, run_dirs):
        record = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"run": run_dir.name, **record})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    keys = ["test_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["测试准确率", "宏平均精确率", "宏平均召回率", "宏平均 F1"]
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(frame))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (_, row) in enumerate(frame.iterrows()):
        values = row[keys].to_numpy(dtype=float)
        bars = ax.bar(x + (index - (len(frame) - 1) / 2) * width, values, width, label=row["run"])
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.12), ylabel="分数", title="两种训练方式的独立测试集结果")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir / "comparison.png")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="可重复传入，分别整理每次运行")
    args = parser.parse_args()
    plot_data_summary(args.manifest, args.data_root, args.out_dir)
    for run_dir in args.run_dir:
        plot_run_results(run_dir, args.data_root, args.manifest, args.out_dir / run_dir.name)
    if len(args.run_dir) >= 2:
        compare_runs(args.run_dir, args.out_dir)
    print(f"分析表和图片已写入：{args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
