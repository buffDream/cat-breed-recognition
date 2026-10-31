"""整理 Oxford-IIIT Pet 的猫图片，提供训练和预测共用的预处理。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode


BREEDS = [
    ("Abyssinian", "阿比西尼亚猫"),
    ("Bengal", "孟加拉猫"),
    ("Birman", "伯曼猫"),
    ("Bombay", "孟买猫"),
    ("British_Shorthair", "英国短毛猫"),
    ("Egyptian_Mau", "埃及猫"),
    ("Maine_Coon", "缅因猫"),
    ("Persian", "波斯猫"),
    ("Ragdoll", "布偶猫"),
    ("Russian_Blue", "俄罗斯蓝猫"),
    ("Siamese", "暹罗猫"),
    ("Sphynx", "斯芬克斯猫"),
]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MANIFEST_COLUMNS = [
    "image_id", "relative_path", "class_index", "breed", "breed_zh",
    "split", "width", "height", "sha256",
]
CLEANING_COLUMNS = [
    "image_id", "relative_path", "official_split", "action", "reason",
    "detail", "duplicate_of", "sha256",
]


def _dataset_folder(data_root: Path) -> Path:
    # torchvision 在给定根目录下创建 oxford-iiit-pet 子目录。
    for folder in (data_root / "oxford-iiit-pet", data_root):
        if (folder / "annotations" / "trainval.txt").is_file():
            return folder
    raise FileNotFoundError(
        f"没有找到 Oxford-IIIT Pet 标注：{data_root}。首次运行请加 --download。"
    )


def download_dataset(
    data_root: str | Path,
    mirror: str = "official",
    archive_path: str | Path | None = None,
) -> None:
    """默认下载官方数据；也可显式选择 fast.ai 的公开镜像。

    mirror='fastai' 时可传入已经下载完成的 archive_path，避免重复下载。
    SHA-256 是本地文件指纹，来源与解压结果记录在 download_source.json。
    """
    if mirror == "official":
        if archive_path is not None:
            raise ValueError("--archive 仅用于 --mirror fastai。")
        OxfordIIITPet(root=str(data_root), split="trainval", download=True)
        return
    if mirror != "fastai":
        raise ValueError(f"不支持的数据来源：{mirror}")

    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    source_url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"
    archive_path = (
        Path(archive_path).resolve() if archive_path is not None
        else data_root / "oxford-iiit-pet.tgz"
    )
    if not archive_path.is_file():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = archive_path.with_name(archive_path.name + ".part")
        # 使用 urllib 默认 TLS 验证；只把完整下载的临时文件改名。
        with urllib.request.urlopen(source_url, timeout=60) as response:
            with partial_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        partial_path.replace(archive_path)

    with archive_path.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    dataset_folder = data_root / "oxford-iiit-pet"
    provenance = {
        "mirror": "fastai", "source_url": source_url,
        "source_reference": "https://docs.fast.ai/data.external.html#urlspets",
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "sha256": sha256, "sha256_role": "local_file_fingerprint",
        "dataset_folder": str(dataset_folder),
        "extraction_completed": False,
    }
    provenance_path = data_root / "download_source.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        first = next((member for member in members if PurePosixPath(member.name).parts), None)
        if first is None:
            raise ValueError("镜像压缩包为空。")
        first_directory = PurePosixPath(first.name).parts[0]
        if first_directory == "oxford-iiit-pet":
            extraction_root = data_root
        elif first_directory in {"images", "annotations"}:
            extraction_root = dataset_folder
        else:
            raise ValueError(f"无法识别镜像压缩包目录：{first.name}")
        # data 过滤器拒绝越出目标目录的路径和危险链接。
        archive.extractall(path=extraction_root, members=members, filter="data")
    expected_paths = [
        dataset_folder / "images", dataset_folder / "annotations" / "trainval.txt",
        dataset_folder / "annotations" / "test.txt",
    ]
    if not all(path.exists() for path in expected_paths):
        raise FileNotFoundError("解压结果缺少 images 或官方 trainval/test 标注。")
    provenance["extraction_completed"] = True
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_dataset(data_root, output_root, seed=42, val_fraction=0.2):
    """筛选猫、检查解码、去除完全重复图片，再划分训练和验证集。

    官方 test 始终作测试集。同一文件内容出现多次时，先保留 test，
    再按 image_id 排序保留一个，避免相同图片出现在不同划分中。
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须在 0 与 1 之间。")
    data_root, output_root = Path(data_root).resolve(), Path(output_root)
    folder = _dataset_folder(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    breed_index = {name: i for i, (name, _) in enumerate(BREEDS)}
    candidates, cleaning = [], []
    official_counts = {"trainval": 0, "test": 0}
    for official_split in ("trainval", "test"):
        annotation = folder / "annotations" / f"{official_split}.txt"
        for line_number, line in enumerate(annotation.read_text().splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"标注格式错误：{annotation}:{line_number}")
            image_id, _, species, _ = fields
            if int(species) != 1:
                continue
            official_counts[official_split] += 1
            breed = image_id.rsplit("_", 1)[0]
            if breed not in breed_index:
                raise ValueError(f"发现未登记的猫品种：{breed}")
            path = folder / "images" / f"{image_id}.jpg"
            relative_path = path.relative_to(data_root).as_posix()
            record = {
                "image_id": image_id, "relative_path": relative_path,
                "class_index": breed_index[breed], "breed": breed,
                "breed_zh": BREEDS[breed_index[breed]][1],
                "official_split": official_split,
            }
            digest = ""
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    image.convert("RGB").load()
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("图片宽度或高度为零")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                cleaning.append({
                    **record, "action": "removed", "reason": "decode_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "duplicate_of": "", "sha256": digest,
                })
                continue
            candidates.append({**record, "width": width, "height": height, "sha256": digest})

    # 先确定重复图片的去留，再分训练/验证，避免数据泄漏。
    candidates.sort(key=lambda row: (row["official_split"] != "test", row["image_id"]))
    seen, kept = {}, []
    for record in candidates:
        previous = seen.get(record["sha256"])
        if previous is not None:
            cleaning.append({
                **record, "action": "removed", "reason": "exact_duplicate",
                "detail": (
                    f"保留 {previous['official_split']} 的 {previous['image_id']}；"
                    f"保留标签 {previous['breed']}"
                ),
                "duplicate_of": previous["image_id"],
            })
            continue
        seen[record["sha256"]] = record
        kept.append(record)

    frame = pd.DataFrame(kept)
    if frame.empty:
        raise ValueError("清理后没有可用猫图片，请检查数据目录和清理记录。")
    frame["split"] = np.where(frame["official_split"] == "test", "test", "train")
    rng = np.random.default_rng(seed)
    for class_index, (breed, _) in enumerate(BREEDS):
        train_indices = frame.index[
            (frame["class_index"] == class_index) & (frame["official_split"] == "trainval")
        ].to_numpy()
        if len(train_indices) < 2:
            raise ValueError(f"{breed} 的有效 trainval 图片不足 2 张。")
        n_val = max(1, min(len(train_indices) - 1, round(len(train_indices) * val_fraction)))
        frame.loc[rng.permutation(train_indices)[:n_val], "split"] = "val"
    frame = frame.sort_values(["class_index", "split", "image_id"]).reset_index(drop=True)
    frame = frame[MANIFEST_COLUMNS]
    frame.to_csv(output_root / "manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cleaning, columns=CLEANING_COLUMNS).sort_values(
        ["official_split", "image_id"]
    ).to_csv(output_root / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    counts = frame.groupby(["breed", "split"]).size().unstack(fill_value=0)
    summary = {
        "dataset": "Oxford-IIIT Pet (official cat species=1)",
        "source": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "seed": int(seed), "val_fraction": float(val_fraction),
        "official_cat_counts": official_counts,
        "num_images": len(frame), "num_classes": len(BREEDS),
        "split_counts": {key: int(value) for key, value in frame["split"].value_counts().items()},
        "removed_counts": {
            reason: sum(row["reason"] == reason for row in cleaning)
            for reason in ("decode_error", "exact_duplicate")
        },
        "duplicate_policy": "test 优先；同一官方划分按 image_id 字母序保留第一张",
        "hash_algorithm": "SHA-256 (原始图片文件字节)",
        "class_names": [name for name, _ in BREEDS],
        "class_names_zh": [name for _, name in BREEDS],
        "class_counts": {
            breed: {split: int(counts.loc[breed].get(split, 0)) for split in ("train", "val", "test")}
            for breed, _ in BREEDS
        },
        "width": {"min": int(frame.width.min()), "max": int(frame.width.max())},
        "height": {"min": int(frame.height.min()), "max": int(frame.height.max())},
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def make_transforms(image_size=160, training=False):
    """训练增加轻微变化；验证、测试和预测使用相同的确定性变换。"""
    if training:
        operations = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.025),
        ]
    else:
        operations = [
            transforms.Resize(round(image_size * 1.1), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
    return transforms.Compose([
        *operations, transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD),
    ])


class CatDataset(Dataset):
    def __init__(self, frame, data_root, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["relative_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class_index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="下载并解压数据；已有镜像压缩包会直接复用")
    parser.add_argument("--mirror", choices=["official", "fastai"], default="official", help="默认使用官方来源")
    parser.add_argument("--archive", type=Path, help="fastai 镜像已下载完整的 tgz 路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.archive is not None and not (args.download and args.mirror == "fastai"):
        parser.error("--archive 需要同时指定 --download --mirror fastai。")
    if args.download:
        download_dataset(args.data_root, mirror=args.mirror, archive_path=args.archive)
    frame = prepare_dataset(args.data_root, args.out_dir, seed=args.seed)
    print(frame.groupby("split").size().to_string())
    print(f"样本清单与清理记录已写入：{args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
