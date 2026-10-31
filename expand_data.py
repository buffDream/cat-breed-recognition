"""清理额外的真实猫照片，只追加训练样本，保留原验证集和测试集。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from data import BREEDS, MANIFEST_COLUMNS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SOURCE = "kaggle_ma7555_cat_breeds_v2"
SOURCE_URL = "https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset"
LOG_COLUMNS = [
    "source_relative_path", "source_label", "class_index", "breed", "action", "reason",
    "width", "height", "gray_std", "sha256", "phash", "matched_image_ids",
    "matched_splits", "matched_distances", "matched_source_path", "detail",
]

# 正交 DCT 的基矩阵只计算一次。图像乘基矩阵后得到不同频率的系数。
_FREQUENCIES = np.arange(32, dtype=np.float64)[:, None]
_POSITIONS = np.arange(32, dtype=np.float64)[None, :] + 0.5
_DCT_BASIS = np.cos(np.pi * _FREQUENCIES * _POSITIONS / 32)
_DCT_BASIS *= np.sqrt(2 / 32)
_DCT_BASIS[0] /= np.sqrt(2)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def perceptual_hash(image: Image.Image) -> int:
    """32×32 灰度图 DCT，取左上 8×8，去掉 DC 后取中位数划分位值。

    DC 位固定为 0，其余 63 位组成一个用 64 位整数保存的 pHash。
    汉明距离就是两个 pHash 中不同位的个数。
    """
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    coefficients = _DCT_BASIS @ pixels @ _DCT_BASIS.T
    low_frequency = coefficients[:8, :8].reshape(-1)
    bits = low_frequency > np.median(low_frequency[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big")


def inspect_image(path: Path) -> tuple[int, int, float, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        width, height = image.size
        gray = image.convert("L")
        gray_std = float(np.asarray(gray, dtype=np.float32).std())
        phash = perceptual_hash(gray)
    return width, height, gray_std, phash


class _HashIndex:
    """用 NumPy 一次比较全部已保留 pHash，避免逐位的 Python 循环。"""

    def __init__(self, capacity: int):
        self.values = np.empty(max(1, capacity), dtype=np.uint64)
        self.image_ids: list[str] = []

    def add(self, phash: int, image_id: str) -> None:
        self.values[len(self.image_ids)] = np.uint64(phash)
        self.image_ids.append(image_id)

    def matches(self, phash: int, max_distance: int) -> list[tuple[str, int]]:
        if not self.image_ids:
            return []
        # 异或后为 1 的位，正是两个 pHash 不同的位。
        different_bits = self.values[:len(self.image_ids)] ^ np.uint64(phash)
        distances = np.bitwise_count(different_bits)
        indices = np.flatnonzero(distances <= max_distance)
        ordered = sorted(indices.tolist(), key=lambda index: (int(distances[index]), index))
        return [(self.image_ids[index], int(distances[index])) for index in ordered]


def _match_fields(matches, split_by_id):
    return {
        "matched_image_ids": ";".join(image_id for image_id, _ in matches),
        "matched_splits": ";".join(split_by_id.get(image_id, "train") for image_id, _ in matches),
        "matched_distances": ";".join(str(distance) for _, distance in matches),
    }


def expand_dataset(base_manifest, data_root, extra_root, out_dir, phash_distance=6):
    """保留原清单的行与划分，把合格的新照片复制后追加为 train。"""
    if not 0 <= phash_distance <= 63:
        raise ValueError("pHash 距离必须在 0 到 63 之间。")
    base_manifest, data_root = Path(base_manifest).resolve(), Path(data_root).resolve()
    extra_root, out_dir = Path(extra_root).resolve(), Path(out_dir).resolve()
    if not extra_root.is_dir():
        raise FileNotFoundError(f"没有找到额外图片目录：{extra_root}")
    target_root = data_root / "extra-cats"
    if extra_root == target_root or extra_root.is_relative_to(target_root):
        raise ValueError("extra-root 应与复制图片的 extra-cats 目录分开。")
    if (out_dir / "manifest.csv").resolve() == base_manifest:
        raise ValueError("输出清单不能覆盖原始清单，请另设 --out-dir。")

    original = pd.read_csv(base_manifest, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(original.columns)
    if missing:
        raise ValueError(f"原始清单缺少列：{sorted(missing)}")
    if original.image_id.duplicated().any():
        raise ValueError("原始清单的 image_id 必须唯一。")
    if not original.split.isin(["train", "val", "test"]).all():
        raise ValueError("原始清单只能包含 train、val、test。")
    breed_indices = {breed: index for index, (breed, _) in enumerate(BREEDS)}
    for row in original.itertuples(index=False):
        if row.breed not in breed_indices or int(row.class_index) != breed_indices[row.breed]:
            raise ValueError(f"原始清单标签与 BREEDS 顺序不一致：{row.image_id}")

    paths = [
        path for path in extra_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda path: (
        breed_indices.get(path.relative_to(extra_root).parts[0], len(BREEDS)),
        path.relative_to(extra_root).as_posix().casefold(),
        path.relative_to(extra_root).as_posix(),
    ))
    if not paths:
        raise ValueError(f"额外目录中没有支持的图片文件：{extra_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    base_sha = {}
    split_by_id = {}
    for row in original.itertuples(index=False):
        digest = str(row.sha256).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"原始清单 SHA-256 无效：{row.image_id}")
        base_sha.setdefault(digest, (row.image_id, row.split))
        split_by_id[row.image_id] = row.split

    heldout = original[original.split.isin(["val", "test"])]
    heldout_index = _HashIndex(len(heldout))
    print(f"计算原验证/测试图片的 pHash：{len(heldout)} 张。", flush=True)
    for number, row in enumerate(heldout.itertuples(index=False), 1):
        path = data_root / row.relative_path
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                phash = perceptual_hash(image)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise RuntimeError(f"无法计算原验证/测试图片的 pHash：{row.image_id}") from exc
        heldout_index.add(phash, row.image_id)
        if number % 500 == 0 or number == len(heldout):
            print(f"原验证/测试 pHash：{number}/{len(heldout)}", flush=True)

    # 只把已经接纳的新照片放进此索引。后续照片与它们近似时只保留第一张。
    extra_index = _HashIndex(len(paths))
    seen_extra_sha = {}
    new_rows = []
    removed_counts, added_counts = Counter(), Counter()
    candidate_counts = Counter()
    base = original.copy()
    if "source" not in base.columns:
        base["source"] = "oxford_iiit_pet_official"
    if "source_label" not in base.columns:
        base["source_label"] = base["breed"]

    log_path = out_dir / "extra_cleaning_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as log_stream:
        writer = csv.DictWriter(log_stream, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for number, path in enumerate(paths, 1):
            relative = path.relative_to(extra_root).as_posix()
            source_label = path.relative_to(extra_root).parts[0]
            candidate_counts[source_label] += 1
            record = {column: "" for column in LOG_COLUMNS}
            record.update(source_relative_path=relative, source_label=source_label, action="removed")
            reason = ""
            if source_label not in breed_indices:
                reason = "unsupported_label"
                record["detail"] = "源目录名不在目标 12 品种中。"
            else:
                class_index = breed_indices[source_label]
                record.update(class_index=class_index, breed=source_label)
                try:
                    digest = file_sha256(path)
                    record["sha256"] = digest
                except OSError as exc:
                    reason = "read_error"
                    record["detail"] = f"{type(exc).__name__}: {exc}"

                if not reason and digest in base_sha:
                    reason = "exact_duplicate_base"
                    matched_id, matched_split = base_sha[digest]
                    record.update(matched_image_ids=matched_id, matched_splits=matched_split)
                if not reason and digest in seen_extra_sha:
                    reason = "exact_duplicate_extra"
                    previous = seen_extra_sha[digest]
                    record.update(
                        matched_image_ids=previous["image_id"], matched_source_path=previous["path"],
                    )
                    if previous["image_id"]:
                        record["matched_splits"] = "train"
                if not reason:
                    seen_extra_sha[digest] = {"path": relative, "image_id": ""}
                    try:
                        width, height, gray_std, phash = inspect_image(path)
                        record.update(width=width, height=height, gray_std=round(gray_std, 6), phash=f"{phash:016x}")
                    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                        reason = "decode_error"
                        record["detail"] = f"{type(exc).__name__}: {exc}"
                    if not reason and min(width, height) < 128:
                        reason = "too_small"
                    if not reason and gray_std < 5:
                        reason = "low_variation"
                    if not reason:
                        matches = heldout_index.matches(phash, phash_distance)
                        if matches:
                            reason = "near_duplicate_heldout"
                            record.update(_match_fields(matches, split_by_id))
                    if not reason:
                        matches = extra_index.matches(phash, phash_distance)
                        if matches:
                            reason = "near_duplicate_extra"
                            record.update(_match_fields(matches, split_by_id))
                    if not reason:
                        image_id = f"extra_{digest}"
                        destination = target_root / source_label / f"{digest}{path.suffix.lower()}"
                        if not destination.resolve().is_relative_to(data_root):
                            raise ValueError(f"图片输出路径越出 data-root：{destination}")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            if file_sha256(destination) != digest:
                                raise FileExistsError(f"同名目标文件内容不同：{destination}")
                        else:
                            shutil.copy2(path, destination)
                        row = {column: "" for column in base.columns}
                        row.update(
                            image_id=image_id, relative_path=destination.relative_to(data_root).as_posix(),
                            class_index=class_index, breed=source_label, breed_zh=BREEDS[class_index][1],
                            split="train", width=width, height=height, sha256=digest,
                            source=SOURCE, source_label=source_label,
                        )
                        new_rows.append(row)
                        extra_index.add(phash, image_id)
                        seen_extra_sha[digest]["image_id"] = image_id
                        added_counts[source_label] += 1
                        record.update(action="added", matched_image_ids="", matched_splits="")
                        reason = "accepted"
            record["reason"] = reason
            if record["action"] == "removed":
                removed_counts[reason] += 1
            writer.writerow(record)
            if number % 500 == 0 or number == len(paths):
                log_stream.flush()
                print(
                    f"额外图片 {number}/{len(paths)}：新增 {len(new_rows)}，移除 {sum(removed_counts.values())}",
                    flush=True,
                )

    if new_rows:
        final = pd.concat([base, pd.DataFrame(new_rows, columns=base.columns)], ignore_index=True)
    else:
        final = base
    if not final.iloc[:len(original)][original.columns].reset_index(drop=True).equals(original.reset_index(drop=True)):
        raise RuntimeError("原清单行发生变化，未写出扩充清单。")
    if final.image_id.duplicated().any():
        raise RuntimeError("新增 image_id 与原清单冲突，未写出扩充清单。")
    final.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    base_counts = original.groupby(["breed", "split"]).size()
    final_counts = final.groupby(["breed", "split"]).size()
    summary = {
        "base_manifest": str(base_manifest), "base_manifest_sha256": file_sha256(base_manifest),
        "data_root": str(data_root), "extra_root": str(extra_root),
        "extra_source": {"name": SOURCE, "url": SOURCE_URL, "version": 2},
        "candidate_images": len(paths),
        "candidate_counts_by_source_label": dict(candidate_counts),
        "base_images": len(original),
        "base_split_counts": {split: int((original.split == split).sum()) for split in ("train", "val", "test")},
        "added_train_images": len(new_rows),
        "new_train_images": int((final.split == "train").sum()),
        "final_images": len(final),
        "final_split_counts": {split: int((final.split == split).sum()) for split in ("train", "val", "test")},
        "removed_images": sum(removed_counts.values()),
        "removed_counts": dict(sorted(removed_counts.items())),
        "breed_counts": {
            breed: {
                "added_train": int(added_counts[breed]),
                "base": {split: int(base_counts.get((breed, split), 0)) for split in ("train", "val", "test")},
                "final": {split: int(final_counts.get((breed, split), 0)) for split in ("train", "val", "test")},
            }
            for breed, _ in BREEDS
        },
        "cleaning": {
            "exif_transpose": True, "minimum_edge": 128,
            "low_variation_rule": "EXIF校正后整张灰度图的像素标准差 < 5",
            "raw_sha256": "排除与原全部样本、额外来源先前文件完全相同的图片",
            "phash_definition": "灰度LANCZOS缩放32x32；正交DCT；左上8x8低频；除DC外63系数中位数；大于中位数置1；DC位固定0；64位整数",
            "phash_distance": int(phash_distance),
            "phash_comparison": "与原val/test全部pHash、已接纳额外图片全部pHash比较；汉明距离小于等于阈值时移除",
            "near_duplicate_policy": "原val/test全部保留；额外来源按BREEDS顺序及相对路径排序保留第一张",
        },
        "original_rows_unchanged": True,
        "extra_cleaning_log": str(log_path),
    }
    source_index = extra_root / "source-index.csv"
    if source_index.is_file():
        summary["source_index"] = {"path": str(source_index), "sha256": file_sha256(source_index)}
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"扩充清单已写入 {out_dir / 'manifest.csv'}；新增训练照片 {len(new_rows)} 张，"
        f"训练集现有 {summary['new_train_images']} 张。",
        flush=True,
    )
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phash-distance", type=int, default=6)
    args = parser.parse_args()
    expand_dataset(args.base_manifest, args.data_root, args.extra_root, args.out_dir, args.phash_distance)


if __name__ == "__main__":
    main()
