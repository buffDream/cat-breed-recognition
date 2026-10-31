"""按公开 ZIP 的目录索引下载所需的 12 类图片。"""

import argparse
import bisect
import concurrent.futures
import csv
import json
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://www.kaggle.com/api/v1/datasets/download/ma7555/cat-breeds-dataset?datasetVersionNumber=2"
TOTAL = 2069609007
BREEDS = {
    "abyssinian": "Abyssinian", "bengal": "Bengal", "birman": "Birman",
    "bombay": "Bombay", "britishshorthair": "British_Shorthair",
    "egyptianmau": "Egyptian_Mau", "mainecoon": "Maine_Coon",
    "persian": "Persian", "ragdoll": "Ragdoll", "russianblue": "Russian_Blue",
    "siamese": "Siamese", "sphynxhairlesscat": "Sphynx", "sphynx": "Sphynx",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    entries = json.loads((ROOT / "docs" / "extra-selected-entries.json").read_text(encoding="utf-8"))
    spans = json.loads((ROOT / "docs" / "extra-selected-spans.json").read_text(encoding="utf-8"))
    parts = args.out_dir / "download-parts"
    images = args.out_dir / "selected-images"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = []
    for span in spans:
        for start in range(span["start"], span["end"] + 1, 1024**2):
            chunks.append((start, min(start + 1024**2 - 1, span["end"])))

    def fetch(item):
        start, end = item
        part = parts / f"{start:012d}-{end:012d}.bin"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return start, end, part
        headers = part.with_suffix(".headers")
        for attempt in range(5):
            command = [
                "curl.exe", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "15", "--max-time", "45",
                "--range", f"{start}-{end}", "--dump-header", str(headers),
                "--output", str(part), "--write-out", "%{http_code}", URL,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            header_text = headers.read_text(encoding="latin-1") if headers.exists() else ""
            expected_header = f"content-range: bytes {start}-{end}/{TOTAL}"
            if (result.returncode == 0 and result.stdout.strip() == "206"
                    and part.exists() and part.stat().st_size == expected
                    and expected_header in header_text.lower()):
                return start, end, part
            if attempt == 4:
                raise RuntimeError(f"下载区间 {start}-{end} 失败：HTTP {result.stdout}；{result.stderr}")
            time.sleep(min(2 ** attempt, 8))

    print(f"下载 12 个品种的 {len(entries)} 张候选图片，所需压缩区间约 279 MiB。", flush=True)
    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            downloaded.append(future.result())
            if len(downloaded) % 25 == 0 or len(downloaded) == len(chunks):
                print(f"下载区间 {len(downloaded)}/{len(chunks)}", flush=True)
    downloaded.sort()
    starts = [item[0] for item in downloaded]

    def read_at(position, length):
        output = bytearray()
        while length:
            index = bisect.bisect_right(starts, position) - 1
            if index < 0:
                raise ValueError(f"缺少 ZIP 区间 {position}")
            start, end, part = downloaded[index]
            if position > end:
                raise ValueError(f"缺少 ZIP 区间 {position}")
            take = min(length, end - position + 1)
            with part.open("rb") as handle:
                handle.seek(position - start)
                value = handle.read(take)
            if len(value) != take:
                raise ValueError("下载区间不完整")
            output.extend(value)
            position += take
            length -= take
        return bytes(output)

    records = []
    for index, entry in enumerate(entries, 1):
        header = read_at(entry["header_offset"], 30)
        values = struct.unpack("<IHHHHHIIIHH", header)
        if values[0] != 0x04034B50:
            raise ValueError(f"ZIP 文件头错误：{entry['name']}")
        data_offset = entry["header_offset"] + 30 + values[-2] + values[-1]
        compressed = read_at(data_offset, entry["compress_size"])
        raw = zlib.decompress(compressed, -15)
        if len(raw) != entry["file_size"] or zlib.crc32(raw) & 0xffffffff != entry["CRC"]:
            raise ValueError(f"图片 CRC 校验失败：{entry['name']}")
        source_breed = entry["name"].split("/")[1]
        breed = BREEDS[re.sub(r"[^a-z]", "", source_breed.lower())]
        basename = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(entry["name"]).name)
        relative = Path(breed) / f"{entry['header_offset']}_{basename}"
        target = images / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        records.append({"relative_path": relative.as_posix(), "breed": breed,
                        "source_breed": source_breed, "source_name": entry["name"],
                        "crc32": entry["CRC"], "file_size": len(raw)})
        if index % 2000 == 0 or index == len(entries):
            print(f"解压并校验图片 {index}/{len(entries)}", flush=True)
    with (images / "source-index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    provenance = {"source": "Kaggle ma7555/cat-breeds-dataset", "version": 2,
                  "source_url": "https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset",
                  "candidate_images": len(records),
                  "downloaded_range_bytes": sum(end - start + 1 for start, end, _ in downloaded),
                  "original_zip_bytes": TOTAL, "validation": "HTTP Content-Range + per-image ZIP CRC32"}
    (args.out_dir / "download_source.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选图片已保存：{images.resolve()}", flush=True)


if __name__ == "__main__":
    main()
