"""猫品种识别的本地桌面入口。"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from predict import Predictor

ROOT = Path(__file__).resolve().parent
BG = "#f2f4f3"
INK = "#203c36"
MUTED = "#62736e"
ACCENT = "#287c67"


class CatApp(tk.Tk):
    def __init__(self, checkpoint: Path | None = None, auto_demo: bool = False):
        super().__init__()
        self.title("Cat Breed Recognition")
        self.geometry("1080x760")
        self.minsize(960, 700)
        self.configure(bg=BG)
        self.jobs: queue.Queue = queue.Queue()
        self.predictors: dict[str, Predictor] = {}
        self.image_path: Path | None = None
        self.busy = False
        self.example_index = 0
        self.examples = self.read_examples()
        self.models = self.find_models(checkpoint)
        self.preview_photo = None
        self.chart_photo = None
        self.build_ui()
        self.after(100, self.poll_results)
        if auto_demo:
            self.after(1800, self.show_next_example)
            self.after(8500, self.show_next_example)
            self.after(15500, self.show_next_example)
            self.after(22000, lambda: self.show_chart("训练曲线"))
            self.after(29000, lambda: self.show_chart("混淆矩阵"))
            self.after(36000, self.show_next_example)

    @staticmethod
    def find_models(checkpoint: Path | None) -> dict[str, Path]:
        if checkpoint:
            return {"指定模型": checkpoint.resolve()}
        models = {}
        for run, label in [("finetune", "手写 ResNet18 · 预训练后微调"), ("scratch", "手写 ResNet18 · 从零训练")]:
            path = ROOT / "artifacts" / run / "best.pt"
            if path.exists():
                models[label] = path
        return models

    @staticmethod
    def read_examples() -> list[dict]:
        path = ROOT / "examples" / "examples.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 11))
        style.configure("TButton", font=("Microsoft YaHei UI", 11), padding=(14, 9))
        style.configure("Accent.TButton", foreground="white", background=ACCENT)
        style.map("Accent.TButton", background=[("active", "#236b59")])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(22, 10), font=("Microsoft YaHei UI", 11))
        style.configure("Cat.Horizontal.TProgressbar", background=ACCENT, troughcolor="#e6ece8", borderwidth=0)

        header = ttk.Frame(self, padding=(28, 18, 28, 12))
        header.pack(fill="x")
        ttk.Label(header, text="看图识猫", font=("Microsoft YaHei UI", 25, "bold")).pack(anchor="w")
        ttk.Label(header, text="12 个猫品种 · 手写 ResNet18", foreground=MUTED).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(self, padding=(28, 0, 28, 16))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="模型").pack(side="left", padx=(0, 10))
        self.model_var = tk.StringVar(value=next(iter(self.models), "请先完成训练"))
        self.model_box = ttk.Combobox(toolbar, textvariable=self.model_var, values=list(self.models), state="readonly", width=38, font=("Microsoft YaHei UI", 10))
        self.model_box.pack(side="left")
        self.model_box.bind("<<ComboboxSelected>>", self.on_model_change)
        self.accuracy_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.accuracy_var, foreground=MUTED).pack(side="right")
        self.refresh_accuracy()

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=28, pady=(0, 18))
        self.predict_tab = ttk.Frame(self.tabs, padding=18)
        self.results_tab = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(self.predict_tab, text="图片识别")
        self.tabs.add(self.results_tab, text="实验结果")

        self.predict_tab.columnconfigure(0, weight=3)
        self.predict_tab.columnconfigure(1, weight=2)
        self.predict_tab.rowconfigure(0, weight=1)
        left = ttk.Frame(self.predict_tab)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        self.image_label = tk.Label(left, text="选择一张猫的图片", bg="white", fg=MUTED, font=("Microsoft YaHei UI", 16), relief="flat")
        self.image_label.pack(fill="both", expand=True)
        controls = ttk.Frame(left)
        controls.pack(fill="x", pady=(14, 4))
        self.open_button = ttk.Button(controls, text="选择图片", style="Accent.TButton", command=self.choose_image)
        self.open_button.pack(side="left")
        self.recognize_button = ttk.Button(controls, text="识别", style="Accent.TButton", command=self.start_prediction)
        self.recognize_button.pack(side="left", padx=(12, 0))
        self.example_button = ttk.Button(controls, text="下一张例图", command=self.show_next_example)
        self.example_button.pack(side="left", padx=12)
        self.source_var = tk.StringVar(value="支持 JPG、PNG、WebP；图片在本机处理。")
        ttk.Label(left, textvariable=self.source_var, wraplength=520, foreground=MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(8, 0))

        right = ttk.Frame(self.predict_tab)
        right.grid(row=0, column=1, sticky="nsew", pady=16)
        ttk.Label(right, text="识别结果", foreground=MUTED).pack(anchor="w")
        self.breed_var = tk.StringVar(value="等待图片")
        ttk.Label(right, textvariable=self.breed_var, font=("Microsoft YaHei UI", 24, "bold"), wraplength=340).pack(anchor="w", pady=(12, 2))
        self.english_var = tk.StringVar()
        ttk.Label(right, textvariable=self.english_var, foreground=MUTED).pack(anchor="w")
        ttk.Separator(right).pack(fill="x", pady=24)
        self.score_var = tk.StringVar()
        ttk.Label(right, textvariable=self.score_var, foreground=ACCENT, font=("Microsoft YaHei UI", 13)).pack(anchor="w")
        self.status_var = tk.StringVar(value="选择图片后，点击“识别”。")
        ttk.Label(right, textvariable=self.status_var, foreground=MUTED, wraplength=330, font=("Microsoft YaHei UI", 10)).pack(anchor="w", pady=(12, 0))

        chart_controls = ttk.Frame(self.results_tab)
        chart_controls.pack(fill="x", pady=(0, 10))
        for label in ["训练曲线", "混淆矩阵", "错分样例", "类别分布"]:
            ttk.Button(chart_controls, text=label, command=lambda value=label: self.show_chart(value)).pack(side="left", padx=(0, 8))
        self.chart_label = tk.Label(self.results_tab, text="选择一项查看实际实验结果", bg="white", fg=MUTED, font=("Microsoft YaHei UI", 14))
        self.chart_label.pack(fill="both", expand=True)
        self.chart_note = tk.StringVar()
        ttk.Label(self.results_tab, textvariable=self.chart_note, foreground=MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(8, 0))

    def refresh_accuracy(self):
        path = self.models.get(self.model_var.get())
        if not path:
            self.accuracy_var.set("请先训练模型")
            return
        metrics_path = path.parent / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.accuracy_var.set(f"独立测试集：{metrics['test_accuracy']:.1%}  ·  {metrics['num_test']} 张")
        else:
            self.accuracy_var.set("")

    def choose_image(self):
        path = filedialog.askopenfilename(title="选择猫的图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("所有文件", "*.*")])
        if path:
            self.show_image(Path(path), f"当前图片：{Path(path).name}")

    def show_next_example(self):
        if self.busy:
            self.after(400, self.show_next_example)
            return
        if not self.examples:
            messagebox.showinfo("例图", "examples 文件夹中还没有例图。可以直接选择自己的图片。")
            return
        example = self.examples[self.example_index % len(self.examples)]
        self.example_index += 1
        path = ROOT / "examples" / example["file"]
        note = f"独立测试集例图 · 标注：{example['breed_zh']} · {example['image_id']}"
        self.show_image(path, note)
        self.start_prediction()

    def show_image(self, path: Path, note: str):
        try:
            with Image.open(path) as source:
                preview = ImageOps.exif_transpose(source).convert("RGB")
                preview.thumbnail((545, 410), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=self.preview_photo, text="")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            messagebox.showerror("图片打不开", f"请选择有效的图片文件。\n{error}")
            return
        self.image_path = path
        self.source_var.set(note)
        self.tabs.select(self.predict_tab)
        self.breed_var.set("等待识别")
        self.english_var.set("")
        self.score_var.set("")
        self.status_var.set("点击“识别”查看品种。")

    def on_model_change(self, _event=None):
        self.refresh_accuracy()
        if self.image_path:
            self.start_prediction()

    def set_busy(self, busy: bool):
        self.busy = busy
        self.open_button.configure(state="disabled" if busy else "normal")
        self.recognize_button.configure(state="disabled" if busy else "normal")
        self.example_button.configure(state="disabled" if busy else "normal")
        self.model_box.configure(state="disabled" if busy else "readonly")

    def start_prediction(self):
        checkpoint = self.models.get(self.model_var.get())
        if not checkpoint or not self.image_path or self.busy:
            return
        self.set_busy(True)
        self.status_var.set("正在加载模型并识别图片……")
        self.breed_var.set("识别中")
        self.english_var.set("")
        self.score_var.set("")
        image_path = self.image_path

        def worker():
            try:
                key = str(checkpoint)
                if key not in self.predictors:
                    self.predictors[key] = Predictor(checkpoint, device="auto")
                result = self.predictors[key].predict(image_path)
                self.jobs.put((result, None))
            except Exception as error:
                self.jobs.put((None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_results(self):
        try:
            result, error = self.jobs.get_nowait()
        except queue.Empty:
            pass
        else:
            self.set_busy(False)
            if error:
                self.breed_var.set("识别未完成")
                self.status_var.set(error)
            else:
                self.breed_var.set(f"这是一只{result['breed_zh']}")
                self.english_var.set(result["breed"].replace("_", " "))
                self.score_var.set(f"识别分数  {result['probability']:.1%}")
                self.status_var.set("识别完成。")
        self.after(100, self.poll_results)

    def show_chart(self, label: str):
        checkpoint = self.models.get(self.model_var.get())
        run = checkpoint.parent.name if checkpoint else "finetune"
        root = ROOT / "artifacts" / "figures"
        paths = {"训练曲线": root / run / "training_curves.png", "混淆矩阵": root / run / "confusion_matrix.png", "错分样例": root / run / "misclassified.png", "类别分布": root / "class_distribution.png"}
        path = paths[label]
        self.tabs.select(self.results_tab)
        if not path.exists():
            self.chart_label.configure(image="", text="这张图尚未生成，请先运行 analyze.py。")
            self.chart_note.set("")
            return
        with Image.open(path) as image:
            chart = image.convert("RGB")
            chart.thumbnail((980, 500), Image.Resampling.LANCZOS)
        self.chart_photo = ImageTk.PhotoImage(chart)
        self.chart_label.configure(image=self.chart_photo, text="")
        self.chart_note.set(f"{label} · {self.model_var.get()}")


def main():
    parser = argparse.ArgumentParser(description="启动猫品种识别桌面界面")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--auto-demo", action="store_true", help="依次展示例图和结果图，便于录制运行视频")
    args = parser.parse_args()
    CatApp(args.checkpoint, args.auto_demo).mainloop()


if __name__ == "__main__":
    main()
