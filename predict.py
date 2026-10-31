"""Predict the most likely breed for one image."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from data import make_transforms
from model import create_model

class Predictor:
    def __init__(self, checkpoint_path, device='auto'):
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA 不可用，请选择 cpu 或 auto。')
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'找不到模型文件：{checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.class_names = checkpoint['class_names']
            self.class_names_zh = checkpoint['class_names_zh']
            if len(self.class_names) != len(self.class_names_zh) or not self.class_names:
                raise ValueError('类别信息不完整')
            self.architecture = checkpoint.get('architecture', 'resnet18')
            self.tta = checkpoint.get('tta', False)
            self.model = create_model(num_classes=len(self.class_names), pretrained=False, architecture=self.architecture)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.transform = make_transforms(image_size=int(checkpoint['image_size']), training=False)
        except (KeyError, RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(f'模型文件无法读取或格式不正确：{checkpoint_path}（{exc}）') from exc
        self.model.to(self.device).eval()

    def predict(self, image_path):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f'找不到图片：{path}')
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                tensor = self.transform(image).unsqueeze(0).to(self.device)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f'无法读取图片：{path}。请检查图片是否损坏或格式是否受支持。') from exc
        with torch.inference_mode():
            scores = self.model(tensor).float().softmax(dim=1)
            if self.tta:
                scores = (scores + self.model(torch.flip(tensor, dims=[3])).float().softmax(dim=1)) / 2
            scores = scores[0]
            index = int(scores.argmax().item())
            probability = float(scores[index].item())
        return {'breed': self.class_names[index], 'breed_zh': self.class_names_zh[index],
                'probability': probability}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    try:
        result = Predictor(args.checkpoint, args.device).predict(args.image)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'错误：{exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
