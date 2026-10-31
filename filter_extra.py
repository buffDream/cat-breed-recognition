"""用已有猫分类模型筛选额外训练图片，保留来源标签与模型高分一致的照片。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CatDataset
from predict import Predictor


def keep_mask(labels, predictions, probabilities, threshold):
    return (np.asarray(labels) == np.asarray(predictions)) & (np.asarray(probabilities) >= threshold)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.7)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error('threshold must be between 0 and 1')
    frame = pd.read_csv(args.manifest, keep_default_na=False)
    extra = frame[frame.source.eq('kaggle_ma7555_cat_breeds_v2')].reset_index(drop=True)
    if not extra.split.eq('train').all():
        raise ValueError('Only additional training images may be filtered.')
    predictor = Predictor(args.checkpoint)
    loader = DataLoader(CatDataset(extra, args.data_root, predictor.transform), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=predictor.device.type == 'cuda')
    predictions, confidence = [], []
    with torch.inference_mode():
        for n, (images, _) in enumerate(loader):
            images = images.to(predictor.device)
            with torch.autocast(device_type=predictor.device.type, enabled=predictor.device.type == 'cuda'):
                scores = predictor.model(images).float().softmax(1)
                if predictor.tta:
                    scores = (scores + predictor.model(torch.flip(images, [3])).float().softmax(1)) / 2
            value, index = scores.max(1)
            predictions.extend(index.cpu().tolist())
            confidence.extend(value.cpu().tolist())
            if (n + 1) % 30 == 0:
                print(f'Scored {len(predictions)}/{len(extra)} extra images', flush=True)
    extra['screen_pred_index'] = predictions
    extra['screen_confidence'] = confidence
    extra['screen_keep'] = keep_mask(extra.class_index, predictions, confidence, args.threshold)
    extra['screen_reason'] = np.where(extra.screen_keep, 'agreement_above_threshold',
                                     np.where(extra.class_index.ne(extra.screen_pred_index), 'label_disagreement', 'low_confidence'))
    selected = extra[extra.screen_keep][frame.columns]
    base = frame[~frame.source.eq('kaggle_ma7555_cat_breeds_v2')]
    final = pd.concat([base, selected], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out_dir/'manifest.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(args.out_dir/'screening.csv', index=False, encoding='utf-8-sig')
    summary = {'method':'source label equals existing classifier prediction and score >= threshold',
               'checkpoint':str(args.checkpoint), 'threshold':args.threshold,
               'candidate_extra_images':len(extra), 'retained_extra_images':len(selected),
               'final_split_counts':{s:int(final.split.eq(s).sum()) for s in ('train','val','test')},
               'reason_counts':extra.screen_reason.value_counts().to_dict(),
               'retained_by_breed':selected.groupby('breed').size().to_dict(),
               'original_rows_preserved':True, 'uses_validation_or_test_images_for_screening':False}
    (args.out_dir/'dataset_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
