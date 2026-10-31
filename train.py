"""Train and evaluate the hand-written ResNet-18 on the prepared manifest."""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from data import BREEDS, CatDataset, make_transforms
from model import create_model

HISTORY_COLUMNS = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr', 'elapsed_seconds']
PREDICTION_COLUMNS = ['image_id', 'relative_path', 'true_index', 'pred_index', 'true_breed', 'pred_breed', 'confidence', 'correct']

def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available; choose --device cpu or auto.')
    return device

def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def evaluate(model, loader, device, criterion, collect=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            probabilities = logits.float().softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (predicted == labels).sum().item()
            total += count
            if collect:
                predictions.extend(zip(labels.cpu().tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()))
    if total == 0:
        raise ValueError('A required dataset split is empty.')
    return total_loss / total, correct / total, predictions

def classification_metrics(labels, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, predictions):
        matrix[true, pred] += 1
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros_like(tp), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros_like(tp), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--run', required=True, choices=['scratch', 'finetune'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 40 if args.run == 'scratch' else 30
    if min(args.epochs, args.batch_size, args.image_size, args.patience) <= 0 or args.workers < 0:
        parser.error('epochs, batch-size, image-size and patience must be positive; workers must be nonnegative')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old.get('run') != args.run:
            raise ValueError('The output directory belongs to a different run; choose a separate --out-dir.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    frame = pd.read_csv(args.manifest)
    parts = {split: frame.loc[frame['split'] == split].reset_index(drop=True) for split in ('train', 'val', 'test')}
    if any(part.empty for part in parts.values()):
        raise ValueError('Manifest must contain nonempty train, val and test splits.')
    names = [item[0] for item in BREEDS]
    names_zh = [item[1] for item in BREEDS]
    if not frame['class_index'].between(0, len(names) - 1).all():
        raise ValueError('Manifest contains class indices outside the breed list.')
    if not all(names[int(row.class_index)] == row.breed for row in frame.itertuples()):
        raise ValueError('Manifest class indices do not match the breed list.')
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for split, part in parts.items():
        dataset = CatDataset(part, args.data_root, make_transforms(image_size=args.image_size, training=split == 'train'))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train', num_workers=args.workers,
                                    pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0,
                                    worker_init_fn=seed_worker, generator=generator if split == 'train' else None)
    lr = 1e-3 if args.run == 'scratch' else 3e-4
    config = vars(args).copy()
    config.update({'learning_rate': lr, 'weight_decay': 1e-4, 'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR',
                   'pretrained': args.run == 'finetune', 'num_classes': len(names), 'class_names': names,
                   'class_names_zh': names_zh, 'device_used': str(device), 'python': platform.python_version(),
                   'pytorch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'split_counts': {key: len(value) for key, value in parts.items()},
                   'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225]})
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Starting {args.run}: device={device}, epochs={args.epochs}, sizes="
          f"{len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}", flush=True)
    model = create_model(len(names), pretrained=args.run == 'finetune').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        f.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_correct = 0
            total = 0
            current_lr = optimizer.param_groups[0]['lr']
            for images, labels in loaders['train']:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                count = labels.size(0)
                running_loss += loss.item() * count
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                total += count
            val_loss, val_acc, _ = evaluate(model, loaders['val'], device, criterion)
            scheduler.step()
            row = {'epoch': epoch, 'train_loss': running_loss / total, 'train_accuracy': running_correct / total,
                   'val_loss': val_loss, 'val_accuracy': val_acc, 'lr': current_lr,
                   'elapsed_seconds': time.perf_counter() - epoch_start}
            writer.writerow(row)
            f.flush()
            print(f"epoch {epoch}/{args.epochs} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={row['elapsed_seconds']:.1f}s", flush=True)
            if val_acc > best_acc:
                best_acc, best_epoch, stale = val_acc, epoch, 0
                torch.save({'state_dict': model.state_dict(), 'class_names': names, 'class_names_zh': names_zh,
                            'num_classes': len(names), 'image_size': args.image_size, 'run': args.run,
                            'best_epoch': epoch, 'best_val_accuracy': best_acc, 'pretrained': args.run == 'finetune',
                            'normalization_mean': config['normalization_mean'], 'normalization_std': config['normalization_std']}, out / 'best.pt')
            else:
                stale += 1
                if stale >= args.patience:
                    print(f'Early stopping after {epoch} epochs.', flush=True)
                    break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, accuracy, predictions = evaluate(model, loaders['test'], device, criterion, collect=True)
    precision, recall, f1 = classification_metrics([p[0] for p in predictions], [p[1] for p in predictions], len(names))
    metrics = {'test_accuracy': accuracy, 'macro_precision': precision, 'macro_recall': recall, 'macro_f1': f1,
               'num_test': len(predictions), 'best_epoch': best_epoch, 'best_val_accuracy': best_acc,
               'training_seconds': training_seconds, 'test_loss': test_loss, 'evaluated_checkpoint': 'best.pt',
               'evaluation_split': 'test'}
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row, (true, pred, confidence) in zip(parts['test'].itertuples(), predictions):
            writer.writerow({'image_id': row.image_id, 'relative_path': row.relative_path, 'true_index': true,
                             'pred_index': pred, 'true_breed': names[true], 'pred_breed': names[pred],
                             'confidence': confidence, 'correct': true == pred})
    print(f'Test accuracy={accuracy:.4f}, macro F1={f1:.4f}; best epoch={best_epoch}', flush=True)

if __name__ == '__main__':
    main()
