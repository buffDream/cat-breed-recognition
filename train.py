"""Train and evaluate the hand-written ResNet-18 on the prepared manifest."""
import argparse
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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

def evaluate(model, loader, device, criterion, collect=False, tta=False):
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
                if tta:
                    flipped_logits = model(torch.flip(images, dims=[3]))
                    probabilities = (logits.float().softmax(dim=1) + flipped_logits.float().softmax(dim=1)) / 2
                    loss = criterion(probabilities.clamp_min(1e-12).log(), labels)
                else:
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

def balanced_sample_weights(frame, num_classes, base_weight=1.0):
    """Give each class equal total mass and Oxford samples a within-class multiplier."""
    labels = frame['class_index'].to_numpy(dtype=np.int64)
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError('Invalid class index in training data.')
    sources = frame['source'].fillna('') if 'source' in frame.columns else pd.Series('Oxford-IIIT Pet', index=frame.index)
    raw = np.where(sources.astype(str).str.contains('Oxford', case=False, regex=False).to_numpy(), base_weight, 1.0)
    totals = np.bincount(labels, weights=raw, minlength=num_classes)
    if (totals == 0).any():
        raise ValueError('Balanced sampling requires at least one training image per class.')
    return raw / totals[labels]

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
    parser.add_argument('--architecture', choices=['resnet18', 'resnet50'], default='resnet18')
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--initial-checkpoint')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--base-weight', type=float, default=1.0)
    parser.add_argument('--backbone-lr', type=float)
    parser.add_argument('--head-lr', type=float)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 40 if args.run == 'scratch' else 30
    if min(args.epochs, args.batch_size, args.image_size, args.patience) <= 0 or args.workers < 0:
        parser.error('epochs, batch-size, image-size and patience must be positive; workers must be nonnegative')
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error('warmup-epochs must be nonnegative and smaller than epochs')
    if not 0 <= args.label_smoothing < 1:
        parser.error('label-smoothing must be in [0, 1)')
    for name in ('base_weight', 'backbone_lr', 'head_lr'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'{name.replace("_", "-")} must be finite and positive')
    if args.initial_checkpoint and args.run != 'finetune':
        parser.error('--initial-checkpoint requires --run finetune')
    out = Path(args.out_dir)
    if args.initial_checkpoint and Path(args.initial_checkpoint).resolve() == (out / 'best.pt').resolve():
        parser.error('initial checkpoint must be outside the output best.pt path')
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
        sampler = None
        if split == 'train' and args.balance:
            weights = balanced_sample_weights(part, len(names), args.base_weight)
            sampler = WeightedRandomSampler(weights, num_samples=len(part), replacement=True,
                                            generator=torch.Generator().manual_seed(args.seed))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=(split == 'train' and sampler is None), sampler=sampler, num_workers=args.workers,
                                    pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0,
                                    worker_init_fn=seed_worker, generator=generator if split == 'train' else None)
    lr = 1e-3 if args.run == 'scratch' else 3e-4
    backbone_lr = 5e-5 if args.architecture == 'resnet50' and args.run == 'finetune' else lr
    head_lr = 5e-4 if args.architecture == 'resnet50' and args.run == 'finetune' else lr
    if args.backbone_lr is not None:
        backbone_lr = args.backbone_lr
    if args.head_lr is not None:
        head_lr = args.head_lr
    config = vars(args).copy()
    config.update({'learning_rate': head_lr, 'backbone_learning_rate': backbone_lr, 'head_learning_rate': head_lr, 'weight_decay': 1e-4, 'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR',
                   'pretrained': args.run == 'finetune', 'num_classes': len(names), 'class_names': names,
                   'class_names_zh': names_zh, 'device_used': str(device), 'python': platform.python_version(),
                   'pytorch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'split_counts': {key: len(value) for key, value in parts.items()},
                   'train_source_counts': (parts['train']['source'].fillna('unknown').astype(str).value_counts().to_dict()
                                           if 'source' in parts['train'].columns else {'Oxford-IIIT Pet': len(parts['train'])}),
                   'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225]})
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Starting {args.run}: device={device}, epochs={args.epochs}, sizes="
          f"{len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}", flush=True)
    initial = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
        if initial.get('architecture', 'resnet18') != args.architecture:
            raise ValueError('Initial checkpoint architecture does not match --architecture.')
        if initial.get('class_names') != names:
            raise ValueError('Initial checkpoint class_names do not match the manifest class order.')
    model = create_model(len(names), pretrained=args.run == 'finetune' and initial is None,
                         architecture=args.architecture)
    if initial is not None:
        model.load_state_dict(initial['state_dict'], strict=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith('fc.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': backbone_lr},
                                   {'params': model.fc.parameters(), 'lr': head_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_acc = -1.0
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    def save_best(epoch, val_accuracy):
        torch.save({'state_dict': model.state_dict(), 'class_names': names, 'class_names_zh': names_zh,
                    'num_classes': len(names), 'image_size': args.image_size, 'run': args.run,
                    'best_epoch': epoch, 'best_val_accuracy': val_accuracy, 'pretrained': args.run == 'finetune',
                    'architecture': args.architecture, 'tta': args.tta, 'warmup_epochs': args.warmup_epochs,
                    'label_smoothing': args.label_smoothing, 'backbone_learning_rate': backbone_lr,
                    'head_learning_rate': head_lr, 'weight_decay': 1e-4, 'epochs': args.epochs,
                    'batch_size': args.batch_size, 'seed': args.seed, 'patience': args.patience,
                    'initial_checkpoint': args.initial_checkpoint, 'balance': args.balance,
                    'base_weight': args.base_weight, 'train_source_counts': config['train_source_counts'],
                    'normalization_mean': config['normalization_mean'], 'normalization_std': config['normalization_std']}, out / 'best.pt')
    if initial is not None:
        val_loss, best_acc, _ = evaluate(model, loaders['val'], device, criterion, tta=args.tta)
        save_best(0, best_acc)
        print(f'Initial checkpoint: val_loss={val_loss:.4f} val_acc={best_acc:.4f} (epoch 0)', flush=True)
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        f.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            warmup = epoch <= args.warmup_epochs
            for parameter in backbone:
                parameter.requires_grad_(not warmup)
            if warmup:
                # Keep the pretrained feature extractor's running statistics fixed.
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            running_loss = 0.0
            running_correct = 0
            total = 0
            current_lr = optimizer.param_groups[-1]['lr']
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
            val_loss, val_acc, _ = evaluate(model, loaders['val'], device, criterion, tta=args.tta)
            if not warmup:
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
                save_best(epoch, best_acc)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f'Early stopping after {epoch} epochs.', flush=True)
                    break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, accuracy, predictions = evaluate(model, loaders['test'], device, criterion, collect=True, tta=args.tta)
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
