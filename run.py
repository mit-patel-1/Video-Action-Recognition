"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.
    
The module also includes a helper function for parsing command-line arguments.
"""

import json
import random
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np

from video_datasets import VideoDataset, load_splits
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import train
from test import test

def model_settings(args):
    """Store the settings needed to build the same model during evaluation."""
    return {'hidden_size': args.rnn_hidden_size, 'n_layers': args.rnn_n_layers,
            'dropout_rate': args.dropout, 'n_classes': args.n_classes,
            'cnn_model': args.cnn_backbone, 'feature_fusion': args.feature_fusion,
            'temporal_pool': args.temporal_pool}


def start_wandb(args, config):
    """Log required metrics to hosted W&B without code or system statistics."""
    if not args.wandb:
        return None
    import wandb  # Imported only when online logging is requested.

    settings = wandb.Settings(
        base_url='https://api.wandb.ai', x_disable_stats=True, x_disable_meta=True,
        disable_git=True, disable_code=True, disable_job_creation=True, console='off',
        x_save_requirements=False,
    )
    resume = {}
    previous = Path(args.output_dir) / 'wandb_run.json'
    if (args.mode == 'eval' or args.ckpt) and previous.exists():
        resume = {'id': json.loads(previous.read_text(encoding='utf-8'))['id'], 'resume': 'must'}
        if args.mode == 'train':
            resume['allow_val_change'] = True
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.run_name, config=config, settings=settings, mode='online', **resume)
    run.define_metric('epoch', hidden=True)
    run.define_metric('train/*', step_metric='epoch')
    run.define_metric('val/*', step_metric='epoch')
    return run


def args_parser(training_only=False):
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', default='UCF50_frames', help='Directory for video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.75, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.15, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--video_dir', help='Original videos for short-clip sampling')
    parser.add_argument('--clip_fps', type=float, default=5.0, help='Frames sampled per second')
    parser.add_argument('--eval_clips', type=int, default=3, help='Fixed windows per evaluation video')
    parser.add_argument('--preserve_aspect', action='store_true',
                        help='Resize the short side to 256 and center crop to 224')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='ResNet image backbone or r2plus1d_18 video backbone')
    parser.add_argument('-p', '--pretrained', action=argparse.BooleanOptionalAction, default=True, help='Use pretrained CNN weights')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", choices=['train', 'eval'], required=not training_only)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr_scheduler', choices=['plateau', 'cosine'], default='plateau',
                        help='Learning rate schedule')
    parser.add_argument('--staged_finetune', action='store_true',
                        help='Freeze the CNN for five epochs, then train its last block')
    parser.add_argument('--backbone_learning_rate', type=float,
                        help='CNN learning rate for staged fine-tuning')

    parser.add_argument('--feature_fusion', action='store_true', help='Combine mean CNN features with LSTM output')
    parser.add_argument('--temporal_pool', action='store_true',
                        help='Classify valid LSTM steps and average their probabilities')
    parser.add_argument('--output_dir', default='models/baseline')
    parser.add_argument('--split_file', default='data/splits.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', action='store_true', help='Log required metrics to hosted W&B')
    parser.add_argument('--wandb_project', default='assignment4-ucf50')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--run_name', default='baseline')
    parser.add_argument('--evaluate_test', action='store_true', help='Evaluate the held-out test set after training')
    parser.add_argument('--smoke_test', action='store_true', help='Check one small training and validation batch')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training device:', device, flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'eval' or args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        settings = checkpoint['model_config']
        model = LRCN(**settings, pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        sampling = checkpoint.get('sampling_config', {
            'frames_per_video': checkpoint['frames_per_video'], 'video_dir': None,
            'clip_fps': 5.0, 'eval_clips': 1, 'preserve_aspect': False})
        args.fr_per_vid = sampling['frames_per_video']
        args.video_dir = sampling['video_dir']
        args.clip_fps = sampling['clip_fps']
        args.eval_clips = sampling['eval_clips']
        args.preserve_aspect = sampling['preserve_aspect']
        print('Using checkpoint sampling settings:', sampling, flush=True)
    else:
        settings = model_settings(args)
        model = LRCN(**settings, pretrained=args.pretrained).to(device)
        checkpoint = None
    sampling = {'frames_per_video': args.fr_per_vid, 'video_dir': args.video_dir,
                'clip_fps': args.clip_fps, 'eval_clips': args.eval_clips,
                'preserve_aspect': args.preserve_aspect}
    splits, labels, digest = load_splits(
        args.frame_dir, args.split_file, args.train_size, args.test_size, args.seed
    )
    if len(labels) != settings['n_classes']:
        raise ValueError('The number of model classes does not match the dataset.')
    if checkpoint is not None and checkpoint['split_sha256'] != digest:
        raise ValueError('The checkpoint was trained with a different dataset split.')
    if args.smoke_test:
        if args.mode != 'train' or args.evaluate_test:
            raise ValueError('The GPU check must not evaluate the held-out test set.')
        for name in ('train', 'val'):
            selected = {}
            for video, label in splits[name]:
                selected.setdefault(label, (video, label))
            splits[name] = list(selected.values())[:args.batch_size]
    transform_model = '3dcnn' if settings['cnn_model'] == 'r2plus1d_18' else args.model_type
    height, width, mean, std = transform_stats(transform_model)
    train_transform, eval_transform = compose_data_transforms(
        height, width, mean, std, preserve_aspect=args.preserve_aspect)
    datasets = {
        name: VideoDataset(samples, args.fr_per_vid,
                           train_transform if name == 'train' else eval_transform,
                           video_dir=args.video_dir, clip_fps=args.clip_fps,
                           eval_clips=args.eval_clips, training=name == 'train')
        for name, samples in splits.items()
    }
    backbone_lr = args.backbone_learning_rate
    if backbone_lr is None:
        backbone_lr = args.learning_rate / 10
    config = dict(settings, frames_per_video=args.fr_per_vid, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, epochs=args.n_epochs, seed=args.seed,
                  split_sha256=digest, smoke_test=args.smoke_test,
                  num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                  pin_memory=args.pin_memory, cudnn_benchmark=args.cudnn_benchmark,
                  lr_scheduler=args.lr_scheduler, staged_finetune=args.staged_finetune,
                  freeze_epochs=5 if args.staged_finetune else 0,
                  backbone_learning_rate=backbone_lr if args.staged_finetune
                  else args.learning_rate, optimizer='AdamW' if args.staged_finetune else 'Adam',
                  weight_decay=0.0001 if args.staged_finetune else 0.0,
                  clip_consistent_augmentation=True)
    config.update(sampling)
    if args.mode == 'eval' and checkpoint is not None:
        # Keep the saved training settings when logging final evaluation.
        config = checkpoint.get('run_config', config)
    loader_settings = {'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
                       'pin_memory': args.pin_memory and device.type == 'cuda', 'seed': args.seed}
    start_epoch = 0
    if args.mode == 'train' and (output_dir / 'metrics.jsonl').exists():
        if checkpoint is None:
            raise FileExistsError('This output folder already contains a training run.')
        if Path(args.ckpt).resolve() != (output_dir / 'best_model_wts.pt').resolve():
            raise ValueError('Continue this run using its own best checkpoint.')
        history = [json.loads(line) for line in
                   (output_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        start_epoch = history[-1]['epoch']
    if args.mode == 'train' and args.n_epochs <= start_epoch:
        raise ValueError('The total epoch count must be higher than the completed count.')
    run = start_wandb(args, config)
    try:
        if run is not None:
            (output_dir / 'wandb_run.json').write_text(
                json.dumps({'id': run.id, 'url': run.url}, indent=2), encoding='utf-8'
            )
        if args.mode == 'train':
            loaders = train_val_dloaders(datasets['train'], datasets['val'],
                                        args.batch_size, args.model_type, **loader_settings)
            criterion = nn.CrossEntropyLoss(reduction='sum')
            if args.staged_finetune:
                head_parameters = list(model.rnn.parameters()) + list(model.fc.parameters())
                optimizer = optim.AdamW([
                    {'params': head_parameters},
                    {'params': model.base_model.parameters(), 'lr': backbone_lr},
                ], lr=args.learning_rate, weight_decay=0.0001)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            if args.lr_scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs - start_epoch,
                                              eta_min=args.learning_rate / 100)
            else:
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            info = {'model_config': settings, 'label_dict': labels, 'split_sha256': digest,
                    'frames_per_video': args.fr_per_vid, 'sampling_config': sampling,
                    'run_config': config}
            model, _, _ = train(loaders, model, criterion, optimizer, scheduler, device,
                                str(output_dir), args.n_epochs,
                                logger=run.log if run is not None else None, checkpoint_info=info,
                                start_epoch=start_epoch,
                                best_val_acc=checkpoint['val_accuracy'] if checkpoint else -1.0,
                                freeze_epochs=5 if args.staged_finetune else 0)
        if args.mode == 'eval' or args.evaluate_test:
            loader = test_dloaders(datasets['test'], args.batch_size, args.model_type,
                                  **loader_settings)['test']
            _, _, accuracy, loss = test(model, loader, device, return_loss=True)
            results = {'test/loss': loss, 'test/accuracy': accuracy}
            (output_dir / 'test_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            if run is not None:
                run.log(results)
            print(json.dumps(results, indent=2))
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main(args_parser())
