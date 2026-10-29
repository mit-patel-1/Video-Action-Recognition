"""Focused tests for the final UCF50 CNN-LSTM pipeline."""

import json
import math
import re

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import models
import train as training
import utils
import video_datasets
from test import test as evaluate


@pytest.fixture(autouse=True)
def cpu_random_seed():
    """Keep synthetic tests repeatable and limit CPU thread overhead."""
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        yield
    torch.set_num_threads(threads)


def test_source_groups_stay_separate_and_saved_split_is_reused(tmp_path, monkeypatch):
    """Keep related videos together, cover all classes, and reuse the saved split."""
    videos = {
        f'Class{label}/v_Class{label}_g{group:02d}_c{clip:02d}': label
        for label in range(50)
        for group in range(1, 26)
        for clip in range(1, 6)
    }
    labels = {f'Class{label}': label for label in range(50)}
    monkeypatch.setattr(video_datasets, 'load_dataset', lambda _path: (videos, labels))
    path = tmp_path / 'splits.json'
    result = video_datasets.load_splits('frames', path, 0.75, 0.15, 0)
    splits = list(result[0].values())
    paths = [{name for name, _ in split} for split in splits]
    groups = [
        {(label, re.search(r'_g(\d+)_', name).group(1)) for name, label in split}
        for split in splits
    ]
    assert set.union(*paths) == set(videos)
    assert sum(map(len, paths)) == len(videos)
    assert all({label for _, label in split} == set(range(50)) for split in splits)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert groups[first].isdisjoint(groups[second])
    assert video_datasets.load_splits('frames', path, 0.75, 0.15, 0) == result

    saved = json.loads(path.read_text(encoding='utf-8'))
    saved['splits']['val'].append(saved['splits']['train'].pop(0))
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='Source groups overlap'):
        video_datasets.load_splits('frames', path, 0.75, 0.15, 0)


def test_augmentation_is_shared_within_each_clip():
    """Identical frames stay identical, while different clips can be augmented differently."""
    transform, _ = utils.compose_data_transforms(*utils.transform_stats('3dcnn'))
    dataset = video_datasets.VideoDataset([], 16, transform)
    frame = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    augmented = []
    for seed in range(4):
        torch.manual_seed(seed)
        clip = dataset._transform_frames([frame.copy() for _ in range(16)])
        torch.testing.assert_close(clip, clip[:1].expand_as(clip), rtol=0, atol=0)
        augmented.append(clip[0])
    assert any(not torch.equal(augmented[0], other) for other in augmented[1:])


def test_video_lrcn_sequence_batch_independence_training_and_reload(tmp_path):
    """Exercise the real video CNN and LSTM without downloading pretrained weights."""
    config = dict(hidden_size=100, n_layers=1, dropout_rate=0.1,
                  n_classes=50, cnn_model='r2plus1d_18')
    model = models.LRCN(**config, pretrained=False).eval()
    clips = torch.randn(2, 16, 3, 16, 16)
    sequence_lengths = []
    hook = model.rnn.register_forward_pre_hook(
        lambda _module, args: sequence_lengths.append(len(args[0].batch_sizes)))
    with torch.no_grad():
        together = model(clips)
        alone = model(clips[1:])
        reordered = model(clips.flip(0)).flip(0)
    hook.remove()
    assert together.shape == (2, 50)
    assert sequence_lengths == [2, 2, 2]
    torch.testing.assert_close(together[1:], alone, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(together, reordered, rtol=1e-4, atol=1e-6)

    # The final schedule initially trains the recurrent layer and classifier only.
    model.base_model.requires_grad_(False)
    model.train()
    model.base_model.eval()
    before = model.rnn.weight_ih_l0.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001)
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(clips), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.rnn.weight_ih_l0.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.rnn.weight_ih_l0)

    model.eval()
    with torch.no_grad():
        expected = model(clips)
    path = tmp_path / 'model.pt'
    torch.save({'model_config': config, 'model_state_dict': model.state_dict()}, path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    restored = models.LRCN(**checkpoint['model_config'], pretrained=False).eval()
    restored.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(clips))


class ClipScores(nn.Module):
    """Return known clip logits so expected video metrics can be calculated by hand."""

    def forward(self, clips, lengths):
        """Read logits stored in the synthetic clip."""
        del lengths
        return clips[:, 0, :, 0, 0]


def test_loss_and_accuracy_count_videos_not_clips():
    """Average clips within a video, then weight videos equally across uneven batches."""
    probabilities = [[0.9, 0.2, 0.6], [0.8], [0.1]]
    samples = [
        ([torch.tensor([p, 1 - p]).log().reshape(1, 2, 1, 1) for p in windows], label)
        for windows, label in zip(probabilities, [0, 1, 1])
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=video_datasets.collate_fn_rnn)
    expected_loss = (
        sum(-math.log(p) for p in probabilities[0]) / 3 - math.log(0.2) - math.log(0.9)
    ) / 3
    model = ClipScores()
    loss, accuracy = training.get_epoch_loss(
        model, nn.CrossEntropyLoss(reduction='sum'), loader, torch.device('cpu'))
    targets, predictions, test_accuracy, test_loss = evaluate(
        model, loader, torch.device('cpu'), return_loss=True)
    assert targets == [0, 1, 1]
    assert predictions == [0, 0, 1]
    assert accuracy == test_accuracy == pytest.approx(2 / 3)
    assert loss == pytest.approx(expected_loss)
    assert test_loss == pytest.approx(expected_loss)


class SmallVideoBackbone(nn.Module):
    """A small video CNN for checking six training epochs quickly on the CPU."""

    def __init__(self, **_kwargs):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(3, 4, 1), nn.BatchNorm3d(4))
        self.layer1 = nn.Identity()
        self.layer2 = nn.AvgPool3d((2, 1, 1))
        self.layer3 = nn.AvgPool3d((2, 1, 1))
        self.layer4 = nn.Sequential(nn.Conv3d(4, 4, 1), nn.AvgPool3d((2, 1, 1)))
        self.fc = nn.Linear(4, 50)


def test_staged_training_freezes_cnn_then_updates_only_last_block(tmp_path, monkeypatch):
    """Check actual weight updates and fixed BatchNorm through the epoch-six transition."""
    monkeypatch.setattr(models, 'r2plus1d_18', SmallVideoBackbone)
    model = models.LRCN(100, 1, 0.1, 50, pretrained=False, cnn_model='r2plus1d_18')
    samples = [(torch.randn(16, 3, 4, 4), label) for label in (0, 1)]
    loader = DataLoader(samples, batch_size=2, collate_fn=video_datasets.collate_fn_rnn)
    optimizer = torch.optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': 0.00001},
        {'params': list(model.rnn.parameters()) + list(model.fc.parameters()), 'lr': 0.001},
    ], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    epoch_loss = training.get_epoch_loss
    updates = []

    def observe_updates(current, criterion, batches, device, optimizer=None):
        """Record changed weights around the real training step."""
        before = {name: value.detach().clone() for name, value in current.named_parameters()}
        buffers = {name: value.clone() for name, value in current.base_model.named_buffers()}
        result = epoch_loss(current, criterion, batches, device, optimizer)
        if optimizer is not None:
            updates.append({name for name, value in current.named_parameters()
                            if not torch.equal(before[name], value)})
            for name, value in current.base_model.named_buffers():
                torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        return result

    monkeypatch.setattr(training, 'get_epoch_loss', observe_updates)
    training.train({'train': loader, 'val': loader}, model, nn.CrossEntropyLoss(reduction='sum'),
                   optimizer, scheduler, torch.device('cpu'), str(tmp_path),
                   n_epochs=6, freeze_epochs=5)
    for changed in updates[:5]:
        assert not any(name.startswith('base_model.') for name in changed)
        assert 'rnn.weight_ih_l0' in changed and 'fc.weight' in changed
    assert 'base_model.layer4.0.weight' in updates[5]
    assert not any(name.startswith('base_model.stem.') for name in updates[5])


def test_best_validation_checkpoint_logging_and_lr_reduction(tmp_path, monkeypatch):
    """Save the best validation model and keep current weights during LR reductions."""
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    # Lowest loss is at epoch 3, but the checkpoint must use epoch 2's best accuracy.
    validation = iter([(1.0, 0.4), (0.8, 0.9), (0.7, 0.6), (0.9, 0.7), (1.0, 0.65)])
    starts, ends, records = [], [], []

    def controlled_epoch(current, _criterion, _loader, _device, optimizer=None):
        """Use fixed validation results to test checkpoint decisions."""
        if optimizer is None:
            return next(validation)
        starts.append(current.weight.item())
        optimizer.zero_grad()
        (-current.weight.sum()).backward()
        optimizer.step()
        ends.append(current.weight.item())
        return 0.5, 1.0

    monkeypatch.setattr(training, 'get_epoch_loss', controlled_epoch)
    training.train({'train': None, 'val': None}, model, None, optimizer, scheduler,
                   torch.device('cpu'), str(tmp_path), n_epochs=5, logger=records.append,
                   checkpoint_info={})
    assert starts[1:] == pytest.approx(ends[:-1])
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.025)
    checkpoint = torch.load(tmp_path / 'best_model_wts.pt', weights_only=True)
    assert checkpoint['epoch'] == 2
    assert checkpoint['val_accuracy'] == 0.9
    assert checkpoint['model_state_dict']['weight'].item() == pytest.approx(ends[1])
    assert model.weight.item() == pytest.approx(ends[1])
    assert [record['epoch'] for record in records] == [1, 2, 3, 4, 5]
    assert all(set(record) == {'epoch', 'train/loss', 'train/accuracy',
                               'val/loss', 'val/accuracy'} for record in records)
    local = [json.loads(line) for line in
             (tmp_path / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert local == records
