"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video sequences.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import glob
import re
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
        video_dir (str, optional): Original AVI folder for sampling short clips.
        clip_fps (float): Sampling rate for short clips.
        eval_clips (int): Number of fixed windows during evaluation.
        training (bool): Select a random window when True.
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None, *, video_dir=None,
                 clip_fps=5.0, eval_clips=3, training=False):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.video_dir = Path(video_dir) if video_dir else None
        self.clip_fps = clip_fps
        self.eval_clips = eval_clips
        self.training = training
        self.video_lengths = {}
        if fr_per_vid < 1 or clip_fps <= 0 or eval_clips < 1:
            raise ValueError('Frame count, sampling rate, and clip count must be positive.')

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: Frames and one video label. Short-clip sampling returns a list of
                (T, C, H, W) tensors. Cached frames return one such tensor.
        """
        if self.video_dir is not None:
            return self._video_clips(idx), self.dataset[idx][1]

        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = sorted(
            glob.glob(self.dataset[idx][0] + '/*.jpg'),
            key=lambda path: int(re.search(r'(\d+)\.jpg$', path).group(1)),
        )
        fr_paths = fr_paths[:self.fpv]
        
        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        
        return self._transform_frames(fr_imgs), fr_label

    def _transform_frames(self, fr_imgs):
        """Use the same flip and translation for all frames in one clip."""
        fr_imgs_trans = fr_imgs
        if self.transforms and fr_imgs:
            random_state = torch.get_rng_state()
            fr_imgs_trans = [self.transforms(fr_imgs[0])]
            next_state = torch.get_rng_state()
            try:
                for frame in fr_imgs[1:]:
                    torch.set_rng_state(random_state)
                    fr_imgs_trans.append(self.transforms(frame))
            finally:
                torch.set_rng_state(next_state)

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans

    def _video_clips(self, idx):
        """Read one random training clip or fixed evaluation windows from a video."""
        sample_path = Path(self.dataset[idx][0])
        source = self.video_dir / sample_path.parent.name / (sample_path.name + '.avi')
        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or fps <= 0:
                raise OSError('Could not read video: ' + str(source))
            if idx not in self.video_lengths:
                # Some AVI headers report frames that cannot be decoded.
                total = 0
                while capture.grab():
                    total += 1
                self.video_lengths[idx] = total
                capture.release()
                capture = cv2.VideoCapture(str(source))
            total = self.video_lengths[idx]
            if total < 1:
                raise OSError('Video has no readable frames: ' + str(source))
            offsets = np.rint(np.arange(self.fpv) * max(fps / self.clip_fps, 1)).astype(int)
            offsets = offsets[offsets < total]
            last_start = total - 1 - int(offsets[-1])
            if self.training:
                starts = [int(torch.randint(last_start + 1, (1,)).item())]
            else:
                starts = np.unique(np.linspace(0, last_start, self.eval_clips, dtype=int))
            windows = [offsets + start for start in starts]
            wanted = set(np.concatenate(windows).tolist())
            first, last = min(wanted), max(wanted)
            if first:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                    raise OSError('Could not seek in video: ' + str(source))
            frames = {}
            for position in range(first, last + 1):
                if not capture.grab():
                    raise OSError('Could not decode video: ' + str(source))
                if position in wanted:
                    success, frame = capture.retrieve()
                    if not success:
                        raise OSError('Could not decode video frame: ' + str(source))
                    frames[position] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return [self._transform_frames([frames[int(position)] for position in window])
                    for window in windows]
        finally:
            capture.release()


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split UCF50 by source group while keeping all classes in each split.

    Clips from the same group and class stay together. Ratios apply to groups,
    so video counts can differ from the requested ratios.
    
    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.
    
    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    if not 0 < tr_ratio < 1 or not 0 < ts_ratio < 1 or tr_ratio + ts_ratio >= 1:
        raise ValueError('Train, validation, and test ratios must all be positive.')
    grouped_videos = {}
    for video_path, label in sorted(vid_dataset.items()):
        video_name = os.path.basename(os.path.normpath(video_path))
        match = re.search(r'_g(\d+)_c\d+(?:\.[^.]+)?$', video_name)
        if match is None:
            raise ValueError('Missing UCF50 source group in video name: ' + video_name)
        group = (label, int(match.group(1)))
        grouped_videos.setdefault(group, []).append((video_path, label))
    group_keys = list(grouped_videos)
    group_indices = np.arange(len(group_keys))
    group_labels = np.array([group[0] for group in group_keys])
    print('Splitting train/validation/test datasets....')

    # Split whole source groups instead of individual clips.
    ts_spliter = StratifiedShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    tr_val_idx, ts_idx = next(ts_spliter.split(group_indices, group_labels))

    # Train/validation split
    val_ratio = 1 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    val_spliter = StratifiedShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    tr_idx, val_idx = next(val_spliter.split(tr_val_idx, group_labels[tr_val_idx]))
    split_indices = (tr_val_idx[tr_idx], tr_val_idx[val_idx], ts_idx)
    tr_dataset, val_dataset, ts_dataset = [
        [video for idx in indices for video in grouped_videos[group_keys[idx]]]
        for indices in split_indices
    ]

    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W), labels, and real lengths.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor, lengths)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
            - lengths (Tensor): Number of real frames in each video.
            Short-clip batches also return video indices for combining clip predictions.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None, None
    imgs_batch, label_batch = zip(*valid_samples)

    # Keep one label per video, while the CNN and LSTM process each clip.
    if isinstance(imgs_batch[0], list):
        clips, video_indices = [], []
        for video_index, windows in enumerate(imgs_batch):
            clips.extend(windows)
            video_indices.extend([video_index] * len(windows))
        return (pad_sequence(clips, batch_first=True), torch.tensor(label_batch),
                torch.tensor([len(clip) for clip in clips], dtype=torch.long),
                torch.tensor(video_indices, dtype=torch.long))
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    lengths = torch.tensor([len(imgs) for imgs in imgs_batch], dtype=torch.long)
    
    return padded_imgs, labels_tensor, lengths


def load_splits(frame_dir, split_file, train_size, test_size, seed):
    """Create one group split, then reuse it for every experiment."""
    videos, labels = load_dataset(frame_dir)
    path = Path(split_file)
    settings = {'train_size': train_size, 'test_size': test_size, 'seed': seed}
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['settings'] != settings or saved['labels'] != labels:
            raise ValueError('The saved split settings or classes do not match this run.')
    else:
        train, validation, test = dataset_split(videos, train_size, test_size, seed)
        saved = {'settings': settings, 'labels': labels,
                 'splits': {'train': train, 'val': validation, 'test': test}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    splits = {name: [(video, int(label)) for video, label in samples]
              for name, samples in saved['splits'].items()}
    groups, seen = [], set()
    for name in ('train', 'val', 'test'):
        current_groups = set()
        if {label for _, label in splits[name]} != set(labels.values()):
            raise ValueError('Each split must contain every action class.')
        for video, label in splits[name]:
            if video in seen or videos.get(video) != label:
                raise ValueError('The saved splits contain repeated or unknown videos.')
            seen.add(video)
            match = re.search(r'_g(\d+)_c\d+(?:\.[^.]+)?$', Path(video).name)
            if match is None:
                raise ValueError('A video is missing its UCF50 source group.')
            current_groups.add((label, int(match.group(1))))
        groups.append(current_groups)
    if seen != set(videos):
        raise ValueError('The saved splits do not cover the current dataset.')
    if any(groups[first] & groups[second] for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Source groups overlap between dataset splits.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return splits, labels, digest
