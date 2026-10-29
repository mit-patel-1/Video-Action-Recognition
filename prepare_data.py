"""Extract UCF50 video frames and save the shared source-group split."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from video_datasets import load_splits
from utils import get_frames


def extract_video(task):
    """Save sampled frames for one video and report decoding failures."""
    source, destination, count = task
    existing = [destination / f'frame{index}.jpg' for index in range(count)]
    if all(path.is_file() and path.stat().st_size > 0 for path in existing):
        return {'video': str(source), 'frames': count}
    cv2.setNumThreads(1)
    frames, _ = get_frames(str(source), count)
    if not frames:
        return {'video': str(source), 'error': 'No readable frames'}
    destination.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination / f'frame{index}.jpg'), image):
            raise OSError('Could not save a video frame: ' + str(destination))
    return {'video': str(source), 'frames': len(frames)}


def prepare(args):
    """Prepare all videos before training starts."""
    source = Path(args.source)
    target = Path(args.output)
    videos = sorted(source.glob('*/*.avi'))
    classes = {video.parent.name for video in videos}
    if len(classes) != 50:
        raise ValueError('UCF50 preparation requires all 50 action classes.')
    tasks = [(video, target / video.parent.name / video.stem, args.frames) for video in videos]
    for class_name in sorted(classes):
        (target / class_name).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(extract_video, tasks), total=len(tasks), desc='Extracting videos'))
    failures = [result for result in results if 'error' in result]
    report = {'classes': len(classes), 'videos': len(videos), 'frames_per_video': args.frames,
              'decoded_frames': sum(result.get('frames', 0) for result in results),
              'failures': failures}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/data_preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if failures:
        raise ValueError('Some videos could not be read. Check reports/data_preparation.json.')
    splits, _, digest = load_splits(target, args.split_file, 0.75, 0.15, 0)
    report['split_sizes'] = {name: len(samples) for name, samples in splits.items()}
    report['split_sha256'] = digest
    Path('reports/data_preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def args_parser():
    """Read input and output paths for frame preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/UCF50')
    parser.add_argument('--output', default='UCF50_frames')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split_file', default='data/splits.json')
    return parser.parse_args()


if __name__ == '__main__':
    prepare(args_parser())
