# UCF50 Video Action Recognition

This project uses a CNN and an LSTM to classify 50 actions in UCF50. The final
R(2+1)D-18 CNN-LSTM completed 50 training epochs and reached **93.25% test accuracy
(939 of 1,007 videos)**, meeting the assignment's 93% requirement.

## Errors Found and Fixed

1. **The LSTM mixed different videos together.** The original LSTM expected the
   sequence dimension first, but it received tensors shaped as batch, one time step,
   and features. It therefore treated videos in the batch as positions in a sequence.
   The corrected model uses `batch_first=True` and sends each video's feature
   sequence to the LSTM. Classification uses that video's final real hidden state.

2. **The dataset split allowed source-group leakage.** The original code split
   individual videos using class labels only. Related clips from the same UCF50
   source group could enter different splits. The corrected code splits whole
   `(action class, source group)` groups and saves the split in `data/splits.json`.
   The final split has no overlapping videos or source groups between training,
   validation, and testing.

3. **Learning-rate reductions mixed old weights with newer optimizer state.** The
   original training loop reloaded older best model weights whenever the learning
   rate dropped, but kept the optimizer's newer state. This could leave the weights
   and optimizer state out of sync. Training now continues from the current weights
   when the learning rate drops. The best validation checkpoint is saved separately
   and loaded after training finishes.

4. **Random augmentations were different across frames in the same video.** The
   original code chose random flips and translations separately for each frame.
   This could introduce artificial changes between frames. The corrected code
   applies the same flip and translation to every frame within a clip, keeping
   the frames consistent for the CNN and LSTM.

The training script also imported a missing `compose_dataloaders` function.
That helper was added, and `run_training.py` now uses the same training code and
saved split as `run.py`. The supplied shell scripts were changed from HMDB51's 51
classes to UCF50's 50 classes, and the testing script now points to the final checkpoint.

## Two Model-Level Improvements

### 1. Use a video CNN before the LSTM

The original model used a pretrained ResNet34 to extract features from each frame
separately. The final model uses R(2+1)D-18, which applies spatial and temporal
convolutions to a video clip. This lets the CNN represent changes across frames
before passing features to the LSTM. Spatial averaging keeps the time dimension;
it does not reduce the whole clip to one feature vector. Sixteen input frames
normally produce two CNN time steps for the LSTM.

The model keeps the LRCN structure, using an R(2+1)D CNN followed by an LSTM.

### 2. Use video pretraining with staged fine-tuning

The original ResNet34 already used ImageNet-pretrained weights. The final model
starts with Kinetics-400 video weights. The CNN is frozen for the first five epochs
while the LSTM and classifier learn the UCF50 labels. From epoch 6, only the CNN's
last block is trained, using a lower learning rate than the LSTM and classifier.
CNN BatchNorm statistics stay fixed. This keeps the earlier pretrained features
stable while allowing the last block to adapt to the action dataset.

## Step 1: Set Up the Environment and Dataset

The final run used Python 3.12.4, PyTorch 2.3.1 with CUDA 12.1, and torchvision
0.18.1 on an NVIDIA RTX 4090. Install the packages and log in to W&B:

```bash
python -m pip install -r requirements.txt
python -m wandb login
```

Download [UCF50](https://www.crcv.ucf.edu/data/UCF50.rar) and extract its class folders
under `data/UCF50`. For a new setup, prepare the frame folders and saved split:

```bash
python prepare_data.py --source data/UCF50 --output UCF50_frames --frames 16 --workers 4
```

The frame folders identify the videos used by the saved split. The final model
reads its clips directly from the original AVI files in `data/UCF50`.

| Split | Videos | Classes |
|---|---:|---:|
| Training | 4,997 | 50 |
| Validation | 677 | 50 |
| Test | 1,007 | 50 |

## Step 2: Train the Model

The final run used the following settings:

| Setting | Value |
|---|---|
| CNN | R(2+1)D-18, Kinetics-400 pretrained |
| LSTM | One layer, 100 hidden units, final-state classification |
| Output / internal LSTM dropout | 0.1 / 0.0 |
| Epochs | 50 |
| Training batch size | 16 videos |
| Validation and test batch size | 32 videos |
| Clip sampling | Up to 16 frames at 15 fps |
| Training windows | One random window per video |
| Evaluation windows | Up to three fixed windows per video |
| Image preprocessing | Full-frame 112 by 112 resize, Kinetics normalization |
| Training augmentation | Horizontal flip probability 0.5 and translation range 0.1, shared within each clip |
| Optimizer | AdamW, weight decay 0.0001 |
| Starting learning rates | 0.001 for LSTM/classifier, 0.00001 for CNN |
| CNN fine-tuning | Frozen for five epochs, then last block only; fixed BatchNorm |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 5 |

To start training, run:

```bash
bash train.sh
```

`run_training.py` is another way to start training using the same training code.
Training outputs are saved in `models/training-video-final`. Set a different
`--output_dir` in `train.sh` for each new run.

## Step 3: Test the Model and View Results

Training selects the checkpoint with the highest validation accuracy. Testing
uses a separate test split.

| Result | Value |
|---|---:|
| Selected checkpoint epoch | 8 |
| Best validation accuracy | 94.68% |
| Test accuracy | 93.25% (939/1,007) |
| Test loss | 0.347456 |

[The W&B run](https://wandb.ai/mp2591/assignment4-ucf50/runs/p37cdpp5)
contains training and validation loss/accuracy for all 50 epochs and the final test
loss/accuracy.

- Checkpoint: `models/training-video-final/best_model_wts.pt`
- Training and validation records: `models/training-video-final/metrics.jsonl`
- Test result: `models/training-video-final/test_results.json`

To run the testing script, use:

```bash
bash test.sh
```

The testing script reads the model and sampling settings saved in the checkpoint.

## Unit Tests and Lint

The unit tests cover source-group splitting, shared clip
augmentation, the R(2+1)D-LSTM model, video-level loss and accuracy, staged
fine-tuning, and validation checkpoint selection with metric logging. The model
test also checks that videos stay separate and saved weights load correctly.
Run the unit tests and check the code with Pylint:

```bash
python -m pytest -q
python -m pylint models.py video_datasets.py utils.py train.py test.py run.py run_training.py prepare_data.py tests/test_assignment.py
```

Known lint issues include style warnings and OpenCV member-detection errors.
