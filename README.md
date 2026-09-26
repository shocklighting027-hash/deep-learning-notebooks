# deep-learning-notebooks

Two PyTorch notebooks built as step-by-step experiment logs: each section changes one thing and records the result,
so the effect of every change is visible.

| Notebook | Task | Best result |
|----------|------|-------------|
| [classification/cifar10_cnn_to_resnet.ipynb](classification/cifar10_cnn_to_resnet.ipynb) | CIFAR-10 image classification: small CNN, augmentation, LR schedules, residual networks, transfer learning | 91.15% test accuracy (ImageNet-pretrained ResNet-18, fine-tuned) |
| [segmentation/oxford_pets_unet.ipynb](segmentation/oxford_pets_unet.ipynb) | Oxford-IIIT Pets semantic segmentation (pet / background / border): FCN, U-Net, class weights, BatchNorm | 0.738 validation mIoU (BatchNorm U-Net) |

Object-detection dataset loading (VOC / COCO / YOLO) lives in a separate package:
[detdata](https://github.com/shocklighting027-hash/detdata).

## Classification: from a small CNN to transfer learning

| # | Experiment | Valid acc | Test acc |
|---|------------|-----------|----------|
| 1 | Small CNN (2 conv blocks), no augmentation, Adam | 0.6374 | not evaluated |
| 2 | 5-conv CNN + flip / crop augmentation, ReduceLROnPlateau | 0.8648 | 0.8642 |
| 3 | + ColorJitter, RandomErasing, weight decay 5e-4 | 0.8626 | 0.8610 |
| 4 | + cosine annealing LR | 0.8649 | 0.8620 |
| 5 | Residual network (custom), Adam + cosine | 0.8708 | 0.8682 |
| 6 | Same network, SGD + momentum, 80 epochs | 0.8816 | 0.8784 |
| 7 | ResNet-18-style network (custom), SGD + momentum | 0.8927 | 0.8907 |
| 8 | ResNet-18 pretrained on ImageNet, full fine-tuning | 0.9203 | 0.9115 |
| 9 | Pretrained ResNet-18, first 7 children frozen | 0.8907 | kernel crashed |
| 10 | Pretrained DenseNet-121 | not completed | not completed |

## Segmentation: FCN to U-Net

Validation metrics at the last epoch:

| # | Model | Valid loss | Accuracy / mIoU | Dice |
|---|-------|-----------|-----------------|------|
| 2 | FCN baseline | 0.6812 | acc 0.5286 | -- |
| 3 | U-Net, 2 levels | 0.3485 | acc 0.7825 | -- |
| 4 | Deeper U-Net, 4 levels | 0.2996 | mIoU 0.7063 | 0.5442 |
| 5 | + computed class weights | 0.3708 | mIoU 0.6925 | 0.5570 |
| 6 | + BatchNorm, fixed class weights | 0.3032 | mIoU 0.7379 | 0.5774 |

## Datasets (not included)

**CIFAR-10** stored as image folders:

```
data/cifar10/
  train/<class_name>/*.jpg    # 50,000 images
  test/<class_name>/*.jpg     # 10,000 images
```

**Oxford-IIIT Pet** images and trimaps, each in a flat folder:

```
data/oxford_pets/
  Pets/*.jpg     # images
  Mask/*.png     # annotations/trimaps
```

Each notebook has a `DATA_ROOT` variable in its first cells; point it to your copy.

## Running

```
pip install -r requirements.txt
jupyter lab
```

All training cells use `'cuda'`, so an NVIDIA GPU with CUDA-enabled PyTorch is required.

## Known limitations

These are recorded here so the numbers are not over-read:

- **Segmentation, "Test" lines.** The cells that evaluate on the test set accumulate test metrics but never compute
  them; the printed line shows the last training loss and the last *validation* mIoU / Dice. All segmentation numbers
  above are therefore validation results; no genuine test score exists.
- **Segmentation, Dice.** `Dice (excl. BG)` uses `ignore_index=0`, but in the mask encoding used (0 = pet,
  1 = background, 2 = border) class 0 is the pet class, so this Dice excludes *pet*, not background.
- **Segmentation, reproducibility.** `random_split` is unseeded and the notebook was executed out of order across
  several kernel sessions, so it is not certain which resolution / batch size produced the stored outputs of the last
  section; re-running gives somewhat different numbers. `dice_loss` is defined but not used in any run.
- **Classification, test set.** The test set was evaluated after every experiment, so it was also used (informally) to
  compare experiments; the reported test accuracy is slightly optimistic as a final estimate.
- **Classification, unfinished runs.** The frozen-layers run (#9) crashed the kernel after 20 logged epochs, and the
  DenseNet run (#10) was never completed. The cells are kept, clearly marked, without results.

## Changes relative to the original notebooks

Comments, section headings and result tables were added and everything was translated to English. Beyond that:

- Hard-coded local paths were replaced by a `DATA_ROOT` variable.
- Cells that only dumped every file path (segmentation) and empty cells were removed; crashed / stale outputs and
  outputs containing local paths were cleared.
- Segmentation: `epochs = 30` is now defined before the scheduler that uses it (it previously relied on leftover
  kernel state and would fail on a fresh run).
- Classification: the loss-curve cell of the frozen-layers run plots `np.arange(0, epochs)` instead of
  `np.arange(0, 100)`, matching its 20 epochs.

The training code, models, hyperparameters and stored results are otherwise unchanged.

## Authorship and AI assistance

The experiments, models and code were written by author. [Claude](https://claude.com) (an AI assistant by
Anthropic) was used to write the explanatory comments and section headings, to make the notebooks easier to navigate,
and to help prepare this README and the repository.

## License

MIT
