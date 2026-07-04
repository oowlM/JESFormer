# JESFormer
This repository accompanies the manuscript:

**Toward Low-Illumination Fundus Imaging via Structure-Aware Low-Light Image Enhancement**

and provides the official PyTorch implementation of JESFormerfor for joint low-illumination fundus image enhancement and vessel segmentation .

## Repository layout🏘️

- `options/`: experiment configs
- `models/`: training model, losses, schedulers
- `models/archs/`: network architecture
- `data/`: datasets and dataloading
- `metrics/`: evaluation metrics
- `utils/`: shared utilities
- `train.py`: training entrypoint
- `test.py`: inference and optional evaluation entrypoint

## Install🛠️
- Supported Python version: `3.10` (tested).
- Create an environment and install:
```bash
conda create -n jesformer python=3.10
pip install -r requirements.txt
```
- Install the torch version corresponding to your CUDA version.

## Training📚

Edit dataset paths in `options/JESFormer_LLRetina.yaml`, then run:

```bash
python3 train.py --opt options/JESFormer_LLRetina.yaml --gpu_id 0
```

For distributed training:

```bash
torchrun --nproc_per_node=2 train.py --opt options/JESFormer_LLRetina.yaml --launcher pytorch
```
Model checkpoints are written under `experiments/`.

## Test / Inference🌟

```bash
python3 test.py \
  --input_dir path/to/your/inputs \
  --weights path/to/your/weight \
  --result_dir ./results \
  --dataset LLRetina \
  --gpus 0 \
  --target_dir /path/to/gt \
  --test_enhance \
  --no-test_segment
```
If you only want prediction and result saving:

```bash
python3 test.py \
  --input_dir path/to/your/inputs \
  --weights path/to/your/weight \
  --result_dir ./results \
  --dataset LLRetina \
  --gpus 0 \
  --no-test_enhance \
  --no-test_segment
```

## Third-party notices🟢

This repository includes a RETFound wrapper used only for LFPS evaluation. RETFound is used only as a frozen feature extractor for LFPS. Codes related are adapted from the official [RETFound repository](https://github.com/rmaphoh/RETFound).

If you want to evaluate LFPS, download the pretrained weight `RETFound_mae_natureCFP` from the official repository and place it under `weights/RETFound_mae_natureCFP.pth`. 

The included third-party components retain their original copyright and license notices.🌻🌻