# Classroom Student Behavior Recognition

A research implementation of a two-stream system for detecting students in classroom video and recognizing their behavior over time. The project combines PACR-DETR for spatial person detection with a Video Swin-based Temporal Segment Transformer (TST) for multi-label action classification.

The accompanying paper, **“Spatiotemporal Framework for Automated Multi-label Student Behavior Recognition in Classrooms,”** describes evaluation on the Student Action Video (SAV) dataset. The paper reports 71.49% multi-label mAP for behavior recognition and 89.14% AP@50 for person detection. These are reported paper results, not results independently reproduced by this repository.

## Project contents

| File | Description |
| --- | --- |
| `Code/pacr-bounding.ipynb` | PACR-DETR person-detection training implementation for SAV bounding boxes. |
| `Code/TST code.py` | Video Swin/TST model and training implementation, including multi-label losses and evaluation. |
| `Code/inference.ipynb` | PACR + TST workflow, inference utilities, and a Gradio interface. |

The paper PDF and any journal material are intentionally not included in this repository.

## Method overview

1. **Spatial detection:** PACR-DETR detects student bounding boxes, using position-aware channel residual attention and multi-scale features.
2. **Temporal recognition:** The TST classifies per-student video segments with 3D shifted-window attention and multi-label outputs.
3. **Class imbalance:** The implementation includes asymmetric loss, class weighting, and related training strategies for the long-tailed action distribution.

The paper describes experiments on 4,324 SAV clips with 15 action classes. The dataset, pretrained weights, and trained checkpoints are not included here; obtain them from their authorized sources and follow their respective terms.

## Requirements

The code uses Python, PyTorch, and the following main libraries:

- NumPy, pandas, Pillow
- SciPy
- einops and tqdm
- OpenCV (`cv2`)
- Matplotlib
- Gradio (for the notebook interface)
- Jupyter/IPython (for notebooks)

Install a PyTorch/torchvision build appropriate for your operating system and CUDA setup, then install the other packages:

```bash
python -m pip install numpy pandas pillow scipy einops tqdm opencv-python matplotlib gradio jupyter
```

## Data and model setup

The training and inference code currently contains Kaggle-specific paths. Before running it:

1. Make the SAV frames and annotation CSVs available in your environment.
2. Obtain the PACR detector checkpoint and Video Swin pretrained weights required by the TST workflow.
3. Update the dataset, annotation, checkpoint, pretrained-weight, and output paths in the relevant notebook or `Config` class.
4. Ensure the `pacr_detr` implementation referenced by `inference.ipynb` is available at the configured path.

The data and weight paths in the code are examples tied to a Kaggle setup, not portable paths. Do not commit datasets, checkpoints, or credentials to this repository.

## Running

### PACR-DETR detection

Open `Code/pacr-bounding.ipynb` in Jupyter or Kaggle, update its `Config` paths, and execute the notebook cells to train/evaluate the person detector.

### TST training

Update the paths and training settings in `Config` in `Code/TST code.py`. Its `__main__` block currently leaves the `train(config)` call commented out; enable that call when you are ready to start training:

```bash
python "Code/TST code.py"
```

Training requires the SAV data and model weights described above. A CUDA-capable GPU is recommended for the video transformer workload.

### Inference and interface

Open `Code/inference.ipynb`, configure the data/model paths and `pacr_detr` dependency, then run its cells in order. The notebook includes inference code and a Gradio interface; it is not a standalone command-line application.

## Reproducibility and responsible use

The reported metrics depend on the dataset split, pretrained weights, runtime, and configuration. Record those details when reproducing experiments. Classroom video can contain sensitive personal information: use data only with appropriate authorization, protect identities and recordings, and do not use model predictions as the sole basis for consequential decisions about students.
