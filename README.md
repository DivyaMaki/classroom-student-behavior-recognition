# Classroom Student Behavior Recognition

Research code and trained checkpoints for a two-stream system that detects students in classroom video and recognizes their behavior over time. The spatial branch uses PACR-DETR; the temporal branch uses a Video Swin-based Temporal Segment Transformer (TST) for multi-label action classification.

The accompanying research paper describes evaluation on the Student Action Video (SAV) dataset. It reports **71.49% multi-label mAP** for behavior recognition and **89.14% AP@50** for student detection. These are results reported in the paper and are not an independent reproduction claim for this repository.

## Repository contents

| Path | Description |
| --- | --- |
| `Code/pacr-bounding.ipynb` | PACR-DETR person-detection training workflow for SAV bounding boxes. |
| `Code/TST code.py` | Video Swin/TST model, training loop, and multi-label evaluation. |
| `Code/inference.ipynb` | Combined PACR + TST inference workflow and Gradio interface. |
| `models/pacr_best.pth` | Trained PACR-DETR detection checkpoint. |
| `models/tst_best.pth` | Trained TST behavior-classification checkpoint. |

The paper PDF and project journal are not included. Dataset files, credentials, and environment configuration are also excluded.

## Method

1. **Student detection:** PACR-DETR uses position-aware channel residual attention and multi-scale features to localize students in crowded classroom scenes.
2. **Behavior recognition:** TST processes per-student video segments with 3D shifted-window attention to predict multiple actions over time.
3. **Class imbalance:** The training implementation includes asymmetric loss, class weighting, and related strategies for long-tailed action labels.

The paper describes experiments on 4,324 SAV clips across 15 action classes. Dataset access and use are subject to the dataset provider's terms.

## Requirements

The implementation uses Python and the following main packages:

- PyTorch and torchvision
- NumPy, pandas, Pillow, and SciPy
- einops and tqdm
- OpenCV (`cv2`)
- Matplotlib
- Jupyter/IPython
- Gradio (for the notebook interface)

Install a PyTorch/torchvision build appropriate for your operating system and CUDA setup, then install the remaining packages:

```bash
python -m pip install numpy pandas pillow scipy einops tqdm opencv-python matplotlib jupyter gradio
```

## Data and checkpoints

The notebooks and training script currently use Kaggle-specific example paths. To run the workflows:

1. Obtain the SAV video frames and annotation CSVs from an authorized source.
2. Update the dataset and annotation paths in the relevant notebook or `Config` class.
3. Update checkpoint and pretrained-weight paths to the files available in your environment.
4. For the combined inference workflow, make the `pacr_detr` implementation referenced by `Code/inference.ipynb` available and configure its path.

The checkpoints are stored with Git LFS. Install Git LFS before cloning so the checkpoint files are downloaded rather than their small pointer files:

```bash
git lfs install
git clone https://github.com/DivyaMaki/classroom-student-behavior-recognition.git
```

The checkpoints do not include the SAV dataset. Checkpoint compatibility depends on using the corresponding model architecture and configuration.

## Running

### PACR-DETR detection

Open `Code/pacr-bounding.ipynb` in Jupyter or Kaggle, configure its data and output paths, and run the notebook cells to train or evaluate the detector.

### TST training

Update the paths and training settings in `Config` in `Code/TST code.py`. The `train(config)` call in the script's `__main__` block is commented out by default; enable it when ready to start training:

```bash
python "Code/TST code.py"
```

Training requires the SAV data and model weights described above. A CUDA-capable GPU is recommended for the video-transformer workload.

### Inference and interface

Open `Code/inference.ipynb`, configure the data and model paths, and run its cells in order. The notebook includes inference code and a Gradio interface; it is not a standalone command-line application.

## Reproducibility and responsible use

Results depend on the dataset split, pretrained weights, runtime, and configuration. Record these details when reproducing experiments. Classroom video may contain sensitive personal information: use recordings only with appropriate authorization, protect identities and data, and do not use model predictions as the sole basis for consequential decisions about students.

## License

No license is currently specified. Contact the repository owner before reusing or redistributing the code or checkpoints.
