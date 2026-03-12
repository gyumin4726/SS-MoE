# SS-MoE

## Environment Setup

This section provides step-by-step instructions for setting up the environment required to run this project.

### Prerequisites

- Conda (or Miniconda/Mambaforge)
- CUDA 11.3 compatible GPU (for PyTorch installation)

### Installation Steps

1. **Create a new conda environment:**
   ```bash
   conda create --name mambafscil python=3.10 -y
   ```

2. **Activate the conda environment:**
   ```bash
   conda activate mambafscil
   ```

3. **Install PyTorch with CUDA 11.3 support:**
   ```bash
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

4. **Install OpenMMLab MIM and MMCV:**
   ```bash
   pip install -U openmim
   mim install mmcv-full==1.7.0
   ```

5. **Install additional Python packages:**
   ```bash
   pip install opencv-python matplotlib einops rope timm==0.6.12 scikit-learn==1.1.3 yapf==0.40.1
   ```

6. **Clone and install Mamba:**
   ```bash
   git clone https://github.com/state-spaces/mamba.git
   cd mamba
   git checkout v1.2.0.post1
   pip install .
   cd ..
   ```

7. **Clone the FSCIL repository:**
   ```bash
   git clone https://github.com/gyumin4726/SS-MoE.git
   cd FSCIL
   ```

8. **Clone the VMamba repository:**
   ```bash
   git clone https://github.com/MzeroMiko/VMamba.git
   ```

9. **Create data directory:**
   ```bash
   mkdir ./data
   ```

10. **Install remaining dependencies:**
    ```bash
    pip install --upgrade transformers==4.38.2
    pip install numpy==1.24.4
    pip install fvcore
    ```

## Pretrained Weights

Our implementation uses the official pretrained VMamba checkpoints.

Download the following checkpoints from the official VMamba repository:

https://github.com/MzeroMiko/VMamba

Required checkpoints:

- `vssm_base_0229_ckpt_epoch_237.pth`
- `vssm_small_0229_ckpt_epoch_222.pth`
- `vssm1_tiny_0230s_ckpt_epoch_264.pth`

After downloading, place the checkpoints directly inside the `SS-MoE` directory.

Expected structure:

- `SS-MoE/vssm_base_0229_ckpt_epoch_237.pth`
- `SS-MoE/vssm_small_0229_ckpt_epoch_222.pth`
- `SS-MoE/vssm1_tiny_0230s_ckpt_epoch_264.pth`

## Dataset Preparation

All datasets should be placed under the `data/` directory.

### CIFAR-100

CIFAR-100 will be automatically downloaded during the first run.  
No manual download is required.

### miniImageNet

Download miniImageNet from the following link:

https://cseweb.ucsd.edu/~weijian/static/datasets/mini-ImageNet/MiniImagenet.tar.gz

After downloading, extract the dataset into:

- `SS-MoE/data/miniimagenet`

### CUB-200-2011

Download CUB-200-2011 from the official dataset page:

https://www.vision.caltech.edu/datasets/cub_200_2011/

After downloading, extract the dataset into:

- `SS-MoE/data/CUB_200_2011`

## Training & Evaluation

Run the training script corresponding to each dataset:

- `sh train_cub.sh`
- `sh train_cifar.sh`
- `sh train_miniimagenet.sh`

Each script first performs training on the **base session**, followed by automatic evaluation across all **incremental sessions** according to the FSCIL protocol.

---

## Configuration

The main experimental settings can be modified in:

- `configs/<dataset>/<dataset>_base.py`
- `configs/<dataset>/<dataset>_inc.py`

These configuration files control the training setup for the **base session** and **incremental sessions**, respectively.

Key configurable options include:

- **Backbone model**
  - `backbone.model_name`
  - `pretrained_path`
  - Select the VMamba backbone variant (Tiny / Small / Base).

- **Number of MoE experts**
  - `neck.num_experts`
  - Specifies the number of experts in the SS-MoE expert pool.

- **Top-k routing**
  - `neck.top_k`
  - `neck.eval_top_k`
  - Determines how many experts are activated during routing.
