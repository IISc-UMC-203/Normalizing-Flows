# Normalizing-Flows
This Repo primarily consists of:

1.  **Implementations of the RealNVP Normalizing Flow architecture** \[[1]\]:
    *   Core affine coupling layers and model structure.
    *   Adaptations for 2D toy datasets (Moons, Circles).
    *   Multi-scale architecture for image datasets (MNIST, CIFAR-10).
2.  **Training and Evaluation Scripts:**
    *   Scripts to train RealNVP models using Negative Log-Likelihood (NLL).
    *   Scripts to evaluate generative performance via Bits Per Dimension (BPD).
3.  **Visualization Tools:**
    *   Code to generate animations visualizing the 2D data-to-latent space transformation learned by RealNVP.
    *   Code to explore the learned latent space of MNIST via interactive interpolation between digit classes.

The project aims to provide a practical understanding of normalizing flows and demonstrate their utility in analyzing complex, potentially multi-modal inverse problems.

*(Note: The INN implementations for GMM/IK mentioned in the original scope are not included in this version of the README, focusing solely on the RealNVP core and visualizations.)*

## Features

*   **RealNVP Core Implementation (`src/`):** Modular code for affine coupling layers, ResNet blocks, custom BatchNorm, masking functions, and preprocessing utilities.
*   **2D RealNVP Model & Training (`src/models_2d.py`, `scripts/train_realnvp_2d.py`):** Model tailored for 2D data with a script for training and generating visualization frames.
*   **Multi-Scale Image RealNVP Model & Training (`src/models_img.py`, `scripts/train_realnvp_img.py`):** Flexible multi-scale architecture for MNIST/CIFAR-10 with a unified training script.
*   **2D Flow Animation (`src/visualize.py`, `scripts/generate_2d_video.py`):** Utilities to visualize the 2D transformation and a script to compile frames into an MP4 video.
*   **MNIST Latent Space Analysis (`src/visualize.py`, `scripts/generate_mnist_visuals.py`):** Functions and a script to generate samples conditioned on digits and create interpolation videos (single pair or grid).
*   **Command-Line Interface:** Training and generation scripts utilize `argparse` for easy configuration.

## Requirements

*   Python 3.11+
*   PyTorch (version 1.x or 2.x recommended)
*   Torchvision
*   NumPy
*   Matplotlib
*   Scikit-learn
*   ImageIO (for creating GIFs)
*   (Optional but recommended) CUDA Toolkit for GPU acceleration

All required Python packages are listed in `requirements.txt`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace with your repository URL
    cd normalizing-flows
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Consider installing ffmpeg support if video saving fails
    # pip install imageio-ffmpeg
    ```

## Dataset Setup

*   **2D Datasets:** Generated automatically by `scripts/train_realnvp_2d.py`.
*   **MNIST & CIFAR-10:** Downloaded automatically by `torchvision.datasets` via `scripts/train_realnvp_img.py` if not found in the specified `--data_dir` (default: `./data`).

## Project Structure:
normalizing-flows/
├── .gitignore         
├── LICENSE            
├── README.md          # Project overview, setup, usage instructions
├── requirements.txt   # Python dependencies (run: pip install -r requirements.txt)
│
├── src/               # Core source code modules (imported by scripts)
│   ├── layers.py      # Coupling layers, ResNet blocks, custom BatchNorm etc.
│   ├── models_2d.py   # RealNVP model definition for 2D data
│   ├── models_img.py  # RealNVP multi-scale model definition for images
│   ├── losses.py      # NLL Loss function implementations
│   ├── utils.py       # Preprocessing, masking, device setup, other helpers
│   └── visualize.py   # Plotting and video generation functions
│
├── scripts/          
│   ├── train_realnvp_2d.py       # Train 2D models and save animation frames
│   ├── train_realnvp_img.py      # Train MNIST/CIFAR-10 models & save checkpoints
│   ├── generate_2d_video.py      # Compile 2D frames into an MP4 video
│   └── generate_mnist_visuals.py # Generate MNIST samples & interpolation videos
│
├── data/              # Directory for datasets 
│
└── results/           # Default output directory (often gitignored)
    ├── realnvp_2d/    # Outputs from the 2D training script
    │   ├── frames/    # Saved animation frames (in subdirs per dataset, e.g., 'moons')
    │   └── models/    # Saved 2D models (if --save_model is used)
    ├── checkpoints/   # Saved image model checkpoints (.pth files from train_realnvp_img.py)
    └── mnist_visuals/ # Output plots (.png) & videos (.mp4) from MNIST generation script

## Usage Examples


1.  **Train RealNVP on 2D Moons dataset and generate animation frames:**
    ```bash
    python scripts/train_realnvp_2d.py --dataset moons --epochs 300 --save_interval 5 --output_dir results/realnvp_2d --hidden_units 128 --num_coupling 8
    python scripts/generate_2d_video.py --frame_dir results/realnvp_2d/frames/moons --output results/realnvp_2d/transformation_moons.mp4 --fps 15
    ```
 
2.  **Train RealNVP on MNIST:**
    ```bash
    python scripts/train_realnvp_img.py --dataset mnist --epochs 15 --lr 5e-4 --batch_size 64 --num_coupling_multi 12 --num_coupling_final 4 --planes 64 --checkpoint_dir checkpoints --data_dir ./data
    ```
    
3.  **Train RealNVP on CIFAR-10(using different parameters):**
    ```bash
    python scripts/train_realnvp_img.py --dataset cifar10 --epochs 20 --lr 5e-4 --batch_size 32 --num_coupling_multi 18 --num_coupling_final 4 --planes 64 --checkpoint_dir checkpoints --data_dir ./data
    ```
    
4.  **Generate Specific MNIST Digit Samples:**
    ```bash
    python scripts/generate_mnist_visuals.py \
    --checkpoint checkpoints/realnvp_mnist_best.pth \
    --generate_digits 7 \
    --num_generate 32 \
    --num_ref 100 \
    --noise_std 0.6 \
    --output_dir results/mnist_visuals
    ```

5.  **Generate MNIST Interpolation Video Between two specific numbers:**
    ```bash
    python scripts/generate_mnist_visuals.py \
    --checkpoint checkpoints/realnvp_mnist_best.pth \
    --interpolate_pair 10 20 \
    --interp_steps 60 \
    --fps 15 \
    --output_dir results/mnist_visuals
    ```
6.  **Generate MNIST Interpolation Grid Video:**
    ```bash
    python scripts/generate_mnist_visuals.py \
    --checkpoint checkpoints/realnvp_mnist_best.pth \
    --interpolate_grid \
    --grid_rows 4 \
    --interp_steps 120 \
    --fps 15 \
    --output_dir results/mnist_visuals
    ```
## Results Summary: 

*   **RealNVP BPD:** Achieved **1.59 BPD** on MNIST and **3.78 BPD** on CIFAR-10, demonstrating successful implementation (though further training could potentially improve scores, limited by computing resources).
*   **Visualizations:** Successfully generated animations showing 2D flow dynamics and smooth semantic interpolations in MNIST's latent space.

## Note : 

This repository focuses on the implementation and understanding of RealNVP. It serves as a strong foundation before tackling more complex applications like solving inverse problems using Invertible Neural Networks.
## References

[1] Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. arXiv preprint arXiv:1605.08803.

[2] Ardizzone, L., Kruse, J., Wirkert, S., Rahner, D., Pellegrini, E. W., Klessen, R. S., ... & Köthe, U. (2019). Analyzing inverse problems with invertible neural networks. International Conference on Learning Representations (ICLR).

## Contributors

* K.Sai Sandesh Reddy 
