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
    git clone https://github.com/IISc-UMC-203/Normalizing-Flows.git
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

normalizing-flows
  - .gitignore
  - LICENSE
  - README.md
  - requirements.txt
  - 2D.ipynb
      -  Contains all the needed code related to applying Normalizing flow model on 2D toy datasets including video generation
  - MNIST.ipynb
      - Contains all the needed code related to applying Normalizing flow model on MNIST dataset including video generation
  - CIFAR-10.ipynb
      -  Contains all the needed code related to applying Normalizing flow model on CIFAR-10 dataset.
  - data
      - (datasets downloaded here automatically, typically gitignored)
  - results
      - realnvp_2d
          - frames
              - moons         # Example dataset subdir
                  - ... (frame_*.png files)
          - models
              - (realnvp_2d_moons_final.pth) # Example saved model
      - checkpoints
          - (realnvp_mnist_best.pth)       # Example checkpoint
      - mnist_visuals
          - (mnist_generated_digit_7.png)  # Example plot output
          - (mnist_interpolation_10_to_20.mp4) # Example video output


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
