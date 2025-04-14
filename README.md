# Normalizing-Flows
This repository primarily consists of:

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
4.  **Implementations of Invertible Neural Networks (INNs)** \[[2]\] for Inverse Problems:
    *   Application of the INN framework (using RealNVP components) to solve the Gaussian Mixture Model (GMM) conditional generation task.
    *   Application of the INN framework to solve the Inverse Kinematics (IK) problem, demonstrating posterior sampling of valid configurations.

The project aims to provide a practical understanding of normalizing flows and demonstrate their utility in analyzing complex, potentially multi-modal inverse problems.
## Features

*   **RealNVP Implementation:** Core RealNVP model with affine coupling layers.
*   **Multi-Scale Architecture:** Adapted RealNVP for image data (MNIST, CIFAR-10) using channel padding, squeeze operations, and factor-out mechanisms.
*   **2D Flow Visualization:** Scripts to train RealNVP on 2D datasets and generate animations of the data-to-latent transformation.
*   **MNIST Latent Space Interpolation:** Script/notebook demonstrating smooth interpolation between digit classes in the learned latent space.
*   **INN for GMM:** Implementation tackling the GMM inverse problem, demonstrating conditional generation and multi-modal posterior estimation.
*   **INN for Inverse Kinematics:** Implementation solving the IK problem, visualizing multiple valid arm configurations for a given target.
*   **Training & Evaluation:** Scripts for training models (NLL loss for RealNVP, combined losses for INN) and evaluating BPD on image datasets.

## Requirements

*   Python 3.8+
*   PyTorch (version 1.x or 2.x recommended)
*   Torchvision
*   NumPy
*   Matplotlib
*   Scikit-learn
*   ImageIO (for creating GIFs)
*   (Optional but recommended) CUDA Toolkit for GPU acceleration
## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace with your repository URL
    cd normalizing-flows
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

*   **2D Datasets:** Generated on-the-fly using `sklearn.datasets`.
*   **MNIST & CIFAR-10:** Downloaded automatically by `torchvision.datasets` upon the first run of relevant scripts if not found locally.

## Usage Examples

*(Note: Adjust script names and arguments based on your actual file structure)*

1.  **Train RealNVP on 2D Moons dataset and generate animation frames:**
    ```bash
    python scripts/train_realnvp_2d.py --dataset moons --epochs 3000 --save_interval 10 --output_dir results/animations/moons
    # After training, compile frames into a GIF (e.g., using imageio or external tool)
    ```

2.  **Train RealNVP on MNIST:**
    ```bash
    python scripts/train_realnvp_mnist.py --epochs 100 --lr 1e-3 --batch_size 128 --save_path results/models/realnvp_mnist.pth
    ```

3.  **Evaluate BPD for trained MNIST model:**
    ```bash
    python scripts/evaluate_bpd.py --model RealNVP --dataset MNIST --checkpoint results/models/realnvp_mnist.pth
    ```

4.  **Generate MNIST Latent Space Interpolation Visualization:**
    ```bash
    # Example using a script:
    python scripts/generate_mnist_interpolation.py --checkpoint results/models/realnvp_mnist.pth --idx1 1 --idx2 7 --steps 10 --output results/animations/mnist_interp_1_7.gif
    ```
## Project Structure:
normalizing-flows
├── .gitignore # Files ignored by git
├── LICENSE # Project license
├── README.md # This file
├── requirements.txt # Python dependencies
│
├── src # Core source code (models, layers, losses, utils)
├── scripts # Runnable scripts for training, evaluation, generation
├── notebooks # Jupyter notebooks for demos and visualization
├── data # Data loading instructionsscripts (if needed)
└── results # Default directory for outputs (models, plots, animations)

*   **RealNVP BPD:** Achieved **1.59 BPD** on MNIST and **3.78 BPD** on CIFAR-10, demonstrating successful implementation (though further training could potentially improve scores, limited by compute).
*   **Visualizations:** Successfully generated animations showing 2D flow dynamics and smooth semantic interpolations in MNIST's latent space.

## Note : This Repo will give you a strong foundation of Normalizing flows before understanding the implementation of the inverse kinematics problem using Invertible Neural Networks
