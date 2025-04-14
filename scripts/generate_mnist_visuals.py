# scripts/generate_mnist_visuals.py
import torch
import argparse
import os
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader # Needed for dataset loading

# Project local imports
from src.models_img import RealNVP_MultiScale # Assuming this model works for MNIST
from src.utils import get_device
from src.visualize import (generate_digit_samples,
                           interpolate_images_to_video,
                           generate_interpolation_grid_video)

def load_model_and_dataset(args):
    """Loads the trained model and dataset."""
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset (only need test set usually for visuals)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    input_shape = (1, 28, 28) # Hardcode for MNIST

    # Load model definition (ensure parameters match the checkpoint)
    model = RealNVP_MultiScale(
        input_shape=input_shape,
        num_coupling_multiscale=args.num_coupling_multi, # Use args from training
        num_coupling_final=args.num_coupling_final,
        planes=args.planes
    ).to(device)

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)

    # Handle potential differences in saved args vs current args
    saved_args = checkpoint_data.get('args', None)
    if saved_args:
        # You might want to compare saved_args with current args or load them
        print("Checkpoint was trained with args:", saved_args)
        # For simplicity, we assume the current model architecture args match the checkpoint
        # If mismatch, model loading might fail or behave unexpectedly.

    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")

    return model, dataset, device


def main(args):
    model, dataset, device = load_model_and_dataset(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Task 1: Generate Samples for Specific Digits ---
    if args.generate_digits:
        print("\n--- Generating Samples for Specific Digits ---")
        for digit in args.generate_digits:
            output_file = os.path.join(args.output_dir, f"mnist_generated_digit_{digit}.png")
            generate_digit_samples(
                model=model,
                dataset=dataset, # Use test dataset for references
                digit=digit,
                num_to_generate=args.num_generate,
                num_ref_samples=args.num_ref,
                noise_std=args.noise_std,
                batch_size=args.gen_batch_size, # Use a generation batch size
                device=device,
                output_file=output_file
            )

    # --- Task 2: Interpolate between two specific images ---
    if args.interpolate_pair:
        if len(args.interpolate_pair) != 2:
            print("Error: --interpolate_pair requires exactly two indices (e.g., --interpolate_pair 10 20)")
        else:
            idx1, idx2 = args.interpolate_pair
            print(f"\n--- Generating Interpolation Video: Index {idx1} to {idx2} ---")
            output_file = os.path.join(args.output_dir, f"mnist_interpolation_{idx1}_to_{idx2}.mp4")
            interpolate_images_to_video(
                model=model,
                dataset=dataset,
                source_idx=idx1,
                target_idx=idx2,
                num_steps=args.interp_steps,
                filename=output_file,
                fps=args.fps,
                device=device,
                batch_size_override=args.gen_batch_size
            )

    # --- Task 3: Generate Interpolation Grid Video ---
    if args.interpolate_grid:
        print("\n--- Generating Interpolation Grid Video ---")
        num_pairs = args.grid_rows * args.grid_rows
        # Select random pairs from test set
        image_pairs = []
        available_indices = list(range(len(dataset)))
        if len(available_indices) < num_pairs * 2:
             print(f"Warning: Not enough unique samples in dataset ({len(available_indices)}) for {num_pairs} pairs. Reducing pairs.")
             num_pairs = len(available_indices) // 2
             args.grid_rows = int(np.sqrt(num_pairs)) # Adjust grid size down
             num_pairs = args.grid_rows * args.grid_rows # Recalculate num_pairs
             if num_pairs == 0:
                 print("Cannot generate grid video.")
                 return

        random.shuffle(available_indices)
        for _ in range(num_pairs):
             idx1 = available_indices.pop()
             idx2 = available_indices.pop()
             image_pairs.append((idx1, idx2))
        print("Selected pairs:", image_pairs)

        output_file = os.path.join(args.output_dir, f"mnist_interpolation_grid_{args.grid_rows}x{args.grid_rows}.mp4")
        generate_interpolation_grid_video(
            model=model,
            dataset=dataset,
            pairs=image_pairs,
            num_steps=args.interp_steps,
            grid_rows=args.grid_rows,
            filename=output_file,
            fps=args.fps,
            device=device
        )

    print("\n--- MNIST Visualization Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualizations for MNIST RealNVP")

    # Model & Data Args
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing MNIST data')
    # Include model architecture args used during training to load the correct model structure
    # These should ideally match the args saved in the checkpoint, but required here for instantiation
    parser.add_argument('--num_coupling_multi', type=int, default=12, help='Number of multiscale coupling layers used for the loaded model')
    parser.add_argument('--num_coupling_final', type=int, default=4, help='Number of final coupling layers used for the loaded model')
    parser.add_argument('--planes', type=int, default=64, help='Initial number of planes used for the loaded model')

    # Task 1: Generate Digit Samples
    parser.add_argument('--generate_digits', type=int, nargs='+', default=None, help='List of digits to generate samples for (e.g., --generate_digits 3 7 9)')
    parser.add_argument('--num_generate', type=int, default=32, help='Number of samples to generate per digit')
    parser.add_argument('--num_ref', type=int, default=100, help='Number of reference images to calculate mean latent')
    parser.add_argument('--noise_std', type=float, default=0.6, help='Std deviation of noise added to mean latent for generation')

    # Task 2: Interpolate Specific Pair
    parser.add_argument('--interpolate_pair', type=int, nargs=2, default=None, metavar=('IDX1', 'IDX2'), help='Generate video interpolating between two specific dataset indices')

    # Task 3: Interpolate Grid
    parser.add_argument('--interpolate_grid', action='store_true', help='Generate video interpolating a grid of random pairs')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows/columns for the interpolation grid video')

    # Common Visualization Args
    parser.add_argument('--interp_steps', type=int, default=60, help='Number of interpolation steps (frames)')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for output videos')
    parser.add_argument('--gen_batch_size', type=int, default=64, help='Batch size for generating images during visualization')
    parser.add_argument('--output_dir', type=str, default='results/mnist_visuals', help='Directory to save plots and videos')

    args = parser.parse_args()
    main(args)