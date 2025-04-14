# scripts/train_realnvp_2d.py
import torch
import argparse
import os
import math
from torch.utils.data import DataLoader
from tqdm import tqdm # Use standard tqdm here

# Project local imports (assuming this script is run from repository root
# or PYTHONPATH includes the 'src' directory)
from src.models_2d import RealNVP_2D
from src.losses import nll_loss_2d
from src.utils import load_2d_data, Dataset2D, get_device
from src.visualize import save_animation_frame

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, num_epochs):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} Training", leave=False)
    for batch_data in pbar:
        x = batch_data.to(device)
        optimizer.zero_grad()
        z, log_det = model(x)
        loss = loss_fn(z, log_det)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\nNaN/Inf loss detected at Epoch {epoch}! Stopping training.")
            return None # Indicate failure

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    """Main training script."""
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading dataset: {args.dataset}")
    data_np, _ = load_2d_data(args.dataset, args.n_samples)
    dataset = Dataset2D(data_np)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    viz_data = dataset.data.to(device) # All data for consistent visualization

    # --- Model ---
    print("Building model...")
    model = RealNVP_2D(
        dim=2,
        num_coupling_layers=args.num_coupling,
        hidden_units=args.hidden_units,
        depth=args.mlp_depth
    ).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Frame Directory ---
    frame_base_dir = os.path.join(args.output_dir, "frames")
    dataset_frame_dir = os.path.join(frame_base_dir, args.dataset)
    os.makedirs(dataset_frame_dir, exist_ok=True)
    print(f"Animation frames will be saved to: {dataset_frame_dir}")

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, nll_loss_2d, device, epoch, args.epochs)

        if avg_loss is None: # Handle NaN/Inf loss
             print("Training stopped due to invalid loss value.")
             break

        log_msg = f"Epoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.4f}"

        # Save animation frame
        if args.save_interval > 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            save_animation_frame(model, viz_data, args.dataset.capitalize(), epoch, device,
                                 dataset_frame_dir, grid_res_vis=args.grid_res)
            log_msg += " | Frame Saved"

        print(log_msg)

    print("--- Training Finished ---")

    # --- Optional: Save Final Model ---
    if args.save_model:
        model_save_dir = os.path.join(args.output_dir, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, f"realnvp_2d_{args.dataset}_final.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RealNVP on 2D Toy Datasets with Animation")

    # Dataset args
    parser.add_argument('--dataset', type=str, required=True, choices=['moons', 'circles'], help='Dataset name')
    parser.add_argument('--n_samples', type=int, default=2000, help='Number of data samples')

    # Model args
    parser.add_argument('--num_coupling', type=int, default=8, help='Number of coupling layers')
    parser.add_argument('--hidden_units', type=int, default=128, help='Hidden units in MLPs')
    parser.add_argument('--mlp_depth', type=int, default=3, help='Depth of MLPs in coupling layers')

    # Training args
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')

    # Output & Visualization args
    parser.add_argument('--output_dir', type=str, default='results/realnvp_2d', help='Base directory for outputs (frames, models)')
    parser.add_argument('--save_interval', type=int, default=5, help='Save animation frame every N epochs (0 to disable)')
    parser.add_argument('--grid_res', type=int, default=20, help='Resolution of grid lines for visualization')
    parser.add_argument('--save_model', action='store_true', help='Save the final trained model')

    args = parser.parse_args()
    main(args)