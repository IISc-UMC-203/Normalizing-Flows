# scripts/train_realnvp_img.py
import torch
import argparse
import os
import math
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Project local imports
from src.models_img import RealNVP_MultiScale
from src.losses import nll_loss_img
from src.utils import pre_process_img, log_preprocessing_grad, get_device # Assuming log_preprocessing_grad exists

def train_epoch_img(model, dataloader, optimizer, loss_fn, device, epoch, num_epochs, batch_size, num_pixels, alpha_logit):
    """Trains the image model for one epoch."""
    model.train() # Ensure training mode
    total_loss = 0.0
    total_bpd = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} Training", leave=False)
    for batch_idx, (X, _) in enumerate(pbar):
        X = X.to(device)
        # Preprocess
        X_pre, log_det_preprocess = pre_process_img(X, add_noise=True, alpha=alpha_logit)

        optimizer.zero_grad()
        # Forward pass
        y, s_log_det, norm_log_det, scale_params = model(X_pre)

        # Calculate loss
        # Note: log_det_preprocess needs to be passed to the loss now
        loss, comps = loss_fn(y, s_log_det, norm_log_det, scale_params, batch_size, log_det_preprocess)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\nNaN/Inf loss detected at Epoch {epoch}, Batch {batch_idx}! Stopping.")
            for name, p in model.named_parameters():
                 if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                     print(f"NaN/Inf gradient in parameter: {name}")
            return None, None # Indicate failure

        loss.backward()

        # Optional: Gradient clipping can help stability
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # --- Calculate BPD for logging ---
        # Loss already includes regularization, subtract it for pure NLL
        nll_term = loss.item() - (comps[3].item() / batch_size) # comps[3] is total reg
        # The NLL calculated is -log p(x) / batch_size. Convert to bits per dimension.
        # bpd = NLL_total / (N * D * log(2))
        # bpd = (nll_term * batch_size) / (batch_size * num_pixels * log(2))
        bpd = nll_term / (num_pixels * np.log(2.0))
        total_bpd += bpd
        total_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.3f}", "BPD": f"{bpd:.3f}"})

    avg_loss = total_loss / len(dataloader)
    avg_bpd = total_bpd / len(dataloader)
    return avg_loss, avg_bpd


def test_epoch_img(model, dataloader, loss_fn, device, batch_size, num_pixels, alpha_logit):
    """Evaluates the image model on the test set."""
    model.validate() # Use validation mode for BN stats
    test_loss = 0.0
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for X, _ in pbar:
            X = X.to(device)
            X_pre, log_det_preprocess = pre_process_img(X, add_noise=False, alpha=alpha_logit) # No noise for eval
            y, s_log_det, norm_log_det, scale_params = model(X_pre)
            loss, _ = loss_fn(y, s_log_det, norm_log_det, scale_params, batch_size, log_det_preprocess)
            test_loss += loss.item() # Accumulate NLL + Reg / batch_size
            # BPD Calculation (similar to train loop)
            nll_term = loss.item() - (_.item() / batch_size if len(_) > 3 else 0) # comps not available, approximate reg if needed or ignore for pure NLL BPD
            bpd = nll_term / (num_pixels * np.log(2.0))
            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "BPD": f"{bpd:.3f}"})

    model.train() # Set back to train mode
    test_loss /= len(dataloader)
    test_bpd = test_loss / (num_pixels * np.log(2.0)) # Avg BPD over batches
    print(f"Validation Avg Loss: {test_loss:.3f}, Avg BPD: {test_bpd:.3f}")
    return test_loss, test_bpd


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)
    elif args.dataset == 'cifar10':
        # Add data augmentation for CIFAR-10? (Original code had RandomHorizontalFlip)
        train_transform = transforms.Compose([
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()
             ])
        test_transform = transforms.ToTensor()
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
        input_shape = (3, 32, 32)
    else:
        raise ValueError("Invalid dataset choice")

    num_pixels = np.prod(input_shape)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    print("Building model...")
    model = RealNVP_MultiScale(
        input_shape=input_shape,
        num_coupling_multiscale=args.num_coupling_multi,
        num_coupling_final=args.num_coupling_final,
        planes=args.planes
    ).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Added weight decay
    # Scheduler similar to CIFAR notebook
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)


    # --- Checkpoint Loading ---
    start_epoch = 1
    best_val_bpd = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_val_bpd = checkpoint['best_val_bpd']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best_val_bpd {best_val_bpd:.3f})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_bpd = train_epoch_img(
            model, train_loader, optimizer, nll_loss_img, device, epoch, args.epochs,
            args.batch_size, num_pixels, args.alpha_logit
        )

        if train_bpd is None: # Handle NaN/Inf
            print("Stopping due to invalid training loss/BPD.")
            break

        val_loss, val_bpd = test_epoch_img(
            model, test_loader, nll_loss_img, device, args.batch_size, num_pixels, args.alpha_logit
        )

        scheduler.step() # Step the scheduler

        is_best = val_bpd < best_val_bpd
        if is_best:
            best_val_bpd = val_bpd
            print(f"*** New best validation BPD: {best_val_bpd:.3f} ***")
            save_path = os.path.join(args.checkpoint_dir, f"realnvp_{args.dataset}_best.pth")
        else:
             save_path = os.path.join(args.checkpoint_dir, f"realnvp_{args.dataset}_epoch_{epoch}.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_bpd': val_bpd,
            'best_val_bpd': best_val_bpd,
            'args': args # Save args for reproducibility
        }, save_path)
        print(f"Checkpoint saved to {save_path} (Best BPD: {best_val_bpd:.3f})")


    print("--- Training Finished ---")
    print(f"Best Validation BPD achieved: {best_val_bpd:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RealNVP MultiScale on MNIST/CIFAR-10")

    # Data args
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing data')
    parser.add_argument('--alpha_logit', type=float, default=0.05, help='Alpha parameter for logit preprocessing')

    # Model args
    parser.add_argument('--num_coupling_multi', type=int, default=12, help='Number of multiscale coupling layers (MNIST default: 12)') # Default based on MNIST notebook
    parser.add_argument('--num_coupling_final', type=int, default=4, help='Number of final coupling layers (MNIST default: 4)')
    parser.add_argument('--planes', type=int, default=64, help='Initial number of planes in conv layers')

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') # Default based on MNIST notebook
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate') # Default based on MNIST notebook
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size') # Default based on MNIST notebook
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Adam weight decay (L2 regularization on weights)') # From loss_fn reg
    parser.add_argument('--lr_step', type=int, default=3, help='Step size for LR scheduler (epochs)') # Based on CIFAR scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Gamma factor for LR scheduler') # Based on CIFAR scheduler
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')

    # Checkpointing args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Adjust defaults based on dataset if not specified (optional)
    if args.dataset == 'cifar10':
        if args.num_coupling_multi == 12: args.num_coupling_multi = 18 # Default from CIFAR notebook
        if args.batch_size == 50: args.batch_size = 20 # Default from CIFAR notebook
        if args.epochs == 10: args.epochs = 15 # Default from CIFAR notebook

    print("Effective Args:", args)
    main(args)