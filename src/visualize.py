# src/visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from tqdm.auto import tqdm
import torchvision.utils as vutils
import random
from torch.utils.data import Subset, DataLoader
from .utils import pre_process_img, post_process_img # Relative imports

# --- 2D Visualization ---

def save_animation_frame(model, data_points, dataset_name, epoch, device, frame_dir, num_samples=500, grid_res_vis=20):
    """Saves a 2x2 plot grid showing the 2D flow's state."""
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle(f'{dataset_name} - Epoch {epoch}', fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

    data_points_np = data_points.cpu().numpy()
    z_lim = 3.5 # Limits for latent space visualization
    x_min, x_max = data_points_np[:, 0].min() - 0.5, data_points_np[:, 0].max() + 0.5
    y_min, y_max = data_points_np[:, 1].min() - 0.5, data_points_np[:, 1].max() + 0.5
    x_lim = (x_min, x_max)
    y_lim = (y_min, y_max)
    # --- Top Left: Original Data (Scaled) ---
    ax = axes[0, 0]
    ax.scatter(data_points_np[:, 0], data_points_np[:, 1], s=5, alpha=0.4, c='blue', label='Input Data')
    ax.set_title('Input Data Space (X)', fontsize=10)
    ax.set_xlabel('x1 (scaled)', fontsize=9); ax.set_ylabel('x2 (scaled)', fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # --- Top Right: Transformed Data & Warped Grid (Data -> Latent) ---
    ax = axes[0, 1]
    with torch.no_grad():
        z, _ = model(data_points)
        z_np = z.cpu().numpy()
        x_range = np.linspace(x_lim[0], x_lim[1], grid_res_vis)
        y_range = np.linspace(y_lim[0], y_lim[1], grid_res_vis)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_data = torch.from_numpy(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)).to(device)
        grid_z, _ = model(grid_data)
        grid_z_np = grid_z.cpu().numpy()
        # Plot warped grid lines
        for i in range(grid_res_vis):
            ax.plot(grid_z_np[i*grid_res_vis:(i+1)*grid_res_vis, 0], grid_z_np[i*grid_res_vis:(i+1)*grid_res_vis, 1], color='gray', linewidth=0.3, alpha=0.6)
            ax.plot(grid_z_np[i::grid_res_vis, 0], grid_z_np[i::grid_res_vis, 1], color='gray', linewidth=0.3, alpha=0.6)
        ax.scatter(z_np[:, 0], z_np[:, 1], s=5, alpha=0.4, c='blue', label='f(X)')

    ax.set_title('Latent Space Z (f(X)) & Warped Grid', fontsize=10)
    ax.set_xlabel('z1', fontsize=9); ax.set_ylabel('z2', fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-z_lim, z_lim); ax.set_ylim(-z_lim, z_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # --- Bottom Right: Latent Samples (Base Distribution) ---
    ax = axes[1, 1]
    z_samples = torch.randn(num_samples, model.dim).to(device)
    z_samples_np = z_samples.cpu().numpy()
    ax.scatter(z_samples_np[:, 0], z_samples_np[:, 1], s=5, alpha=0.4, c='red', label='Z ~ N(0,I)')
    ax.set_title('Latent Space Z (Samples)', fontsize=10)
    ax.set_xlabel('z1', fontsize=9); ax.set_ylabel('z2', fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-z_lim, z_lim); ax.set_ylim(-z_lim, z_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # --- Bottom Left: Generated Data & Warped Grid (Latent -> Data) ---
    ax = axes[1, 0]
    with torch.no_grad():
        x_generated = model.inverse(z_samples)
        x_generated_np = x_generated.cpu().numpy()
        z_range = np.linspace(-z_lim, z_lim, grid_res_vis)
        grid_zx, grid_zy = np.meshgrid(z_range, z_range)
        grid_latent = torch.from_numpy(np.stack([grid_zx.ravel(), grid_zy.ravel()], axis=1).astype(np.float32)).to(device)
        grid_x_inv = model.inverse(grid_latent)
        grid_x_inv_np = grid_x_inv.cpu().numpy()
        # Plot warped grid lines
        for i in range(grid_res_vis):
            ax.plot(grid_x_inv_np[i*grid_res_vis:(i+1)*grid_res_vis, 0], grid_x_inv_np[i*grid_res_vis:(i+1)*grid_res_vis, 1], color='gray', linewidth=0.3, alpha=0.6)
            ax.plot(grid_x_inv_np[i::grid_res_vis, 0], grid_x_inv_np[i::grid_res_vis, 1], color='gray', linewidth=0.3, alpha=0.6)
        ax.scatter(x_generated_np[:, 0], x_generated_np[:, 1], s=5, alpha=0.4, c='red', label='X = f$^{-1}$(Z)')

    ax.set_title('Generated Data Space (X = f$^{-1}$(Z))', fontsize=10)
    ax.set_xlabel('x1 (scaled)', fontsize=9); ax.set_ylabel('x2 (scaled)', fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Save the frame
    frame_path = os.path.join(frame_dir, f"{dataset_name}_epoch_{epoch:04d}.png")
    plt.savefig(frame_path, dpi=100) # Adjust dpi if needed
    plt.close(fig) # Close the figure to free memory
    model.train() # Set model back to training mode


def create_video_from_frames(frame_dir, output_path, fps=15):
    """Creates an MP4 video from PNG frames in a directory."""
    frames_data = []
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])

    if not frame_files:
        print(f"No PNG frames found in {frame_dir}. Skipping video creation.")
        return

    print(f"Creating video at {output_path} from {len(frame_files)} frames...")
    try:
        for filename in tqdm(frame_files, desc=f"Loading frames from {os.path.basename(frame_dir)}"):
            frames_data.append(imageio.v3.imread(filename))

        imageio.mimwrite(output_path, frames_data, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        print(f"Video saved successfully to {output_path}")
    except Exception as e:
        print(f"\n-------------------------------------")
        print(f"Error creating MP4 video: {e}")
        print("Ensure imageio and ffmpeg backend are installed correctly.")
        print("Try: pip install imageio-ffmpeg")
        print("-------------------------------------")

# --- MNIST Visualization Helpers ---

def get_image_and_latent(model, dataset, index, device):
    """Gets a single image, its latent vector, and label."""
    model.validate() # Ensure correct mode for forward pass (image->latent)
    img, label = dataset[index]
    img = img.unsqueeze(0) # Add batch dimension
    img_pre, _ = pre_process_img(img, add_noise=False) # Preprocess without noise for inference
    img_pre = img_pre.to(device)
    with torch.no_grad():
        y, _, _, _ = model(img_pre)
    model.train(False) # Switch back to standard eval mode
    model.eval()
    return img, y.cpu(), label # Return original image, latent, label

def get_latents_for_digit(model, dataset, digit, num_samples, batch_size, device):
    """Gets multiple latent vectors for a specific digit."""
    model.validate() # Ensure correct mode for forward pass
    latents = []
    indices = [i for i, (_, label) in enumerate(dataset) if label == digit]
    if not indices:
         raise ValueError(f"No samples found for digit {digit} in the dataset.")
    if len(indices) < num_samples:
        print(f"Warning: Only found {len(indices)} samples for digit {digit}, using all.")
        num_samples = len(indices)

    random_indices = random.sample(indices, num_samples)
    subset = Subset(dataset, random_indices)
    loader = DataLoader(subset, batch_size=batch_size)

    with torch.no_grad():
        for X, _ in loader:
            # Preprocess without noise for consistency
            X_pre, _ = pre_process_img(X, add_noise=False)
            X_pre = X_pre.to(device)
            y, _, _, _ = model(X_pre) # Image -> Latent
            latents.append(y.cpu())

    model.train(False) # Switch back to standard eval mode
    model.eval()
    return torch.cat(latents, dim=0)

def generate_digit_samples(model, dataset, digit, num_to_generate, num_ref_samples, noise_std, batch_size, device, output_file=None):
    """Generates samples resembling a specific digit and optionally saves plot."""
    print(f"\n--- Generating {num_to_generate} samples for digit {digit} ---")
    try:
        ref_latents = get_latents_for_digit(model, dataset, digit, num_ref_samples, batch_size, device)
    except ValueError as e:
        print(e)
        return

    z_mean = ref_latents.mean(dim=0, keepdim=True)
    z_samples = z_mean.repeat(num_to_generate, 1)
    noise = torch.randn_like(z_samples) * noise_std
    z_samples = (z_samples + noise).to(device)

    model.eval() # Ensure inverse pass mode (latent->image)
    generated_images_pre = []
    gen_batch_size = min(batch_size, num_to_generate)
    with torch.no_grad():
         for i in range(0, num_to_generate, gen_batch_size):
             z_batch = z_samples[i:i+gen_batch_size]
             # Model inverse pass expects z, returns pre-processed image
             # Check if model.inverse exists, otherwise assume forward in eval mode is inverse
             if hasattr(model, 'inverse'):
                 y_gen = model.inverse(z_batch)
             else:
                 # Assuming forward pass in eval mode performs latent -> image
                 y_gen_output = model(z_batch)
                 # Handle potential tuple output if forward still returns dets etc.
                 y_gen = y_gen_output[0] if isinstance(y_gen_output, tuple) else y_gen_output
             generated_images_pre.append(y_gen.cpu())

    generated_images_pre = torch.cat(generated_images_pre, dim=0)
    generated_images = post_process_img(generated_images_pre) # Post-process

    # Plotting
    cols = 8
    rows = (num_to_generate + cols - 1) // cols
    figure = plt.figure(figsize=(cols * 1.5, rows * 1.5))
    plt.suptitle(f"Generated Samples for Digit: {digit} (Noise Std: {noise_std:.2f})", fontsize=14)
    for i in range(num_to_generate):
        if i >= generated_images.shape[0]: break # Handle case if less generated than requested
        img = generated_images[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved generated digit plot to {output_file}")
    else:
        plt.show()
    plt.close(figure)


def interpolate_images_to_video(model, dataset, source_idx, target_idx, num_steps, filename, fps, device, batch_size_override=None):
    """Generates an interpolation sequence and saves it as an MP4 video."""
    print(f"\n--- Interpolating from index {source_idx} to {target_idx} into video {filename} ---")
    assert filename.lower().endswith('.mp4'), "Filename must end with .mp4"

    try:
        _, z_src, label_src = get_image_and_latent(model, dataset, source_idx, device)
        _, z_tgt, label_tgt = get_image_and_latent(model, dataset, target_idx, device)
    except IndexError:
        print(f"Error: Index out of bounds (Source: {source_idx}, Target: {target_idx}, Dataset size: {len(dataset)})")
        return

    print(f"Interpolating: {label_src} (idx {source_idx}) -> {label_tgt} (idx {target_idx})")

    alphas = torch.linspace(0, 1, num_steps)
    interpolated_latents = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_src + alpha * z_tgt
        interpolated_latents.append(z_interp)

    interpolated_latents = torch.cat(interpolated_latents, dim=0).to(device)

    # Generate images
    model.eval() # Ensure inverse pass mode
    generated_images_pre = []
    gen_batch_size = batch_size_override if batch_size_override else min(128, num_steps)
    with torch.no_grad():
         for i in range(0, num_steps, gen_batch_size):
             z_batch = interpolated_latents[i:i+gen_batch_size]
             # Assume forward pass in eval mode is latent -> image
             y_gen_output = model(z_batch)
             y_gen = y_gen_output[0] if isinstance(y_gen_output, tuple) else y_gen_output
             generated_images_pre.append(y_gen.cpu())

    generated_images_pre = torch.cat(generated_images_pre, dim=0)
    interpolated_images = post_process_img(generated_images_pre)

    # Create video frames
    frames = []
    for i in range(num_steps):
        img_tensor = interpolated_images[i].squeeze()
        frame_np = np.stack([img_tensor.numpy()] * 3, axis=-1) # Grayscale to RGB
        frame_np_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8) # Clip before converting
        frames.append(frame_np_uint8)

    # Save video
    create_video_from_frames(frames, filename, fps) # Use helper

def generate_interpolation_grid_video(model, dataset, pairs, num_steps, grid_rows, filename, fps, device):
    """Generates an MP4 video showing interpolations for multiple image pairs."""
    print(f"\n--- Generating interpolation grid video for {len(pairs)} pairs ---")
    num_pairs = len(pairs)
    assert num_pairs == grid_rows * grid_rows, "Number of pairs must match grid size (rows*rows)"
    assert filename.lower().endswith('.mp4'), "Filename should end with .mp4"

    z_sources = []
    z_targets = []
    print("Getting latent vectors...")
    valid_pairs = [] # Store pairs for which indices are valid
    for source_idx, target_idx in pairs:
        try:
            _, z_src, _ = get_image_and_latent(model, dataset, source_idx, device)
            _, z_tgt, _ = get_image_and_latent(model, dataset, target_idx, device)
            z_sources.append(z_src)
            z_targets.append(z_tgt)
            valid_pairs.append((source_idx, target_idx))
        except IndexError:
            print(f"Warning: Skipping pair ({source_idx}, {target_idx}) due to invalid index.")

    if not valid_pairs:
        print("Error: No valid image pairs found.")
        return

    z_sources = torch.cat(z_sources, dim=0).to(device)
    z_targets = torch.cat(z_targets, dim=0).to(device)

    model.eval()
    frames = []
    alphas = torch.linspace(0, 1, num_steps)
    print("Generating video frames...")
    for i, alpha in enumerate(tqdm(alphas, desc="Generating frames")):
        z_interp_batch = (1 - alpha) * z_sources + alpha * z_targets
        with torch.no_grad():
            # Assume forward pass in eval mode is latent -> image
            y_gen_output = model(z_interp_batch)
            generated_images_pre = y_gen_output[0] if isinstance(y_gen_output, tuple) else y_gen_output

        interpolated_images = post_process_img(generated_images_pre).cpu()
        grid_img = vutils.make_grid(interpolated_images, nrow=grid_rows, padding=2, normalize=False)
        frame_np = grid_img.permute(1, 2, 0).numpy()
        frame_np_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        frames.append(frame_np_uint8)

    # Save video
    create_video_from_frames(frames, filename, fps) # Use helper

# --- Helper to create video from frames list ---
def create_video_from_frames(frames_list, output_path, fps):
    """Creates an MP4 video from a list of numpy frame arrays."""
    if not frames_list:
        print("No frames provided to create video.")
        return
    print(f"Saving MP4 video to {output_path} ({len(frames_list)} frames)...")
    try:
        imageio.mimwrite(output_path, frames_list, fps=fps, codec='libx264', quality=8, macro_block_size=1)
        print("MP4 video saved successfully.")
    except Exception as e:
        print(f"\n-------------------------------------")
        print(f"Error saving MP4 video: {e}")
        print("Ensure imageio and ffmpeg backend are installed correctly.")
        print("Try: pip install imageio-ffmpeg")
        print("-------------------------------------")