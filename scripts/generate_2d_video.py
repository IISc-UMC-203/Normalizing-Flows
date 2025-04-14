# scripts/generate_2d_video.py
import argparse
import os

# Project local imports
from src.visualize import create_video_from_frames

def main(args):
    if not os.path.isdir(args.frame_dir):
        print(f"Error: Frame directory not found: {args.frame_dir}")
        return

    # Determine output filename based on frame directory if not specified
    if args.output is None:
        dataset_name = os.path.basename(args.frame_dir) # Assumes dir name is dataset name
        output_filename = f"transformation_{dataset_name}.mp4"
        output_path = os.path.join(os.path.dirname(args.frame_dir), output_filename) # Save in parent dir
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure output dir exists

    # Ensure output has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path += '.mp4'

    create_video_from_frames(args.frame_dir, output_path, args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile 2D animation frames into an MP4 video")
    parser.add_argument('--frame_dir', type=str, required=True, help='Directory containing the saved PNG frames')
    parser.add_argument('--output', type=str, default=None, help='Output video file path (e.g., results/videos/moons.mp4). If None, saves in parent of frame_dir.')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for the output video')
    args = parser.parse_args()
    main(args)