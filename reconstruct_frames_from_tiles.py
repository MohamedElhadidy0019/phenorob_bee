#!/usr/bin/env python3
"""
Helper script to reconstruct full frames from tiles.
Reads the tile metadata files and reconstructs the original frames.
"""

import os
import cv2
import numpy as np
import glob
import argparse
from collections import defaultdict

def parse_tile_metadata(metadata_file):
    """
    Parse tile metadata file.
    Returns list of tile info dicts.
    """
    tiles = []
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 7:
                tiles.append({
                    'filename': parts[0],
                    'row': int(parts[1]),
                    'col': int(parts[2]),
                    'x_offset': int(parts[3]),
                    'y_offset': int(parts[4]),
                    'width': int(parts[5]),
                    'height': int(parts[6])
                })

    return tiles

def reconstruct_frame_from_tiles(tile_folder, metadata_file, output_path=None):
    """
    Reconstruct a full frame from tiles using metadata.

    Args:
        tile_folder: Folder containing the tiles
        metadata_file: Path to the tile metadata file
        output_path: Optional path to save reconstructed frame

    Returns:
        Reconstructed frame as numpy array
    """
    # Parse metadata
    tiles_info = parse_tile_metadata(metadata_file)

    if not tiles_info:
        print(f"No tiles found in metadata: {metadata_file}")
        return None

    # Determine full frame dimensions
    max_x = max(t['x_offset'] + t['width'] for t in tiles_info)
    max_y = max(t['y_offset'] + t['height'] for t in tiles_info)

    # Create empty frame (will blend overlapping regions)
    frame = np.zeros((max_y, max_x, 3), dtype=np.float32)
    weights = np.zeros((max_y, max_x), dtype=np.float32)

    # Load and place each tile
    for tile_info in tiles_info:
        tile_path = os.path.join(tile_folder, tile_info['filename'])

        if not os.path.exists(tile_path):
            print(f"Warning: Tile not found: {tile_path}")
            continue

        tile = cv2.imread(tile_path)
        if tile is None:
            print(f"Warning: Could not read tile: {tile_path}")
            continue

        # Get placement coordinates
        x = tile_info['x_offset']
        y = tile_info['y_offset']
        h, w = tile.shape[:2]

        # Add tile to frame with averaging for overlaps
        frame[y:y+h, x:x+w] += tile
        weights[y:y+h, x:x+w] += 1

    # Normalize by weights (average overlapping regions)
    mask = weights > 0
    frame[mask] /= weights[mask, np.newaxis]

    # Convert back to uint8
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        print(f"Saved reconstructed frame to: {output_path}")

    return frame

def reconstruct_all_frames(input_folder, output_folder):
    """
    Reconstruct all frames from a video processing output folder.

    Expected structure:
        input_folder/
            video_name/
                tiles/
                    video_name_frame000000_tile_r000_c000.jpg
                    ...
                frames/
                    video_name_frame000000_tile_metadata.txt
                    ...
    """
    # Find all video folders
    video_folders = [d for d in glob.glob(os.path.join(input_folder, "*"))
                    if os.path.isdir(d)]

    for video_folder in video_folders:
        video_name = os.path.basename(video_folder)
        tile_folder = os.path.join(video_folder, "tiles")
        frames_folder = os.path.join(video_folder, "frames")

        if not os.path.exists(tile_folder) or not os.path.exists(frames_folder):
            print(f"Skipping {video_name}: missing tiles or frames folder")
            continue

        # Find all metadata files
        metadata_files = sorted(glob.glob(os.path.join(frames_folder, "*_tile_metadata.txt")))

        print(f"\nProcessing video: {video_name}")
        print(f"Found {len(metadata_files)} frames to reconstruct")

        # Create output folder for this video
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Reconstruct each frame
        for metadata_file in metadata_files:
            # Extract frame number from filename
            basename = os.path.basename(metadata_file)
            frame_name = basename.replace('_tile_metadata.txt', '')

            output_path = os.path.join(video_output_folder, f"{frame_name}_reconstructed.jpg")

            reconstruct_frame_from_tiles(tile_folder, metadata_file, output_path)

        print(f"Completed reconstruction for {video_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct full frames from tiles using metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reconstruct all frames from video processing output
  python reconstruct_frames_from_tiles.py --input_folder output/ --output_folder reconstructed/

  # Reconstruct a single frame
  python reconstruct_frames_from_tiles.py --metadata video1/frames/video1_frame000001_tile_metadata.txt --tiles video1/tiles/ --output frame.jpg
        """
    )

    parser.add_argument("--input_folder", type=str,
                       help="Input folder containing video processing results")
    parser.add_argument("--output_folder", type=str,
                       help="Output folder for reconstructed frames")
    parser.add_argument("--metadata", type=str,
                       help="Single metadata file to process")
    parser.add_argument("--tiles", type=str,
                       help="Folder containing tiles (for single frame reconstruction)")
    parser.add_argument("--output", type=str,
                       help="Output path for single reconstructed frame")

    args = parser.parse_args()

    if args.input_folder and args.output_folder:
        # Batch mode: reconstruct all frames
        reconstruct_all_frames(args.input_folder, args.output_folder)
    elif args.metadata and args.tiles and args.output:
        # Single frame mode
        reconstruct_frame_from_tiles(args.tiles, args.metadata, args.output)
    else:
        parser.print_help()
        print("\nError: Please provide either:")
        print("  1. --input_folder and --output_folder for batch processing, OR")
        print("  2. --metadata, --tiles, and --output for single frame reconstruction")

if __name__ == "__main__":
    main()
