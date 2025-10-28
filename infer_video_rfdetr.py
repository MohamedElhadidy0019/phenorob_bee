import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
import argparse

from rfdetr import RFDETRBase  # Replace with correct import if needed

# ----------------------------
# Tiling helper
# ----------------------------
def tile_frame(frame, tile_size, overlap=32):
    """
    Create overlapping tiles to better handle objects at tile boundaries.
    Returns tiles with their grid position (row, col) for reconstruction.
    """
    tiles = []
    h, w, _ = frame.shape

    # Calculate step size (tile_size - overlap for overlapping tiles)
    step_size = tile_size - overlap

    row = 0
    for y in range(0, h, step_size):
        col = 0
        for x in range(0, w, step_size):
            # Ensure we don't go beyond image boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            # Only process tiles that are reasonably sized
            if (y_end - y) >= tile_size // 2 and (x_end - x) >= tile_size // 2:
                tile = frame[y:y_end, x:x_end]
                tiles.append((tile, row, col, x, y))
                col += 1
        if col > 0:  # Only increment row if we added tiles
            row += 1

    return tiles

# ----------------------------
# Frame extraction from video
# ----------------------------
def extract_frames_from_video(video_path, fps=1):
    """
    Extract frames from video at specified FPS.
    If fps=1, extracts 1 frame per second.

    Returns: list of (frame_number, frame) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Could not determine FPS for {video_path}, using default 30")
        video_fps = 30

    # Calculate frame interval: how many frames to skip
    frame_interval = int(video_fps / fps)
    if frame_interval < 1:
        frame_interval = 1

    frames = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frames.append((extracted_count, frame))
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from {os.path.basename(video_path)} (video FPS: {video_fps:.2f}, extraction FPS: {fps})")
    return frames

# ----------------------------
# Improved box merging helper
# ----------------------------
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_distance(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def should_merge_boxes(box1, box2, iou_threshold=0.1, distance_threshold=50, class_match=True):
    """
    Determine if two boxes should be merged based on multiple criteria
    """
    # Check if classes match (if required)
    if class_match and len(box1) > 4 and len(box2) > 4:
        if box1[4] != box2[4]:  # Different classes
            return False

    # Calculate IoU
    iou = calculate_iou(box1, box2)

    # Calculate distance between centers
    distance = calculate_distance(box1, box2)

    # Merge if:
    # 1. IoU is above threshold (overlapping boxes)
    # 2. OR distance is small (adjacent boxes for same bee)
    return iou > iou_threshold or distance < distance_threshold

def non_max_suppression_custom(boxes, iou_threshold=0.3):
    """
    Custom NMS that's more suitable for merging split detections
    """
    if len(boxes) == 0:
        return []

    # Sort by confidence (descending)
    if len(boxes[0]) > 5:  # Has confidence scores
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

    merged = []
    used = set()

    for i, box1 in enumerate(boxes):
        if i in used:
            continue

        # Start with current box
        current_boxes = [box1]
        used.add(i)

        # Find all boxes that should be merged with current box
        for j, box2 in enumerate(boxes):
            if j in used:
                continue

            if should_merge_boxes(box1, box2, iou_threshold=iou_threshold):
                current_boxes.append(box2)
                used.add(j)

        # Merge all boxes in current group
        if len(current_boxes) == 1:
            merged.append(current_boxes[0])
        else:
            # Calculate merged box coordinates
            x1_coords = [b[0] for b in current_boxes]
            y1_coords = [b[1] for b in current_boxes]
            x2_coords = [b[2] for b in current_boxes]
            y2_coords = [b[3] for b in current_boxes]

            merged_box = [
                min(x1_coords),  # x1
                min(y1_coords),  # y1
                max(x2_coords),  # x2
                max(y2_coords),  # y2
            ]

            # Add class and confidence if available
            if len(current_boxes[0]) > 4:
                merged_box.append(current_boxes[0][4])  # class
            if len(current_boxes[0]) > 5:
                # Use maximum confidence
                merged_box.append(max(b[5] for b in current_boxes))

            merged.append(tuple(merged_box))

    return merged

# ----------------------------
# Main processing function for video frames
# ----------------------------
def process_video_with_rfdetr(video_path, output_folder, tile_size, model, conf_threshold, fps, overlap=64):
    """
    Process a single video: extract frames and detect objects in tiles.

    Naming scheme for tiles: {video_name}_frame{frame_num:06d}_tile_r{row}_c{col}.jpg
    This allows reconstruction by sorting tiles by row and column for each frame.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create output directories
    video_output_folder = os.path.join(output_folder, video_name)
    tile_images_folder = os.path.join(video_output_folder, "tiles", "images")
    tile_labels_folder = os.path.join(video_output_folder, "tiles", "labels")
    os.makedirs(tile_images_folder, exist_ok=True)
    os.makedirs(tile_labels_folder, exist_ok=True)

    # Extract frames from video
    frames = extract_frames_from_video(video_path, fps=fps)

    if len(frames) == 0:
        print(f"No frames extracted from {video_path}")
        return

    # Process each frame
    for frame_num, frame in tqdm(frames, desc=f"Processing {video_name}"):
        # Use overlapping tiles to better handle boundary cases
        tiles = tile_frame(frame, tile_size, overlap=overlap)

        for tile, row, col, x_off, y_off in tiles:
            # Naming scheme: {video_name}_frame{frame_num:06d}_tile_r{row}_c{col}.jpg
            tile_filename = f"{video_name}_frame{frame_num:06d}_tile_r{row:03d}_c{col:03d}.jpg"
            tile_image_path = os.path.join(tile_images_folder, tile_filename)
            tile_label_path = os.path.join(tile_labels_folder, tile_filename.replace(".jpg", ".txt"))

            # Save raw tile (without bounding boxes)
            cv2.imwrite(tile_image_path, tile)

            # Run inference
            pil_tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            pred = model.predict(pil_tile, threshold=conf_threshold)

            # Save tile-level YOLO format annotation only if detections exist
            if len(pred.xyxy) > 0:
                with open(tile_label_path, "w") as f:
                    for box, cls_id, conf in zip(pred.xyxy, pred.class_id, pred.confidence):
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box
                            w = x2 - x1
                            h = y2 - y1
                            x_center = x1 + w / 2
                            y_center = y1 + h / 2

                            x_center_n = x_center / tile.shape[1]
                            y_center_n = y_center / tile.shape[0]
                            w_n = w / tile.shape[1]
                            h_n = h / tile.shape[0]

                            f.write(f"{int(cls_id)} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}\n")

# ----------------------------
# Find all videos in folder structure
# ----------------------------
def find_videos(root_folder, extensions=['.mov', '.MOV', '.mp4', '.MP4', '.avi', '.AVI']):
    """
    Find all video files in the folder structure.
    """
    video_files = []
    for ext in extensions:
        pattern = os.path.join(root_folder, "**", f"*{ext}")
        video_files.extend(glob.glob(pattern, recursive=True))

    return sorted(video_files)

# ----------------------------
# Entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Process videos with RF-DETR object detection")
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Root folder containing videos (will search recursively)")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Output folder for results")
    parser.add_argument("--model_ckpt", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Size of tiles for processing (default: 256)")
    parser.add_argument("--overlap", type=int, default=64,
                       help="Overlap between tiles in pixels (default: 64)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                       help="Confidence threshold for detections (default: 0.5)")
    parser.add_argument("--fps", type=float, default=1.0,
                       help="Frames per second to extract (e.g., 1.0 = 1 frame per second, 0.5 = 1 frame per 2 seconds)")

    args = parser.parse_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Find all videos
    video_files = find_videos(args.input_folder)
    print(f"Found {len(video_files)} video files")

    if len(video_files) == 0:
        print("No video files found. Exiting.")
        return

    # Load model
    print(f"Loading model from {args.model_ckpt}")
    model = RFDETRBase(pretrain_weights=args.model_ckpt)

    # Process each video
    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        process_video_with_rfdetr(
            video_path=video_path,
            output_folder=args.output_folder,
            tile_size=args.tile_size,
            model=model,
            conf_threshold=args.conf_threshold,
            fps=args.fps,
            overlap=args.overlap
        )

    print("\nProcessing complete!")
    print(f"Results saved to: {args.output_folder}")

if __name__ == "__main__":
    main()
