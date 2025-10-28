import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import glob
from PIL import Image

from rfdetr import RFDETRBase  # Replace with correct import if needed

# ----------------------------
# Tiling helper
# ----------------------------
def tile_frame(frame, tile_size, overlap=32):
    """
    Create overlapping tiles to better handle objects at tile boundaries
    """
    tiles = []
    h, w, _ = frame.shape
    tile_idx = 0
    
    # Calculate step size (tile_size - overlap for overlapping tiles)
    step_size = tile_size - overlap
    
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            # Ensure we don't go beyond image boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Only process tiles that are reasonably sized
            if (y_end - y) >= tile_size // 2 and (x_end - x) >= tile_size // 2:
                tile = frame[y:y_end, x:x_end]
                tiles.append((tile, tile_idx, x, y))
                tile_idx += 1
    return tiles

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
# Main processing function
# ----------------------------
def process_images_with_rfdetr(image_paths, output_folder, tile_size, model, conf_threshold):
    os.makedirs(output_folder, exist_ok=True)
    tile_output_folder = os.path.join(output_folder, "tiles")
    fullimg_output_folder = os.path.join(output_folder, "full_images")
    os.makedirs(tile_output_folder, exist_ok=True)
    os.makedirs(fullimg_output_folder, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing Images"):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read {img_path}")
            continue

        # Use overlapping tiles to better handle boundary cases
        tiles = tile_frame(frame, tile_size, overlap=64)
        full_img_boxes = []

        for tile, tile_idx, x_off, y_off in tiles:
            pil_tile = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            pred = model.predict(pil_tile, threshold=conf_threshold)

            if len(pred.xyxy) > 0:
                # Save tile with detections for debugging
                tile_filename = f"{image_name}_tile{tile_idx}.jpg"
                tile_path = os.path.join(tile_output_folder, tile_filename)
                
                # Draw boxes on tile for visualization
                tile_with_boxes = tile.copy()
                for box, cls_id, conf in zip(pred.xyxy, pred.class_id, pred.confidence):
                    if conf >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(tile_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(tile_with_boxes, f"{int(cls_id)}:{conf:.2f}", 
                                  (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.imwrite(tile_path, tile_with_boxes)

                # Save tile-level YOLO format annotation
                ann_path = tile_path.replace(".jpg", ".txt")
                with open(ann_path, "w") as f:
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

                            # Convert to full image coordinates
                            abs_x_center = x_center + x_off
                            abs_y_center = y_center + y_off
                            abs_w = w
                            abs_h = h

                            x1_full = int(abs_x_center - abs_w / 2)
                            y1_full = int(abs_y_center - abs_h / 2)
                            x2_full = int(abs_x_center + abs_w / 2)
                            y2_full = int(abs_y_center + abs_h / 2)

                            # Clamp to image boundaries
                            x1_full = max(0, min(x1_full, frame.shape[1] - 1))
                            y1_full = max(0, min(y1_full, frame.shape[0] - 1))
                            x2_full = max(0, min(x2_full, frame.shape[1] - 1))
                            y2_full = max(0, min(y2_full, frame.shape[0] - 1))

                            full_img_boxes.append((x1_full, y1_full, x2_full, y2_full, int(cls_id), float(conf)))

        # Apply improved box merging
        if full_img_boxes:
            print(f"Before merging: {len(full_img_boxes)} boxes")
            merged_boxes = non_max_suppression_custom(full_img_boxes, iou_threshold=0.2)
            print(f"After merging: {len(merged_boxes)} boxes")
            
            # Draw merged boxes on full image
            for box in merged_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cls = int(box[4]) if len(box) > 4 else 0
                conf = box[5] if len(box) > 5 else 1.0
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            full_img_out = os.path.join(fullimg_output_folder, f"{image_name}.jpg")
            cv2.imwrite(full_img_out, frame)
            
            # Save merged annotations in YOLO format
            ann_out = os.path.join(fullimg_output_folder, f"{image_name}.txt")
            with open(ann_out, "w") as f:
                for box in merged_boxes:
                    x1, y1, x2, y2 = box[:4]
                    cls = int(box[4]) if len(box) > 4 else 0
                    
                    # Convert to YOLO format
                    w = x2 - x1
                    h = y2 - y1
                    x_center = x1 + w / 2
                    y_center = y1 + h / 2
                    
                    x_center_n = x_center / frame.shape[1]
                    y_center_n = y_center / frame.shape[0]
                    w_n = w / frame.shape[1]
                    h_n = h / frame.shape[0]
                    
                    f.write(f"{cls} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}\n")

# ----------------------------
# Entry point
# ----------------------------
def main():
    image_paths = glob.glob("/scratch/s52melba/phenorob_bee/infer_temp/*.jpg")  # Adjust path
    output_folder = "/scratch/s52melba/phenorob_bee/infer_temp_output_rfdetr"
    model_ckpt = "/scratch/s52melba/phenorob_bee/detr_output_v2/checkpoint_best_total.pth"
    os.makedirs(output_folder, exist_ok=True)
    tile_size = 256
    conf_threshold = 0.5

    model = RFDETRBase(pretrain_weights=model_ckpt)
    process_images_with_rfdetr(
        image_paths=image_paths,
        output_folder=output_folder,
        tile_size=tile_size,
        model=model,
        conf_threshold=conf_threshold
    )

if __name__ == "__main__":
    main()