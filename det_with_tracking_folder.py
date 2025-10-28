import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from collections import defaultdict
import glob
import torch

from rfdetr import RFDETRBase
import norfair
from norfair import Detection, Tracker

class BeeTracker:
    def __init__(self, model_path, conf_threshold=0.5, distance_threshold=30, hit_counter_max=10, initialization_delay=3, frame_rate=30, batch_size=1):
        """
        Initialize the bee tracking system with Norfair tracker
        
        Args:
            model_path: Path to RF-DETR model weights
            conf_threshold: Confidence threshold for detections
            distance_threshold: Maximum distance for matching detections to tracks (pixels)
            hit_counter_max: Maximum frames to keep a track alive without detections
            initialization_delay: Frames to wait before initializing a new track
            frame_rate: Video frame rate
            batch_size: Number of frames to process in batch
        """
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model on GPU
        self.model = RFDETRBase(pretrain_weights=model_path)
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        
        self.conf_threshold = conf_threshold
        self.frame_rate = frame_rate
        self.batch_size = batch_size
        
        # Initialize Norfair tracker
        self.tracker = Tracker(
            distance_function="euclidean",  # Use centroid distance for bee tracking
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=hit_counter_max
        )
        
        # Track statistics
        self.track_history = defaultdict(list)  # track_id -> [frame_numbers]
        self.track_durations = {}  # track_id -> duration in seconds
        self.track_classes = defaultdict(list)  # track_id -> [class_ids for each detection]
        self.frame_count = 0
        
    def reset_tracking_state(self):
        """Reset only the tracking state for new video, keep the model loaded"""
        # Create new tracker instance with same parameters
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=self.tracker.distance_threshold,
            hit_counter_max=self.tracker.hit_counter_max,
            initialization_delay=self.tracker.initialization_delay,
            pointwise_hit_counter_max=self.tracker.pointwise_hit_counter_max
        )
        
        # Reset tracking data but keep model loaded
        self.track_history = defaultdict(list)
        self.track_durations = {}
        self.track_classes = defaultdict(list)
        self.frame_count = 0
        
    def detect_frames_batch(self, frames):
        """
        Run RF-DETR inference on a batch of frames
        
        Args:
            frames: List of OpenCV frames (BGR format)
            
        Returns:
            List of lists of Norfair Detection objects (one list per frame)
        """
        all_detections = []
        
        try:
            # Convert frames to PIL images
            pil_images = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_images.append(pil_image)
            
            # Process batch (if model supports batch processing)
            if len(pil_images) == 1 or not hasattr(self.model, 'predict_batch'):
                # Process frames individually if no batch support
                for pil_image in pil_images:
                    frame_detections = self._process_single_frame(pil_image)
                    all_detections.append(frame_detections)
            else:
                # Use batch processing if available
                try:
                    batch_preds = self.model.predict_batch(pil_images, threshold=self.conf_threshold)
                    for pred in batch_preds:
                        frame_detections = self._convert_pred_to_detections(pred)
                        all_detections.append(frame_detections)
                except AttributeError:
                    # Fallback to individual processing
                    for pil_image in pil_images:
                        frame_detections = self._process_single_frame(pil_image)
                        all_detections.append(frame_detections)
                        
        except Exception as e:
            print(f"Error in batch detection: {e}")
            # Return empty detections for all frames in batch
            all_detections = [[] for _ in frames]
            
        return all_detections
    
    def _process_single_frame(self, pil_image):
        """Process a single frame and return detections"""
        try:
            pred = self.model.predict(pil_image, threshold=self.conf_threshold)
            return self._convert_pred_to_detections(pred)
        except Exception as e:
            print(f"Error processing single frame: {e}")
            return []
    
    def _convert_pred_to_detections(self, pred):
        """Convert model prediction to Norfair Detection objects"""
        detections = []
        
        if hasattr(pred, 'xyxy') and len(pred.xyxy) > 0:
            for box, cls_id, conf in zip(pred.xyxy, pred.class_id, pred.confidence):
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = box
                    
                    # Calculate centroid for Norfair
                    centroid_x = (x1 + x2) / 2
                    centroid_y = (y1 + y2) / 2
                    
                    # Create Norfair Detection object
                    detection = Detection(
                        points=np.array([[centroid_x, centroid_y]]),
                        scores=np.array([conf]),
                        data={
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': int(cls_id),
                            'confidence': float(conf)
                        }
                    )
                    detections.append(detection)
        
        return detections
    
    def update_tracker(self, detections, frame_idx):
        """
        Update tracker with new detections
        
        Args:
            detections: List of Norfair Detection objects
            frame_idx: Current frame index
            
        Returns:
            List of TrackedObject instances
        """
        # Update tracker
        tracked_objects = self.tracker.update(detections=detections)
        
        # Update track history and class information
        for tracked_obj in tracked_objects:
            track_id = tracked_obj.id
            if track_id is not None and tracked_obj.last_detection is not None:
                self.track_history[track_id].append(frame_idx)
                class_id = tracked_obj.last_detection.data['class_id']
                self.track_classes[track_id].append(class_id)
        
        return tracked_objects
    
    def get_dominant_class(self, track_id):
        """Get the most common class for a track"""
        if track_id not in self.track_classes or not self.track_classes[track_id]:
            return None
        
        class_counts = {}
        for class_id in self.track_classes[track_id]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        return max(class_counts, key=class_counts.get)
    
    def calculate_track_durations(self):
        """Calculate duration for each track ID"""
        for track_id, frame_list in self.track_history.items():
            if len(frame_list) > 0:
                start_frame = min(frame_list)
                end_frame = max(frame_list)
                duration_frames = end_frame - start_frame + 1
                duration_seconds = duration_frames / self.frame_rate
                self.track_durations[track_id] = duration_seconds

def process_video_analysis_only(video_path, output_json_path, tracker):
    """
    Process video with bee detection and tracking, output only JSON analysis
    
    Args:
        video_path: Input video path
        output_json_path: Output JSON path for statistics
        tracker: BeeTracker instance
    """
    # Reset tracker state for new video
    tracker.reset_tracking_state()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Update tracker frame rate
    tracker.frame_rate = fps
    
    frame_idx = 0
    frames_buffer = []
    
    # Process video in batches
    progress_desc = f"  Processing {os.path.basename(video_path)}"
    with tqdm(total=total_frames, desc=progress_desc, unit="frames", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.1f}%') as pbar:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in buffer
                if frames_buffer:
                    batch_detections = tracker.detect_frames_batch(frames_buffer)
                    for i, detections in enumerate(batch_detections):
                        current_frame_idx = frame_idx - len(frames_buffer) + i
                        tracker.frame_count = current_frame_idx
                        tracker.update_tracker(detections, current_frame_idx)
                        
                        # Update progress for each processed frame
                        pbar.update(1)
                        
                        # Print periodic progress updates
                        if current_frame_idx % max(1, total_frames // 20) == 0 or current_frame_idx == total_frames - 1:
                            percentage = (current_frame_idx + 1) / total_frames * 100
                            tracks_found = len([t for t in tracker.track_durations.keys()])
                            print(f"    Progress: {percentage:.1f}% - Frame {current_frame_idx + 1}/{total_frames} - Tracks found: {tracks_found}")
                break
            
            frames_buffer.append(frame)
            
            # Process batch when buffer is full
            if len(frames_buffer) >= tracker.batch_size:
                batch_detections = tracker.detect_frames_batch(frames_buffer)
                for i, detections in enumerate(batch_detections):
                    current_frame_idx = frame_idx - tracker.batch_size + i + 1
                    tracker.frame_count = current_frame_idx
                    tracker.update_tracker(detections, current_frame_idx)
                
                frames_buffer = []
                pbar.update(tracker.batch_size)
                
                # Print periodic progress updates
                if current_frame_idx % max(1, total_frames // 20) == 0:
                    percentage = current_frame_idx / total_frames * 100
                    current_tracks = len([t for t in tracker.track_history.keys() if tracker.track_history[t]])
                    print(f"    Progress: {percentage:.1f}% - Frame {current_frame_idx}/{total_frames} - Active tracks: {current_tracks}")
            
            frame_idx += 1
    
    # Clean up
    cap.release()
    
    # Calculate track durations
    tracker.calculate_track_durations()
    
    # Create statistics
    total_track_time = sum(tracker.track_durations.values())
    
    # Create track details with class information
    track_details = {}
    for track_id in tracker.track_durations.keys():
        dominant_class = tracker.get_dominant_class(track_id)
        track_details[str(track_id)] = {
            'duration_seconds': tracker.track_durations[track_id],
            'frame_count': len(tracker.track_history[track_id]),
            'dominant_class_id': dominant_class,
            'all_classes': list(set(tracker.track_classes[track_id])) if track_id in tracker.track_classes else [],
            'first_frame': min(tracker.track_history[track_id]) if tracker.track_history[track_id] else 0,
            'last_frame': max(tracker.track_history[track_id]) if tracker.track_history[track_id] else 0
        }
    
    # Get video filename without path and extension - handle multiple dots correctly
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]  # Correctly handles "20230630_plot1_11.55.MOV"
    
    stats = {
        'video_name': video_name,
        'video_path': video_path,
        'video_properties': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps
        },
        'tracking_results': {
            'total_tracks': len(tracker.track_durations),
            'total_tracking_time_seconds': total_track_time,
            'average_track_duration_seconds': total_track_time / len(tracker.track_durations) if len(tracker.track_durations) > 0 else 0,
            'longest_track_seconds': max(tracker.track_durations.values()) if tracker.track_durations else 0,
            'shortest_track_seconds': min(tracker.track_durations.values()) if tracker.track_durations else 0,
            'track_details': track_details
        },
        'processing_parameters': {
            'confidence_threshold': tracker.conf_threshold,
            'distance_threshold': tracker.tracker.distance_threshold,
            'hit_counter_max': tracker.tracker.hit_counter_max,
            'initialization_delay': tracker.tracker.initialization_delay,
            'batch_size': tracker.batch_size
        }
    }
    
    # Save statistics to JSON
    try:
        with open(output_json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Analysis saved to: {output_json_path}")
        return True
    except Exception as e:
        print(f"  Error saving JSON: {e}")
        return False

def process_video_folder(input_folder, output_folder, model_path, conf_threshold=0.5, distance_threshold=30, 
                        hit_counter_max=10, initialization_delay=3, batch_size=4):
    """
    Process all videos in a folder with bee detection and tracking
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder for output JSON files
        model_path: RF-DETR model path
        conf_threshold: Detection confidence threshold
        distance_threshold: Maximum distance for matching tracks (pixels)
        hit_counter_max: Maximum frames to keep a track alive
        initialization_delay: Frames to wait before initializing new track
        batch_size: Number of frames to process in batch
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Supported video extensions
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm']

    # Find all video files recursively
    video_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if file has a video extension (case-insensitive)
            if any(file.lower().endswith(f'.{ext}') for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"No video files found in {input_folder} or its subdirectories")
        return
    print(f"Found {len(video_files)} video files to process")
    # print names of video files
    print(f"Video files {video_files}")
    
    # Initialize tracker once for all videos - this is the expensive operation
    print("🔄 Initializing model (this may take a moment)...")
    tracker = BeeTracker(
        model_path=model_path,
        conf_threshold=conf_threshold,
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
        batch_size=batch_size
    )
    print("✅ Model initialized successfully!")
    
    successful_processes = 0
    failed_processes = 0
    
    # Process each video
    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(video_files)}] Processing: {video_filename}")
        print(f"{'='*80}")
        
        if not os.path.exists(video_path):
            print(f"  ❌ Warning: File does not exist, skipping...")
            failed_processes += 1
            continue
        
        # Create output JSON filename - handle multiple dots in filename
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]  # This correctly handles multiple dots
        output_json_path = os.path.join(output_folder, f"{video_name}_analysis.json")
        
        # Show estimated progress for the overall batch
        batch_percentage = (i / len(video_files)) * 100
        print(f"  📊 Overall batch progress: {batch_percentage:.1f}% ({i}/{len(video_files)} videos completed)")
        
        # Process the video
        success = process_video_analysis_only(video_path, output_json_path, tracker)
        
        if success:
            successful_processes += 1
            # Print brief statistics
            if tracker.track_durations:
                total_tracks = len(tracker.track_durations)
                longest_track = max(tracker.track_durations.values())
                avg_track = sum(tracker.track_durations.values()) / total_tracks
                print(f"  ✅ Analysis complete!")
                print(f"     📈 Found {total_tracks} tracks")
                print(f"     ⏱️  Longest track: {longest_track:.2f}s")
                print(f"     📊 Average track: {avg_track:.2f}s")
            else:
                print(f"  ✅ Analysis complete - No tracks found")
        else:
            failed_processes += 1
            print(f"  ❌ Processing failed")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"📁 Total videos processed: {len(video_files)}")
    print(f"✅ Successful: {successful_processes}")
    print(f"❌ Failed: {failed_processes}")
    print(f"📈 Success rate: {(successful_processes/len(video_files)*100):.1f}%")
    print(f"💾 Output folder: {output_folder}")
    print(f"{'='*80}")

def main():
    """Main function with updated parameters"""
    # Configuration parameters
    input_folder = "/datan/bee_data/video_records_2024"  # Folder containing videos
    output_folder = "/scratch/s52melba/videos_tracking_output_2024"  # Folder for JSON outputs
    model_path = "/scratch/s52melba/detr_train_2/checkpoint_best_total.pth"
    
    # Detection and tracking parameters
    conf_threshold = 0.3         # Detection confidence threshold
    distance_threshold = 90      # Max pixel distance for track matching
    hit_counter_max = 10         # Max frames to keep track alive without detection
    initialization_delay = 5     # Frames to wait before confirming new track
    batch_size = 256              # Number of frames to process in batch (adjust based on GPU memory)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        print("Please modify the 'input_folder' variable in the main() function")
        return
    
    print("Starting batch bee tracking analysis...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model path: {model_path}")
    print(f"Detection threshold: {conf_threshold}")
    print(f"Distance threshold: {distance_threshold} pixels")
    print(f"Max missing frames: {hit_counter_max}")
    print(f"Batch size: {batch_size}")
    
    # Process all videos in folder
    process_video_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        conf_threshold=conf_threshold,
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()

# Installation requirements:
# pip install norfair torch torchvision opencv-python pillow tqdm
#
# Usage:
# 1. Set the input_folder path to your folder containing videos
# 2. Set the output_folder path where you want JSON analysis files
# 3. Adjust batch_size based on your GPU memory (start with 4, increase if you have more VRAM)
# 4. Run the script
#
# The script will:
# - Process all video files in the input folder
# - Generate individual JSON analysis files for each video
# - Use GPU acceleration with batch processing
# - Print progress and statistics for each video