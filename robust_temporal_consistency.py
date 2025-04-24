# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle
from collections import deque, Counter
import colorsys
from tqdm import tqdm

class TemporalConsistencyEnhancer:
    """
    Enhance face identification in videos by using temporal consistency across frames.
    Fills in unmatched faces using historical matches from previous frames.
    """
    
    def __init__(self, temporal_window=10, confidence_threshold=0.5, min_votes=3):
        """
        Initialize the temporal consistency enhancer
        
        Args:
            temporal_window: Number of previous frames to consider
            confidence_threshold: Minimum confidence score for a match
            min_votes: Minimum votes required to consider a match valid
        """
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes
        self.face_history = {}  # face_id -> deque of (frame_idx, match_idx, similarity)
    
    def process_face(self, face_id, frame_idx, match_idx, similarity):
        """
        Process a face detection and return an enhanced match using temporal consistency
        
        Args:
            face_id: Unique identifier for the face
            frame_idx: Current frame index
            match_idx: Current match index (-1 if unmatched)
            similarity: Current match similarity score
            
        Returns:
            Tuple of (match_idx, similarity, is_filled_from_history)
        """
        # Initialize history for new faces
        if face_id not in self.face_history:
            self.face_history[face_id] = deque(maxlen=self.temporal_window)
        
        # Store original match result
        original_match = (match_idx, similarity)
        
        # Add current detection to history
        self.face_history[face_id].append((frame_idx, match_idx, similarity))
        
        # Check if face is unmatched or low confidence match
        if match_idx < 0 or similarity < self.confidence_threshold:
            # Try to use history to find a match
            if len(self.face_history[face_id]) >= self.min_votes:
                # Get match history excluding the current frame
                history = list(self.face_history[face_id])[:-1]
                
                # Count votes for each match_idx
                vote_counter = Counter()
                for _, hist_match, hist_sim in history:
                    if hist_match >= 0 and hist_sim >= self.confidence_threshold * 0.9:
                        vote_counter[hist_match] += 1
                
                # Get the most common match_idx with at least min_votes
                for common_match, count in vote_counter.most_common():
                    if count >= self.min_votes and common_match >= 0:
                        # Use historical match
                        return common_match, max(similarity, self.confidence_threshold * 0.8), True
        
        # Return original match if no historical match found
        return match_idx, similarity, False
    
    def reset(self):
        """Reset the history"""
        self.face_history = {}

def enhance_video_temporal_consistency(input_video, annotation_file, output_video, centers_data_path,
                                    temporal_window=10, confidence_threshold=0.5, min_votes=3):
    """
    Enhance a video with face annotations by applying temporal consistency
    
    Args:
        input_video: Path to the original video
        annotation_file: Path to the pickle file with face annotation results
        output_video: Path to save the enhanced video
        centers_data_path: Path to the cluster centers data
        temporal_window: Number of frames to consider for temporal consistency
        confidence_threshold: Minimum confidence threshold for matches
        min_votes: Minimum votes needed for a historical match
    """
    # Load annotation results
    print(f"Loading annotation results from {annotation_file}...")
    with open(annotation_file, 'rb') as f:
        results = pickle.load(f)
    
    # Load centers data for visualization
    print("Loading cluster center data...")
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    
    if 'cluster_centers' not in centers_data:
        raise ValueError("Missing cluster center information in the data")
    
    _, center_paths = centers_data['cluster_centers']
    n_centers = len(center_paths)
    
    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create colors for visualization
    colors = []
    for i in range(n_centers):
        h = i / n_centers
        s = 0.8
        v = 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    # Initialize temporal consistency enhancer
    enhancer = TemporalConsistencyEnhancer(
        temporal_window=temporal_window,
        confidence_threshold=confidence_threshold,
        min_votes=min_votes
    )
    
    # Statistics for reporting
    total_faces = 0
    unmatched_before = 0
    filled_matches = 0
    
    # Process each frame
    print(f"Processing {total_frames} frames with temporal consistency...")
    pbar = tqdm(total=total_frames)
    
    for frame_idx in range(total_frames):
        # Read the frame
        ret, frame = cap.read()