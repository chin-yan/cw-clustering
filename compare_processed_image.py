import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def compare_face_preprocessing(result_dir):
    """
    Compare original and preprocessed face images
    """
    # Original face directory
    original_faces_dir = os.path.join(result_dir, 'faces')
    
    # Preprocessed face directory
    preprocessed_faces_dir = os.path.join(result_dir, 'retrieval_faces')
    
    # Get image lists
    original_faces = [f for f in os.listdir(original_faces_dir) if f.endswith('.jpg')]
    preprocessed_faces = [f for f in os.listdir(preprocessed_faces_dir) if f.endswith('.jpg')]
    
    print("Sample original filenames:")
    for f in original_faces[:10]:
        print(f)
    
    print("\nSample preprocessed filenames:")
    for f in preprocessed_faces[:10]:
        print(f)
    
    # Compare filename structures
    def extract_base_name(filename):
        # Try different patterns to extract base name
        patterns = [
            r'frame_(\d+)_face_(\d+)\.jpg',
            r'frame_(\d+)_face_(\d+)',
            r'retrieval_frame_(\d+)_face_(\d+)\.jpg',
            r'retrieval_frame_(\d+)_face_(\d+)',
            r'frame_(\d+)_sideface_(\d+)\.jpg'
        ]
        
        import re
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                # Standardize to a common format
                return f'frame_{match.group(1)}_face_{match.group(2)}.jpg'
        return filename

    # Use base name matching
    original_base_names = set(extract_base_name(f) for f in original_faces)
    preprocessed_base_names = set(extract_base_name(f) for f in preprocessed_faces)
    
    matching_faces = list(original_base_names & preprocessed_base_names)
    
    print(f"\nOriginal face image count: {len(original_faces)}")
    print(f"Preprocessed face image count: {len(preprocessed_faces)}")
    print(f"Matching base filenames count: {len(matching_faces)}")
    
    if not matching_faces:
        print("No matching face images found")
        return
    
    # Limit display count
    matching_faces = matching_faces[:10]
    
    # Create visualization
    fig, axes = plt.subplots(len(matching_faces), 2, figsize=(12, 4*len(matching_faces)))
    
    for i, base_name in enumerate(matching_faces):
        # Find corresponding actual filenames in both directories
        original_path = next(
            os.path.join(original_faces_dir, f) 
            for f in original_faces 
            if extract_base_name(f) == base_name
        )
        preprocessed_path = next(
            os.path.join(preprocessed_faces_dir, f) 
            for f in preprocessed_faces 
            if extract_base_name(f) == base_name
        )
        
        # Read original and preprocessed images
        original_img = cv2.imread(original_path)
        preprocessed_img = cv2.imread(preprocessed_path)
        
        # Convert color space
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
        
        # Plot images
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f'Original: {os.path.basename(original_path)}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(preprocessed_img)
        axes[i, 1].set_title(f'Preprocessed: {os.path.basename(preprocessed_path)}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Face Image Preprocessing Comparison', fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Save comparison result
    comparison_path = os.path.join(result_dir, 'face_preprocessing_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison result saved to {comparison_path}")
    plt.close()

# Usage example
if __name__ == "__main__":
    # Replace with your actual result directory path
    result_dir = r"C:\Users\VIPLAB\Desktop\Yan\video-face-clustering\result"
    compare_face_preprocessing(result_dir)