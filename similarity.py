# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_insightface_model():
    print("Loading InsightFace model...")
    # Initialize FaceAnalysis
    # Set ctx_id=0 to use GPU. If an error occurs, change to -1 to use CPU.
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        print("Warning: GPU not available, using CPU.")
        app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Return the internal recognition model
    return app.models['recognition']

def get_embedding(model, image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image: {image_path}")
        return None

    # Simulate preprocessing: resize to 112x112 (ArcFace standard input)
    # Assuming the input is already a cropped face
    input_face = cv2.resize(img, (112, 112))

    # Extract features (512-dim)
    # get_feat returns unnormalized feature vectors
    embedding = model.get_feat(input_face).flatten()

    # === Key Step: L2 Normalization ===
    # This is necessary for calculating Cosine Similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    
    return embedding

def main():
    parser = argparse.ArgumentParser(description='Compare two face images using ArcFace Cosine Similarity')
    parser.add_argument('--img1', type=str, required=True, help='Path to the first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to the second image')
    args = parser.parse_args()

    # 1. Load model
    rec_model = load_insightface_model()

    # 2. Extract embeddings
    emb1 = get_embedding(rec_model, args.img1)
    emb2 = get_embedding(rec_model, args.img2)

    if emb1 is None or emb2 is None:
        return

    # 3. Calculate Cosine Similarity
    # Since L2 normalization is done, the Dot Product equals Cosine Similarity
    similarity = np.dot(emb1, emb2)

    # 4. Output results
    print("\n" + "="*40)
    print(f"Image 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    print("-" * 40)
    print(f"Cosine Similarity: {similarity:.4f}")
    print("="*40)
    
    # Judgment suggestion (based on general experience)
    print("\n[Reference Judgment]")
    if similarity > 0.5:
        print(">> High probability: Same Person")
    elif similarity > 0.35:
        print(">> Likely Same Person (Depends on threshold)")
    else:
        print(">> Different People")

if __name__ == "__main__":
    main()