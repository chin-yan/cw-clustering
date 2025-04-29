# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import facenet.src.facenet as facenet
from tqdm import tqdm

def load_model(sess, model_dir):
    """
    Loading the FaceNet model

    Args:
    sess: TensorFlow session
    model_dir: model directory
    """
    meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
    print(f'Metagraph file: {meta_file}')
    print(f'Checkpoint file: {ckpt_file}')
    
    saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
    saver.restore(sess, os.path.join(model_dir, ckpt_file))
    print("Model loaded successfully")

def compute_facial_encodings(sess, images_placeholder, embeddings, phase_train_placeholder, 
                             image_size, embedding_size, nrof_images, nrof_batches, 
                             emb_array, batch_size, paths):
    """
        Calculate facial feature encoding

        Args:
            sess: TensorFlow session
            images_placeholder: image input placeholder
            embeddings: embedding vector output
            phase_train_placeholder: training phase placeholder
            image_size: image size
            embedding_size: embedding vector size
            nrof_images: number of images
            nrof_batches: number of batches
            emb_array: embedding vector array
            batch_size: batch size
            paths: list of image paths

        Returns:
            Facial feature encoding dictionary
    """
    print("Calculating facial feature encoding...")
    for i in tqdm(range(nrof_batches)):
        start_index = i * batch_size
        end_index = min((i+1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        
        # Loading image data
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        
        # Calculate the embedding vector
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
    
    # Create a mapping of paths to embedding vectors
    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x, :]
    
    print("Facial feature coding calculation completed")
    return facial_encodings