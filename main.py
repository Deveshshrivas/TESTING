import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog

def select_query_image():
    """Open a file dialog for the user to select a query image."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    return file_path

def build_faiss_index(embeddings):
    """
    Build and return a Faiss index from a list of embeddings using cosine similarity.
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty, cannot build Faiss index.")
    
    embeddings_array = np.array(embeddings).astype('float32')
    # Since the embeddings are normalized, using inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index

def extract_embeddings(model, image_path):
    """
    Extract normalized embeddings for faces in the image using the InsightFace model.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return [], []
    
    # Detect faces and extract embeddings
    faces = model.get(img)
    embeddings = []

    for face in faces:
        bbox = face.bbox
        print(f"Detected face bounding box: {bbox}")  # Debug output
        cropped_face = preprocess_face(img, bbox)
        if cropped_face is None:
            continue
        
        embedding = face.embedding
        # Normalize the embedding
        embeddings.append(normalize(embedding.reshape(1, -1))[0])
    
    return faces, embeddings

def preprocess_face(image, bbox):
    """
    Crop and preprocess the face region from the image using the provided bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (112, 112))  # Resize to the expected size

def find_closest_face(embedding, index, profiles, threshold=0.6):
    """
    Find the closest match in the Faiss index using cosine similarity.
    """
    distances, indices = index.search(np.array([embedding]).astype('float32'), 1)
    closest_idx = indices[0][0]
    closest_distance = distances[0][0]

    # If the similarity is greater than the threshold, consider it a match.
    if closest_distance > threshold:
        return profiles[closest_idx], closest_distance
    return None, None

def main():
    known_faces_dir = "known_faces"
    os.makedirs(known_faces_dir, exist_ok=True)

    print("Loading Face Recognition Model from local directory...")
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU if available; otherwise, falls back to CPU

    print("Processing known faces...")
    known_embeddings = []
    profile_names = []

    # Process each image in the known_faces directory
    for filename in os.listdir(known_faces_dir):
        filepath = os.path.join(known_faces_dir, filename)
        if os.path.isfile(filepath):
            name, _ = os.path.splitext(filename)
            faces, embeddings = extract_embeddings(model, filepath)
            if embeddings:
                known_embeddings.extend(embeddings)
                profile_names.extend([name] * len(embeddings))

    if not known_embeddings:
        print("No known faces found. Add images to the 'known_faces' directory.")
        return

    print("Building Faiss index with cosine similarity...")
    index = build_faiss_index(known_embeddings)

    print("Please select a query image for comparison...")
    query_image_path = select_query_image()
    if not query_image_path:
        print("No image was selected. Exiting.")
        return

    print(f"Processing query image: {query_image_path}...")
    faces, embeddings = extract_embeddings(model, query_image_path)
    if not embeddings:
        print("No faces found in the query image.")
        return

    print("Matching each detected face with known faces:")
    # For each detected face in the query image, find and print its best match
    for i, embedding in enumerate(embeddings):
        name, similarity = find_closest_face(embedding, index, profile_names, threshold=0.4)
        print(f"Face {i + 1}:")
        if name:
            print(f"  Match found: {name} (Similarity: {similarity:.4f})")
        else:
            print("  No match found for this face.")

if __name__ == "__main__":
    main()
