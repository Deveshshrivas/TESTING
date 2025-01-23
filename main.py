import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import faiss
from sklearn.metrics.pairwise import cosine_similarity

def build_faiss_index(embeddings):
    """
    Build and return a Faiss index from a list of embeddings using cosine similarity.
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty, cannot build Faiss index.")

    embeddings_array = np.array(embeddings).astype('float32')
    # Create the Faiss index for cosine similarity (use IP for inner product)
    index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Use inner product (cosine similarity)
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

    faces = model.get(img)  # Detect faces and extract embeddings
    embeddings = []

    for face in faces:
        bbox = face.bbox
        print(f"Detected face bounding box: {bbox}")  # Debugging bounding box
        cropped_face = preprocess_face(img, bbox)
        if cropped_face is None:
            continue

        embedding = face.embedding
        embeddings.append(normalize(embedding.reshape(1, -1))[0])  # Normalize embeddings

    return faces, embeddings

def preprocess_face(image, bbox):
    """
    Crop and preprocess face region from the image using the provided bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (112, 112))  # Resize to expected size

def find_closest_face(embedding, index, profiles, threshold=0.6):
    """
    Find the closest match in the Faiss index with a strict similarity threshold using cosine similarity.
    """
    distances, indices = index.search(np.array([embedding]).astype('float32'), 1)
    closest_idx = indices[0][0]
    closest_distance = distances[0][0]

    if closest_distance > threshold:  # Use higher threshold to be more lenient with partial faces
        return profiles[closest_idx], closest_distance  # Use cosine similarity directly
    return None, None

def find_best_match(embeddings, known_embeddings, profile_names, index, threshold=0.6):
    """
    Find the best match for a given query image by comparing all faces' embeddings.
    """
    best_name = None
    best_similarity = float('-inf')  # Initialize with negative value for similarity

    for embedding in embeddings:
        name, similarity = find_closest_face(embedding, index, profile_names, threshold)
        if similarity and similarity > best_similarity:
            best_name = name
            best_similarity = similarity

    return best_name, best_similarity

def main():
    known_faces_dir = "known_faces"
    query_faces_dir = "query_faces"

    os.makedirs(known_faces_dir, exist_ok=True)
    os.makedirs(query_faces_dir, exist_ok=True)

    print("Loading Face Recognition Model from local directory...")
    # Initialize the InsightFace model
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))  # Prepare model with GPU if available

    print("Processing known faces...")
    known_embeddings = []
    profile_names = []

    # Process the known faces and extract embeddings
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

    print("Processing query faces...")
    # Process the query faces and compare them with known embeddings
    for filename in os.listdir(query_faces_dir):
        filepath = os.path.join(query_faces_dir, filename)
        if os.path.isfile(filepath):
            print(f"Processing {filename}...")
            faces, embeddings = extract_embeddings(model, filepath)
            if not embeddings:
                print(f"No faces found in {filename}. Skipping...")
                continue

            # Find the best match for all detected faces in the query image
            name, similarity = find_best_match(embeddings, known_embeddings, profile_names, index, threshold=0.4)
            if name:
                print(f"Match found: {name} (Similarity: {similarity:.4f})")
            else:
                print("No match found for the key person in this image.")

if __name__ == "__main__":
    main()
