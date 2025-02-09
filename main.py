import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import faiss

# Define a very strict similarity threshold (values close to 1.0 are very strict)
HIGH_SIMILARITY_THRESHOLD = 0.9

def build_faiss_index(embeddings):
    """
    Build and return a Faiss index from a list of embeddings using cosine similarity.
    For normalized embeddings, inner product is equivalent to cosine similarity.
    """
    if not embeddings:
        raise ValueError("Embeddings list is empty, cannot build Faiss index.")

    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index

def extract_embeddings(model, image_path):
    """
    Extract normalized embeddings for faces in the image using the InsightFace model.
    Returns both the raw face detections (with bounding boxes) and the normalized embeddings.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return [], []

    faces = model.get(img)  # Detect faces and extract embeddings
    embeddings = []

    for face in faces:
        bbox = face.bbox
        print(f"Detected face bounding box: {bbox}")  # Debug: show bounding box
        cropped_face = preprocess_face(img, bbox)
        if cropped_face is None:
            continue

        # Ensure the embedding is normalized.
        embedding = face.embedding
        normalized_embedding = normalize(embedding.reshape(1, -1))[0]
        embeddings.append(normalized_embedding)
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

def find_closest_face(embedding, index, profiles, threshold=HIGH_SIMILARITY_THRESHOLD):
    """
    Find the closest known face for a given query embedding.
    Returns the matched profile name and similarity if the similarity is above the threshold,
    otherwise returns (None, None).
    """
    # Search for the top 1 nearest neighbor
    distances, indices = index.search(np.array([embedding]).astype('float32'), 1)
    closest_idx = indices[0][0]
    closest_similarity = distances[0][0]  # For normalized embeddings, this is the cosine similarity

    if closest_similarity > threshold:
        return profiles[closest_idx], closest_similarity
    return None, None

def main():
    known_faces_dir = "known_faces"
    query_faces_dir = "query_faces"

    os.makedirs(known_faces_dir, exist_ok=True)
    os.makedirs(query_faces_dir, exist_ok=True)

    print("Loading Face Recognition Model from local directory...")
    # Initialize the InsightFace model
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU if available

    print("Processing known faces...")
    known_embeddings = []
    profile_names = []

    # Process known faces and extract embeddings
    for filename in os.listdir(known_faces_dir):
        filepath = os.path.join(known_faces_dir, filename)
        if os.path.isfile(filepath):
            name, _ = os.path.splitext(filename)
            faces, embeddings = extract_embeddings(model, filepath)
            if embeddings:
                known_embeddings.extend(embeddings)
                # Assign the same name to each detected face in the image.
                profile_names.extend([name] * len(embeddings))

    if not known_embeddings:
        print("No known faces found. Add images to the 'known_faces' directory.")
        return

    print("Building Faiss index with cosine similarity...")
    index = build_faiss_index(known_embeddings)

    print("Processing query (group) faces...")
    # Process query (group) images and compare each detected face with known embeddings
    for filename in os.listdir(query_faces_dir):
        filepath = os.path.join(query_faces_dir, filename)
        if os.path.isfile(filepath):
            print(f"\nProcessing {filename}...")
            faces, embeddings = extract_embeddings(model, filepath)
            if not embeddings:
                print(f"No faces found in {filename}. Skipping...")
                continue

            # Iterate over each detected face in the query image
            for i, embedding in enumerate(embeddings):
                name, similarity = find_closest_face(embedding, index, profile_names, threshold=HIGH_SIMILARITY_THRESHOLD)
                if name:
                    print(f"Face {i+1}: Match found: {name} (Similarity: {similarity:.4f})")
                else:
                    print(f"Face {i+1}: No match found.")

if __name__ == "__main__":
    main()
