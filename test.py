from deepface import DeepFace
import cv2
import os

# Paths to images
known_faces_dir = "known_faces"  # Folder with known faces
unknown_faces_dir = "unknown_faces"  # Folder with unknown faces

# Load embeddings for known faces
print("Processing known faces...")
known_embeddings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    filepath = os.path.join(known_faces_dir, filename)
    try:
        embedding = DeepFace.represent(filepath, model_name="Facenet", enforce_detection=False)
        known_embeddings.append(embedding[0]["embedding"])
        known_names.append(os.path.splitext(filename)[0])
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Process unknown faces
print("Processing unknown faces...")
for filename in os.listdir(unknown_faces_dir):
    filepath = os.path.join(unknown_faces_dir, filename)
    try:
        # Detect and analyze face
        results = DeepFace.find(img_path=filepath, db_path=known_faces_dir, model_name="Facenet", enforce_detection=False, detector_backend="opencv")
        if len(results) > 0:
            print(f"Match found for {filename}: {results[0]}")
        else:
            print(f"No match found for {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
