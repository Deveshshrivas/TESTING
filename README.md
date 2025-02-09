# Face Recognition Using InsightFace and Faiss

This project demonstrates face recognition using the InsightFace library for facial analysis and Faiss for fast similarity search. The script processes images to extract and normalize face embeddings, builds a Faiss index for known faces, and then compares faces in query images to find the best match.

## Project Structure

- `known_faces/`: Directory containing images of known faces.
- `query_faces/`: Directory containing images of query faces for which matches need to be found.

## Requirements

- Python 3.x
- OpenCV
- InsightFace
- Faiss
- NumPy
- Scikit-learn

## Installation

1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install opencv-python-headless insightface faiss-cpu numpy scikit-learn
