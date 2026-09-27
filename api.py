import os
import tempfile
import numpy as np
import face_recognition

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(
    title="Facial Recognition API",
    description="API for facial recognition and 68-point facial landmarks",
    version="1.0.0"
)

DATASET_PATH = "dataset"
MATCH_THRESHOLD = 0.50


def load_known_faces():
    known_encodings = []
    known_names = []

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset folder not found.")

    for person in os.listdir(DATASET_PATH):

        person_folder = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):

            image_path = os.path.join(
                person_folder,
                image_name
            )

            try:
                image = face_recognition.load_image_file(image_path)

                locations = face_recognition.face_locations(image)

                encodings = face_recognition.face_encodings(
                    image,
                    locations
                )

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person)

            except Exception as e:
                print(
                    f"Error processing {image_path}: {e}"
                )

    return known_encodings, known_names


known_encodings, known_names = load_known_faces()


@app.get("/")
def home():
    return {
        "message": "Facial Recognition API is running",
        "endpoint": "/recognize",
        "method": "POST"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "known_face_samples": len(known_encodings),
        "people": sorted(set(known_names))
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_data = await file.read()

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temporary_file.write(image_data)
    temporary_file.close()

    try:

        image = face_recognition.load_image_file(
            temporary_file.name
        )

        face_locations = face_recognition.face_locations(
            image
        )

        face_encodings = face_recognition.face_encodings(
            image,
            face_locations
        )

        facial_landmarks = face_recognition.face_landmarks(
            image,
            face_locations
        )

        results = []

        for index, (encoding, location) in enumerate(
            zip(face_encodings, face_locations)
        ):

            if len(known_encodings) == 0:
                name = "Unknown"
                distance = None

            else:

                distances = face_recognition.face_distance(
                    known_encodings,
                    encoding
                )

                best_match_index = np.argmin(distances)

                distance = float(
                    distances[best_match_index]
                )

                if distance < MATCH_THRESHOLD:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"

            top, right, bottom, left = location

            landmark_data = {}

            if index < len(facial_landmarks):
                for region, points in facial_landmarks[index].items():
                    landmark_data[region] = [
                        {
                            "x": int(x),
                            "y": int(y)
                        }
                        for x, y in points
                    ]

            results.append(
                {
                    "person": name,
                    "face_distance": distance,
                    "face_location": {
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left)
                    },
                    "landmarks": landmark_data
                }
            )

        return {
            "filename": file.filename,
            "faces_detected": len(face_locations),
            "results": results
        }

    finally:

        if os.path.exists(temporary_file.name):
            os.remove(temporary_file.name)
