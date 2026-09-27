# Facial Recognition API Documentation

## 1. Project Overview

This project performs:
- Face detection
- Facial recognition
- Face encoding
- Face comparison
- 68-point facial landmark detection

The dataset contains five known personalities:
- Imran Khan
- Cristiano Ronaldo
- Lionel Messi
- Elon Musk
- Bill Gates

## 2. Technologies Used

- Python
- FastAPI
- face_recognition
- NumPy
- OpenCV
- Pillow
- Uvicorn

## 3. Code Flow

The system follows this process:

Image Upload
↓
Face Detection
↓
Face Encoding
↓
Face Distance Calculation
↓
Best Matching Face
↓
Recognition Result
↓
Facial Landmark Detection
↓
JSON Response

## 4. Face Detection

The API uses:

face_recognition.face_locations()

This detects the location of faces in the uploaded image.

## 5. Face Encoding

The API uses:

face_recognition.face_encodings()

The detected face is converted into a numerical face encoding.

## 6. Face Recognition

The API compares the uploaded face encoding with the known face encodings using:

face_recognition.face_distance()

The smallest distance is selected as the closest match.

A threshold of 0.50 is used.

If the distance is below 0.50, the person is identified.

Otherwise, the result is "Unknown".

## 7. Facial Landmarks

The API uses:

face_recognition.face_landmarks()

This extracts facial landmark points for regions such as:

- Chin
- Left eyebrow
- Right eyebrow
- Nose
- Left eye
- Right eye
- Mouth

## 8. API Endpoints

### GET /

Checks that the API is running.

### GET /health

Returns the API health status and information about the known face samples.

### POST /recognize

Accepts an image and performs:
- Face detection
- Face recognition
- Face landmark detection

The endpoint returns the results in JSON format.

## 9. Recognition Process

When an image is uploaded:

1. The API receives the image.
2. The image is temporarily stored.
3. Faces are detected.
4. Face encodings are generated.
5. The encodings are compared with the dataset.
6. The closest match is selected.
7. The 0.50 threshold is applied.
8. The person is identified or marked as Unknown.
9. Facial landmarks are extracted.
10. Results are returned as JSON.
11. The temporary image is deleted.

## 10. Testing

After deployment, the API can be tested through:

/docs

The user can open the `/recognize` endpoint, click "Try it out", upload an image, and click "Execute".

## 11. Limitations

Recognition performance can be affected by:

- Image quality
- Lighting
- Face angle
- Image resolution
- Quality and quantity of dataset images

This project is intended as a facial-recognition demonstration.
