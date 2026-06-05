import cv2
from typing import Tuple, Optional

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

class FaceDetector:
    def __init__(self, cascade_path: str = HAAR_CASCADE_PATH) -> None:
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image: "np.ndarray") -> list[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return faces.tolist() if len(faces) > 0 else []

    def crop_face(self, image: "np.ndarray", face_box: Tuple[int, int, int, int]) -> "np.ndarray":
        x, y, w, h = face_box
        return image[y:y + h, x:x + w]
