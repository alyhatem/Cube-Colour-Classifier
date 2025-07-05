import numpy as np
import cv2
from ColourExtraction import CubeFaceProcessor


class CubeClassifier:
    """
    Classifies stickers from LAB colors to facelet letters based on calibrated centers.
    """
    calibrated_centers = np.array([
        [255, 128, 128],  # White
        [32, 150, 64],    # Blue
        [60, 200, 190],   # Red
        [230, 128, 200],  # Yellow
        [150, 80, 80],    # Green
        [120, 180, 170],  # Orange
    ], dtype=np.uint8)

    face_letters = ['W', 'B', 'R', 'Y', 'G', 'O']

    def classify(self, stickers_lab: np.ndarray) -> str:
        """
        stickers_lab: 6x9x3 NumPy array of LAB values.
        Returns: 54-character classification string.
        """
        result = []
        centers_ab = self.calibrated_centers[:, 1:3]
        for face_idx in range(6):
            for sticker_idx in range(9):
                lab = stickers_lab[face_idx, sticker_idx, :]
                lab_ab = lab[1:3]
                dists = np.linalg.norm(centers_ab - lab_ab, axis=1)
                nearest_face_idx = np.argmin(dists)
                result.append(self.face_letters[nearest_face_idx])
        return ''.join(result)


# USAGE
list_of_image_paths = [f"Media/batch{i}.jpg" for i in range(1, 7)]

# Extract LAB color data for each face using CubeFaceProcessor
faces_data = []
for path in list_of_image_paths:
    processor = CubeFaceProcessor(path)
    face_colors = processor.process_image()
    faces_data.append(face_colors)
faces_data = np.array(faces_data, dtype=np.uint8)

# Classify stickers into facelet letters
classifier = CubeClassifier()
cube_string = classifier.classify(faces_data)

# Print cube string per face
for i, face in enumerate(CubeClassifier.face_letters):
    face_stickers = cube_string[i*9:(i+1)*9]
    print(f"{face}: {face_stickers}")
