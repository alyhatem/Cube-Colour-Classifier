import cv2
import numpy as np

class CubeFaceProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.resized = None
        self.square_contours = []
        self.sorted_contours = []
        self.mean_lab_values = []

    @staticmethod
    def contour_center(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return np.array([0, 0])
        return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    def read_and_preprocess(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        self.resized = cv2.resize(self.image, (480, 640), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 60)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=4)
        return dilated

    def detect_squares(self, dilated):
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            area = w * h
            if 0.8 < aspect_ratio < 1.2 and 1000 < area < 10000:
                box = cv2.boxPoints(rect).astype(int)
                self.square_contours.append(box)

        if not self.square_contours:
            raise ValueError(f"No valid contours detected in image: {self.image_path}")

    def select_and_sort_contours(self):
        centers = [
            (c, self.contour_center(c))
            for c in self.square_contours
            if cv2.moments(c)["m00"] != 0
        ]
        all_centers = np.array([center for _, center in centers])
        avg_center = np.mean(all_centers, axis=0)
        distances = [(c, center, np.linalg.norm(center - avg_center)) for c, center in centers]
        distances.sort(key=lambda x: x[2])

        selected = distances[:9]
        areas = [cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1] for c, _, _ in selected]
        avg_area = np.mean(areas)
        filtered = [
            (c, center) for c, center, _ in selected
            if 0.5 * avg_area < cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1] < 1.5 * avg_area
        ]

        contour_data = [(c, self.contour_center(c)) for c, _ in filtered]
        contour_data.sort(key=lambda x: x[1][1])  # sort by Y

        self.sorted_contours.clear()
        for i in range(0, len(contour_data), 3):
            row = contour_data[i:i+3]
            row.sort(key=lambda x: x[1][0])  # sort by X
            self.sorted_contours.extend([c for c, _ in row])

    def compute_colors(self):
        self.mean_lab_values.clear()
        for contour in self.sorted_contours:
            mask = np.zeros(self.resized.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_bgr = cv2.mean(self.resized, mask=mask)[:3]
            # mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            mean_lab = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            # Convert to signed type first to avoid overflow
            lab = mean_lab.astype(np.int16)

            # Unscale channels
            L_true = lab[0] * 100 / 255.0
            A_true = lab[1] - 128
            B_true = lab[2] - 128

            true_lab = np.array([L_true, A_true, B_true], dtype=np.float32)
            self.mean_lab_values.append(true_lab)

    def process_image(self):
        dilated = self.read_and_preprocess()
        self.detect_squares(dilated)
        self.select_and_sort_contours()
        self.compute_colors()
        return self.mean_lab_values


# Example usage:
if __name__ == "__main__":
    list_of_image_paths = [f"Media/batch{i}.jpg" for i in range(1, 7)]
    faces_data = []
    for path in list_of_image_paths:
        processor = CubeFaceProcessor(path)
        face_colors = processor.process_image()
        # cv2.imshow("Img", cv2.drawContours(processor.resized, processor.sorted_contours, -1, (0, 0, 255), 2))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        faces_data.append(face_colors)

    faces_data = np.array(faces_data, dtype=np.int16)
    print(faces_data)
