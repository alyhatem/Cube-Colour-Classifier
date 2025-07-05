import cv2
import numpy as np

def process_cube_face(image_path):
    img = cv2.imread(image_path)
    resized = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 60)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=4)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_contours = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        area = w * h
        if 0.8 < aspect_ratio < 1.2 and 1000 < area < 10000:
            box = cv2.boxPoints(rect).astype(int)
            square_contours.append(box)

    centers = []
    for c in square_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            centers.append((c, np.array([cx, cy])))

    if not centers:
        raise ValueError(f"No valid contours detected in image: {image_path}")

    all_centers = np.array([pt for _, pt in centers])
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

    def contour_center(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return np.array([0, 0])
        return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    contour_data = [(c, contour_center(c)) for c, _ in filtered]
    contour_data.sort(key=lambda x: x[1][1])  # sort by Y

    sorted_contours = []
    for i in range(0, len(contour_data), 3):
        row = contour_data[i:i+3]
        row.sort(key=lambda x: x[1][0])  # sort by X
        sorted_contours.extend([c for c, _ in row])

    mean_hsv_values = []
    mean_bgr_values = []
    for contour in sorted_contours:
        mask = np.zeros(resized.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_bgr = cv2.mean(resized, mask=mask)[:3]
        mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        mean_bgr_values.append(mean_bgr)
        mean_hsv_values.append(mean_hsv)

    # Draw filtered squares on image
    # filtered_contours = [c for c, _ in filtered]
    # cv2.drawContours(resized, filtered_contours, -1, (0, 0, 255), 2)

    # Display result
    # cv2.imshow("Detected Stickers", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return mean_bgr_values
    return mean_hsv_values


def classify_stickers(stickers_hsv: np.ndarray) -> str:
    """
    stickers_hsv: 6x9x3 NumPy array of HSV values ordered for Kociemba input.
    Returns a 54-character string of facelet colors: e.g., 'UUUUUUUUURRRRRRRRR...'
    """
    # Define standard Kociemba color letters in the order of faces: U, R, F, D, L, B
    face_colors = ['U', 'R', 'F', 'D', 'L', 'B']

    # Extract center colors (shape 6x3)
    centers = stickers_hsv[:, 4, :]
    print(centers)

    # Prepare output
    result = []

    # For each face
    for face_idx in range(6):
        for sticker_idx in range(9):
            hsv = stickers_hsv[face_idx, sticker_idx, :]
            # Compute distances to all centers
            dists = np.linalg.norm(centers - hsv, axis=1)
            nearest_face_idx = np.argmin(dists)
            result.append(face_colors[nearest_face_idx])

    # Return as string suitable for Kociemba solver
    return ''.join(result)



list_of_image_paths = [f"batch{i}.jpg" for i in range(1, 7)]
faces_data = []
for path in list_of_image_paths:
    face_colors = process_cube_face(path)
    # print(f"Face: {face_colors}")
    faces_data.append(face_colors)
faces_data = np.array(faces_data, dtype=np.uint8)

cube_string = classify_stickers(faces_data)
print(cube_string)
