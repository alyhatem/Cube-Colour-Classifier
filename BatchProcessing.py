import cv2
import numpy as np

def hsv_to_lab(hsv):
    """
    hsv: (3,) NumPy array or list, e.g., [H, S, V] with H in [0,179], S,V in [0,255].
    Returns: (3,) LAB NumPy array.
    """
    hsv_reshaped = np.uint8([[hsv]])            # shape (1,1,3)
    bgr = cv2.cvtColor(hsv_reshaped, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0,0]

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
    mean_lab_values = []
    for contour in sorted_contours:
        mask = np.zeros(resized.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_bgr = cv2.mean(resized, mask=mask)[:3]
        mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        mean_bgr_values.append(mean_bgr)
        mean_hsv_values.append(mean_hsv)
        mean_lab = hsv_to_lab(mean_hsv)
        mean_lab_values.append(mean_lab)

    # Draw filtered squares on image
    # filtered_contours = [c for c, _ in filtered]
    # cv2.drawContours(resized, filtered_contours, -1, (0, 0, 255), 2)

    # Display result
    # cv2.imshow("Detected Stickers", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return mean_bgr_values
    # return mean_hsv_values
    return mean_lab_values

# Order: [W, B, R, Y, G, O] — match your face_colors order
calibrated_centers = np.array([
    [255, 128, 128],  # White: bright neutral
    [ 32, 150,  64],  # Blue: dark, more A/B towards blue
    [ 60, 200, 190],  # Red: medium-light, high A+B
    [230, 128, 200],  # Yellow: bright, high B
    [150,  80,  80],  # Green: mid, A/B towards green
    [120, 180, 170],  # Orange: between red/yellow
], dtype=np.uint8)

face_colors = ['W', 'B', 'R', 'Y', 'G', 'O'] # Change to ['U', 'R', 'F', 'D', 'L', 'B']

def classify_stickers_lab(stickers_lab: np.ndarray) -> str:
    """
    stickers_lab: 6x9x3 NumPy array of LAB values ordered for Kociemba input.
    Returns a 54-character string of facelet colors: e.g., 'UUUUUUUUURRRRRRRRR...'
    """
    result = []

    # Use fixed calibrated centers ignoring L (compare only A,B)
    centers_ab = calibrated_centers[:, 1:3]

    for face_idx in range(6):
        for sticker_idx in range(9):
            lab = stickers_lab[face_idx, sticker_idx, :]
            lab_ab = lab[1:3]
            dists = np.linalg.norm(centers_ab - lab_ab, axis=1)
            nearest_face_idx = np.argmin(dists)
            result.append(face_colors[nearest_face_idx])

    return ''.join(result)

def visualize_classification(image_paths, faces_data, cube_string):
    """
    image_paths: list of 6 image paths in the same order as faces_data.
    faces_data: 6x9x3 LAB array.
    cube_string: 54-character classification result.
    """
    face_colors = ['W', 'B', 'R', 'Y', 'G', 'O']

    for face_idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)

        # Recompute contours to get sticker positions (needed since process_cube_face doesn't store them)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 60)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=4)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            area = w * h
            if 0.8 < aspect_ratio < 1.2 and 1000 < area < 10000:
                box = cv2.boxPoints(rect).astype(int)
                M = cv2.moments(box)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    centers.append((box, np.array([cx, cy])))

        # Sort the detected centers into 3x3 grid order
        all_centers = np.array([pt for _, pt in centers])
        avg_center = np.mean(all_centers, axis=0)
        distances = [(c, center, np.linalg.norm(center - avg_center)) for c, center in centers]
        distances.sort(key=lambda x: x[2])
        selected = distances[:9]
        selected = [ (c, center) for c, center, _ in selected ]

        def contour_center(c):
            M = cv2.moments(c)
            if M["m00"] == 0:
                return np.array([0, 0])
            return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

        contour_data = [(c, contour_center(c)) for c, center in selected]
        contour_data.sort(key=lambda x: x[1][1])  # sort by Y

        sorted_centers = []
        for i in range(0, len(contour_data), 3):
            row = contour_data[i:i+3]
            row.sort(key=lambda x: x[1][0])  # sort by X
            sorted_centers.extend([center for _, center in row])

        # Draw classification letters on image
        face_stickers = cube_string[face_idx*9:(face_idx+1)*9]
        for i, center in enumerate(sorted_centers):
            label = face_stickers[i]
            cv2.putText(img, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow(f"Face {face_colors[face_idx]}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



list_of_image_paths = [f"Media/batch{i}.jpg" for i in range(1, 7)]
faces_data = []
for path in list_of_image_paths:
    face_color_values = process_cube_face(path)
    # print(f"Face: {face_colors}")
    faces_data.append(face_color_values)
faces_data = np.array(faces_data, dtype=np.uint8)

cube_string = classify_stickers_lab(faces_data)
# print(cube_string)
for i, face in enumerate(face_colors):
    face_stickers = cube_string[i*9:(i+1)*9]
    print(f"{face_colors[i]}: {face_stickers}")

visualize_classification(list_of_image_paths, faces_data, cube_string)
