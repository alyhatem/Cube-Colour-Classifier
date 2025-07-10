import cv2
import numpy as np

# Load and preprocess image
image_path = "batch1.jpg"
img = cv2.imread(image_path)
resized = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blurred, 30, 60)

# Dilate edges to strengthen shapes
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=4)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter square-like contours
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

# Compute centers of detected squares
centers = []
for c in square_contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        centers.append((c, np.array([cx, cy])))

if not centers:
    raise ValueError("No valid contours detected.")

# Compute average center and distances
all_centers = np.array([pt for _, pt in centers])
avg_center = np.mean(all_centers, axis=0)
distances = [(c, center, np.linalg.norm(center - avg_center)) for c, center in centers]
distances.sort(key=lambda x: x[2])  # Closest first

# Select closest 9 contours and filter by area consistency
selected = distances[:9]
areas = [cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1] for c, _, _ in selected]
avg_area = np.mean(areas)
filtered = [
    (c, center) for c, center, _ in selected
    if 0.5 * avg_area < cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1] < 1.5 * avg_area
]

# Collect corners of filtered contours for convex hull
all_corners = []
for c, _ in filtered:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(int)
    all_corners.extend(box)
hull = cv2.convexHull(np.array(all_corners))

# Sort contours into 3x3 grid: top-to-bottom, then left-to-right in rows
def contour_center(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return np.array([0, 0])
    return np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

contour_data = [(c, contour_center(c)) for c in square_contours]
contour_data.sort(key=lambda x: x[1][1])  # sort by Y

sorted_contours = []
for i in range(0, len(contour_data), 3):
    row = contour_data[i:i+3]
    row.sort(key=lambda x: x[1][0])  # sort by X
    sorted_contours.extend([c for c, _ in row])

# Compute mean color in each sorted contour
for contour in sorted_contours:
    mask = np.zeros(resized.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_bgr = cv2.mean(resized, mask=mask)[:3]
    mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    # print(f"Mean BGR: {mean_bgr}")
    print(f"Mean HSV: {mean_hsv}")

# Draw convex hull and filtered squares on image
cv2.polylines(resized, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
filtered_contours = [c for c, _ in filtered]
cv2.drawContours(resized, filtered_contours, -1, (0, 0, 255), 2)

# Display result
cv2.imshow("Detected Stickers", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
