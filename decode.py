import cv2
import numpy as np

# Load the image
image = cv2.imread('qr.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to get the black regions
_, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours, using RETR_TREE to detect both outer and inner contours
contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank matrix with 1's (representing the background as black)
contour_matrix = np.ones_like(gray, dtype=np.uint8)

# Draw the contours as 0's (representing contours as black)
cv2.drawContours(contour_matrix, contours, -1, (0), thickness=cv2.FILLED)

# Find the bounding box around the region of interest
rows = np.any(contour_matrix == 0, axis=1)  # Check rows with 0's
cols = np.any(contour_matrix == 0, axis=0)  # Check columns with 0's

# Get the coordinates of the bounding box
y_min, y_max = np.where(rows)[0][[0, -1]]
x_min, x_max = np.where(cols)[0][[0, -1]]

# Reduce the bounding box by 1 on each side if possible
if y_max - y_min > 1 and x_max - x_min > 1:
    y_min += 1
    y_max -= 1
    x_min += 1
    x_max -= 1

# Crop the matrix
cropped_matrix = contour_matrix[y_min:y_max+1, x_min:x_max+1]

# Find the smallest black square (smallest contour bounding box)
min_size = float('inf')
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w == h and w > 1:  # Ensure it's a square and has non-zero dimensions
        min_size = min(min_size, w)

# Create the grid
if min_size != float('inf'):

    h, w = cropped_matrix.shape

    # Calculate how many rows and columns fit in the bounding box based on the smallest square size
    rows_count = h // min_size
    cols_count = w // min_size

    # Resize the cropped matrix to fit into the grid
    resized_image = cv2.resize(cropped_matrix, (cols_count * min_size, rows_count * min_size))

    grid_matrix = np.zeros((rows_count, cols_count), dtype=int)

    # Reconstruct the grid with resized image
    for row in range(rows_count):
        for col in range(cols_count):
            start_x = col * min_size
            start_y = row * min_size
            end_x = (col + 1) * min_size
            end_y = (row + 1) * min_size
            cell = resized_image[start_y:end_y, start_x:end_x]

        # Calculate the average intensity to classify the cell as black or white
            avg_intensity = np.mean(cell)
            # print(avg_intensity)
            if avg_intensity < 0.5:  # Threshold for black
                grid_matrix[row, col] = 1  # Black
            else:
                grid_matrix[row, col] = 0  # White
            # Copy the respective section of the resized image
            # grid_matrix[start_y:end_y, start_x:end_x] = resized_image[start_y:end_y, start_x:end_x]
    # Overlay the grid on the cropped image to check fit
    grid_overlay = resized_image.copy()

    # Draw horizontal grid lines
    for y in range(0, h, min_size):
        cv2.line(grid_overlay, (0, y), (w, y), (127), 1)  # Gray grid lines

    # Draw vertical grid lines
    for x in range(0, w, min_size):
        cv2.line(grid_overlay, (x, 0), (x, h), (127), 1)  # Gray grid lines

    # print(grid_matrix)
    # # Visualize the grid overlay
    # print(grid_overlay)

    cv2.imshow('Grid Overla.y on Cropped Image', grid_overlay * 255)
def print_matrix(matrix):
    for row in matrix:
        print(' '.join(['█' if cell else ' ' for cell in row]))

print(grid_matrix)

mask = grid_matrix[8][2]*4 + grid_matrix[8][3]*2 + grid_matrix[8][4]*1
print(grid_matrix[8][2:5],mask)

cv2.waitKey(0)
cv2.destroyAllWindows()









