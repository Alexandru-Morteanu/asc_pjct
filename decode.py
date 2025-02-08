import cv2
import numpy as np
import matplotlib.pyplot as plt
import reedsolo

# Load the image
image = cv2.imread('qr.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_matrix = np.ones_like(gray, dtype=np.uint8)
cv2.drawContours(contour_matrix, contours, -1, (0), thickness=cv2.FILLED)

rows = np.any(contour_matrix == 0, axis=1)
cols = np.any(contour_matrix == 0, axis=0)

y_min, y_max = np.where(rows)[0][[0, -1]]
x_min, x_max = np.where(cols)[0][[0, -1]]

if y_max - y_min > 1 and x_max - x_min > 1:
    y_min += 1
    y_max -= 1
    x_min += 1
    x_max -= 1

cropped_matrix = contour_matrix[y_min:y_max+1, x_min:x_max+1]

min_size = float('inf')
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w == h and w > 1:
        min_size = min(min_size, w)

if min_size != float('inf'):

    h, w = cropped_matrix.shape

    rows_count = h // min_size
    cols_count = w // min_size

    resized_image = cv2.resize(cropped_matrix, (cols_count * min_size, rows_count * min_size))

    grid_matrix = np.zeros((rows_count, cols_count), dtype=int)

    for row in range(rows_count):
        for col in range(cols_count):
            start_x = col * min_size
            start_y = row * min_size
            end_x = (col + 1) * min_size
            end_y = (row + 1) * min_size
            cell = resized_image[start_y:end_y, start_x:end_x]

            avg_intensity = np.mean(cell)
            if avg_intensity < 0.5:
                grid_matrix[row, col] = 1  # Black
            else:
                grid_matrix[row, col] = 0  # White
type_information_bits = {
    ('L', '0'): np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]),
    ('L', '1'): np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]),
    ('L', '2'): np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]),
    ('L', '3'): np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1]),
    ('L', '4'): np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]),
    ('L', '5'): np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]),
    ('L', '6'): np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
    ('L', '7'): np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]),
    ('M', '0'): np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
    ('M', '1'): np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]),
    ('M', '2'): np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]),
    ('M', '3'): np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1]),
    ('M', '4'): np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]),
    ('M', '5'): np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]),
    ('M', '6'): np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]),
    ('M', '7'): np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]),
    ('Q', '0'): np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]),
    ('Q', '1'): np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]),
    ('Q', '2'): np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]),
    ('Q', '3'): np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0]),
    ('Q', '4'): np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0]),
    ('Q', '5'): np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]),
    ('Q', '6'): np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]),
    ('Q', '7'): np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]),
    ('H', '0'): np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]),
    ('H', '1'): np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]),
    ('H', '2'): np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]),
    ('H', '3'): np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    ('H', '4'): np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0]),
    ('H', '5'): np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]),
    ('H', '6'): np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
    ('H', '7'): np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
}

first5 = grid_matrix[8, 0:5]
def find_tuple_by_bits(input_bits, type_information_bits):
    for key, value in type_information_bits.items():
        if np.array_equal(value[:5], input_bits):
            return key
    return None

result = find_tuple_by_bits(first5, type_information_bits)


size = len(grid_matrix)
version = int((size - 21)/4) + 1
mat_for_mask = np.full((size, size), -1)

print(size, version)
# Function to place a QR finder pattern
def place_qr_finder(matrix, x, y):
    matrix[x:x+7, y:y+7] = 1  # 7x7 black border
    matrix[x+1:x+6, y+1:y+6] = 0  # 5x5 white
    matrix[x+2:x+5, y+2:y+5] = 1  # 3x3 black
    matrix[x:x+8, y+7:y+8] = 0
    matrix[x+7:x+8, y:y+8] = 0

def place_mini_qr_finder(matrix, x, y):
    matrix[x:x+5, y:y+5] = 1  # 7x7 black border
    matrix[x+1:x+4, y+1:y+4] = 0  # 5x5 white
    matrix[x+2, y+2] = 1  # 3x3 black

# Place the 3 finder patterns
place_qr_finder(mat_for_mask, 0, 0)
place_qr_finder(mat_for_mask, 0, size-7)
place_qr_finder(mat_for_mask, size-7, 0)

if version > 1:
    place_mini_qr_finder(mat_for_mask, size-9, size-9)

mat_for_mask[size-7-1, 0:0+8] = 0
mat_for_mask[0:0+8, size-7-1] = 0
for j in range(8, size-8):
    mat_for_mask[6, j] = 1 - (j % 2)

for i in range(8, size-8):
    mat_for_mask[i, 6] = 1 - (i % 2)

for i in range(8):
    mat_for_mask[8, i if i < 6 else i + 1] = 0
    mat_for_mask[8, size - 8 + i] = 0

for i in range(8):
    mat_for_mask[size - 1 - i if i < 7 else i + 1, 8] = 0
    mat_for_mask[0 + i if i != 6 else i + 1 , 8] = 0
mat_for_mask[size - 8, 8] = 1
def mask_bit(matrix, mask_num, mat_for_mask):
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            if mat_for_mask[row, col] == -1:
                if mask_num == "0":
                    matrix[row][col] ^= (row + col) % 2 == 0
                elif mask_num == "1":
                    matrix[row][col] ^= row % 2 == 0
                elif mask_num == "2":
                    matrix[row][col] ^= col % 3 == 0
                elif mask_num == "3":
                    matrix[row][col] ^= (row + col) % 3 == 0
                elif mask_num == "4":
                    matrix[row][col] ^= (np.floor(row / 2) + np.floor(col / 3)) % 2 == 0
                elif mask_num == "5":
                    matrix[row][col] ^= ((row * col) % 2) + ((row * col) % 3) == 0
                elif mask_num == "6":
                    matrix[row][col] ^= (((row * col) % 2) + ((row * col) % 3)) % 2 == 0
                elif mask_num == "7":
                    matrix[row][col] ^= (((row + col) % 2) + ((row * col) % 3)) % 2 == 0
    return matrix
grid_matrix = mask_bit(grid_matrix, result[1], mat_for_mask)

def extract_binary_from_qr(matrix, mask, size):
    binary_data = []
    col = size - 1
    row = size - 1

    while col >= 0:
        if col == 6:
            col -= 1
        
        while row >= 0 and col >= 0:
            if mask[row, col] == -1 and matrix[row, col] in [0, 1]:
                binary_data.append(str(matrix[row, col]))
            if col > 0 and mask[row, col - 1] == -1 and matrix[row, col - 1] in [0, 1]:
                binary_data.append(str(matrix[row, col - 1]))
            row -= 1

        col -= 2 
        row += 1

        if col < 0:
            break

        if col == 6:
            col -= 1

        while row < size and col >= 0:
            if mask[row, col] == -1 and matrix[row, col] in [0, 1]:
                binary_data.append(str(matrix[row, col]))
            if col > 0 and mask[row, col - 1] == -1 and matrix[row, col - 1] in [0, 1]:
                binary_data.append(str(matrix[row, col - 1]))
            row += 1 

        col -= 2 
        row -= 1

    return "".join(binary_data)

binary_sequence = extract_binary_from_qr(grid_matrix, mat_for_mask, size)
print(binary_sequence)

# def binary_to_bytes(binary_string):
#     """Convert a binary string to a list of 8-bit bytes (integers)."""
#     return [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]

# # Example binary sequence (replace with your actual binary data)
# binary_sequence = binary_sequence

# # Extract Data and ECC Codewords
# data_bin = binary_sequence[:19*8]  # First 19 codewords (19 bytes * 8 bits)
# ecc_bin = binary_sequence[19*8:]   # Last 7 codewords (7 bytes * 8 bits)

# # Convert binary to byte values
# data_codewords = binary_to_bytes(data_bin)
# ecc_codewords = binary_to_bytes(ecc_bin)

# # Convert to hexadecimal for readability
# data_hex = [hex(b) for b in data_codewords]
# ecc_hex = [hex(b) for b in ecc_codewords]

# # Print results
# print("🔹 Data Codewords (Hex):", data_hex)
# print("🔹 ECC Codewords (Hex):", ecc_hex)
# # Convert Data Codewords to ASCII
# ascii_text = ''.join(chr(b) for b in data_codewords if 32 <= b <= 126)
# print("Decoded ASCII:", ascii_text)
# rsc = reedsolo.RSCodec(7)  # 7 ECC symbols

# Given Data and ECC Codewords (Hex format)
import reedsolo

# Provided Data and ECC Codewords (Hex format)
data_codewords_hex = ['0x41', '0x14', '0x86', '0x56', '0xc6', '0xc6', '0xf2', '0xc2', 
                      '0x7', '0x76', '0xf7', '0x26', '0xc6', '0x42', '0x12', '0x3', 
                      '0x13', '0x23', '0x30']
ecc_codewords_hex = ['0x85', '0xa9', '0x5e', '0x7', '0xa', '0x36', '0xc9']

# Convert hex values to bytes
data_codewords = bytes([int(x, 16) for x in data_codewords_hex])
ecc_codewords = bytes([int(x, 16) for x in ecc_codewords_hex])

# Combine Data and ECC Codewords to simulate a transmission
encoded_message = data_codewords + ecc_codewords

# Initialize Reed-Solomon codec (matching ECC length)
rs = reedsolo.RSCodec(len(ecc_codewords))

# Attempt to decode and repair the binary data
try:
    # The rs.decode function will correct any errors in the data and ECC codewords
    repaired_data = rs.decode(encoded_message)[0]
    repaired_binary = bytes(repaired_data)  # Get the repaired binary data
    repaired_binary = ''.join(format(byte, '08b') for byte in repaired_data)
    print("Repaired Binary Data:", repaired_binary)
except reedsolo.ReedSolomonError as e:
    print("Decoding failed:", e)



# try:
#     # Decode the message
#     decoded_data = rs.decode(byte_array)  # Returns a tuple
#     decoded_bytes = bytes(decoded_data)   # Convert tuple to bytes
#     print("Decoded Data:", decoded_bytes.decode("utf-8"))  # Convert to UTF-8 string
# except reedsolo.ReedSolomonError as e:
#     print("Decoding failed:", e)




# plt.imshow(grid_matrix, cmap='gray_r', interpolation='nearest')
# plt.show()




