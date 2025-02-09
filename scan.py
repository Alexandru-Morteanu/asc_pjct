import cv2
import reedsolo
import numpy as np
import matplotlib.pyplot as plt
import reedsolo

image = cv2.imread('qralpha.png')
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
                grid_matrix[row, col] = 1
            else:
                grid_matrix[row, col] = 0
grid_matrix = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
]
)

mask=[]
for i in range(0,9):
    if i != 6:
        mask.append(int(grid_matrix[8,i]))

for i in range(7,-1,-1):
    if i != 6:
        mask.append(int(grid_matrix[i,8]))

def hamming_distance(arr1, arr2):
    return np.sum(arr1 != arr2)

def find_closest_format(input_bits):
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
    input_bits = np.array(input_bits)
    min_distance = float('inf')
    best_match = None
    
    for key, bits in type_information_bits.items():
        distance = hamming_distance(input_bits, bits)
        if distance < min_distance:
            min_distance = distance
            best_match = key
    
    return best_match, min_distance

closest_match, distance = find_closest_format(mask)
ec_num, mask_num = closest_match
print(f"Closest match: {ec_num}, {mask_num}, Hamming distance: {distance}")

size = len(grid_matrix)
version = int((size - 21)/4) + 1
mat_for_mask = np.full((size, size), -1)

def place_qr_finder(matrix, x, y):
    matrix[x:x+7, y:y+7] = 1
    matrix[x+1:x+6, y+1:y+6] = 0
    matrix[x+2:x+5, y+2:y+5] = 1
    matrix[x:x+8, y+7:y+8] = 0
    matrix[x+7:x+8, y:y+8] = 0

def place_mini_qr_finder(matrix, x, y):
    matrix[x:x+5, y:y+5] = 1
    matrix[x+1:x+4, y+1:y+4] = 0
    matrix[x+2, y+2] = 1

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
def mask_matrix(matrix, mask_num, mat_for_mask):
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

grid_matrix = mask_matrix(grid_matrix, mask_num, mat_for_mask)

plt.imshow(grid_matrix, cmap='gray_r', interpolation='nearest')
plt.show()


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

target_lengths = {
    1: {"L": 19, "M": 16, "Q": 13, "H": 9},
    2: {"L": 34, "M": 28, "Q": 22, "H": 16},
    3: {"L": 55, "M": 44, "Q": 34, "H": 26},
    4: {"L": 80, "M": 64, "Q": 48, "H": 36},
    5: {"L": 108, "M": 86, "Q": 62, "H": 46},
    6: {"L": 136, "M": 108, "Q": 76, "H": 60},
    7: {"L": 156, "M": 124, "Q": 88, "H": 66},
    8: {"L": 194, "M": 154, "Q": 110, "H": 86},
    9: {"L": 232, "M": 182, "Q": 132, "H": 100},
    10: {"L": 274, "M": 216, "Q": 154, "H": 122},
    11: {"L": 324, "M": 254, "Q": 180, "H": 140},
    12: {"L": 370, "M": 290, "Q": 206, "H": 158},
    13: {"L": 428, "M": 334, "Q": 244, "H": 180},
    14: {"L": 461, "M": 365, "Q": 261, "H": 197},
    15: {"L": 523, "M": 415, "Q": 295, "H": 223},
    16: {"L": 589, "M": 453, "Q": 325, "H": 253},
    17: {"L": 647, "M": 507, "Q": 367, "H": 283},
    18: {"L": 721, "M": 563, "Q": 397, "H": 313},
    19: {"L": 795, "M": 627, "Q": 445, "H": 341},
    20: {"L": 861, "M": 669, "Q": 485, "H": 385},
    21: {"L": 932, "M": 714, "Q": 512, "H": 406},
    22: {"L": 1006, "M": 782, "Q": 568, "H": 442},
    23: {"L": 1094, "M": 860, "Q": 614, "H": 464},
    24: {"L": 1174, "M": 914, "Q": 664, "H": 514},
    25: {"L": 1276, "M": 1000, "Q": 718, "H": 538},
    26: {"L": 1370, "M": 1062, "Q": 754, "H": 596},
    27: {"L": 1468, "M": 1128, "Q": 808, "H": 628},
    28: {"L": 1531, "M": 1193, "Q": 871, "H": 661},
    29: {"L": 1631, "M": 1267, "Q": 911, "H": 701},
    30: {"L": 1735, "M": 1373, "Q": 985, "H": 745},
    31: {"L": 1843, "M": 1455, "Q": 1033, "H": 793},
    32: {"L": 1955, "M": 1541, "Q": 1115, "H": 845},
    33: {"L": 2071, "M": 1631, "Q": 1171, "H": 901},
    34: {"L": 2191, "M": 1725, "Q": 1231, "H": 961},
    35: {"L": 2306, "M": 1812, "Q": 1286, "H": 986},
    36: {"L": 2434, "M": 1914, "Q": 1354, "H": 1054},
    37: {"L": 2566, "M": 1992, "Q": 1426, "H": 1096},
    38: {"L": 2702, "M": 2102, "Q": 1502, "H": 1142},
    39: {"L": 2812, "M": 2216, "Q": 1582, "H": 1222},
    40: {"L": 2956, "M": 2334, "Q": 1666, "H": 1276}
}

binary_sequence = extract_binary_from_qr(grid_matrix, mat_for_mask, size)
print(binary_sequence)
if version >= 2:
    binary_sequence = binary_sequence[:-7]


def binary_to_bytes(binary_string):
    """Convert a binary string to a list of 8-bit bytes (integers)."""
    return [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]

data_bin = binary_sequence[:target_lengths[version][ec_num]*8]
ecc_bin = binary_sequence[target_lengths[version][ec_num]*8:] 

data_codewords = binary_to_bytes(data_bin)
ecc_codewords = binary_to_bytes(ecc_bin)

data_hex = [hex(b) for b in data_codewords]
ecc_hex = [hex(b) for b in ecc_codewords]
print("sequence",binary_sequence)
encoded_message = data_codewords + ecc_codewords
print(ecc_codewords)
print("length of ecc",len(ecc_codewords))
print(encoded_message)
print("length of messahe",len(encoded_message))

rs = reedsolo.RSCodec(len(ecc_codewords))

try:
    print("before")
    repaired_data = rs.decode(encoded_message)[0]
    print("after")
    repaired_binary_data = bytes(repaired_data)
    print("afterr")
    repaired_binary_data = ''.join(format(byte, '08b') for byte in repaired_data)
    print("Repaired Binary Data:", repaired_binary_data)
except reedsolo.ReedSolomonError as e:
    print("Decoding failed:", e)

mod_num = repaired_binary_data[:4]
char_count_bites = 1
if mod_num == "0001":
    char_count_bites = 10
    value_size = 10
elif mod_num == "0010":
    char_count_bites = 9
    value_size = 11
elif mod_num == "0100":
    char_count_bites = 8
    value_size = 8
elif mod_num == "1000":
    char_count_bites = 7
    value_size = 10
char_count = repaired_binary_data[4:4+char_count_bites]

repaired_binary_data = repaired_binary_data[4+char_count_bites:]

print("Mod",mod_num,"Char count bits",char_count)
char_count = int(char_count,2) #
print("Actual char count",char_count)

alpha_dict = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

#alfa - 11 2 char      (daca e numar impar, ultimul character e pe 6 biti)           
#numeric - 10 3 char      (daca e numar impar, ultimul character e pe 7 biti)
i = 0
result_string = ""
if value_size == 8:
    cnt = 0
    while i+value_size < len(repaired_binary_data) and cnt <= char_count:
        result_string += chr(binary_to_bytes(repaired_binary_data[i:i+value_size])[0])
        i+=value_size
        cnt+=1
elif value_size == 10:
    cnt = 0
    while i+value_size < len(repaired_binary_data) and cnt+3 <= char_count:
        result_string += str(int(repaired_binary_data[i:i+value_size],2))
        i+=value_size
        cnt+=3
    if char_count - cnt == 2:
        result_string+=str(int(repaired_binary_data[i:i+7],2))
    elif char_count - cnt == 1:
        result_string += str(int(repaired_binary_data[i:i+4],2))
elif value_size == 11:
    cnt = 0
    while i+value_size < len(repaired_binary_data) and cnt+2 <= char_count:
        result_string+=alpha_dict[int(repaired_binary_data[i:i+11],2)//45]+alpha_dict[int(repaired_binary_data[i:i+11],2)%45]
        i += value_size
        cnt += 2
    if char_count - cnt == 1:
        result_string += alpha_dict[int(repaired_binary_data[i:i+6],2)%45]

print(result_string)
