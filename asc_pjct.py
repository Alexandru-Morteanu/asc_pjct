import math
import numpy as np
import matplotlib.pyplot as plt
import reedsolo
from matplotlib.widgets import TextBox
from PIL import Image, ImageDraw
num_ec_codewords = 0
mat=[]

def big_problem(input_string):
    global mat
    qr_capacity = {
        1: {"L": {0b1: 41, 0b10: 25, 0b100: 17, 0b1000: 10}, "M": {0b1: 34, 0b10: 20, 0b100: 14, 0b1000: 8}, 
            "Q": {0b1: 27, 0b10: 16, 0b100: 11, 0b1000: 7}, "H": {0b1: 17, 0b10: 10, 0b100: 7, 0b1000: 4}},
        2: {"L": {0b1: 77, 0b10: 47, 0b100: 32, 0b1000: 20}, "M": {0b1: 63, 0b10: 38, 0b100: 26, 0b1000: 16}, 
            "Q": {0b1: 48, 0b10: 29, 0b100: 20, 0b1000: 12}, "H": {0b1: 34, 0b10: 20, 0b100: 14, 0b1000: 8}},
        3: {"L": {0b1: 127, 0b10: 77, 0b100: 53, 0b1000: 32}, "M": {0b1: 101, 0b10: 61, 0b100: 42, 0b1000: 26}, 
            "Q": {0b1: 77, 0b10: 47, 0b100: 32, 0b1000: 20}, "H": {0b1: 58, 0b10: 35, 0b100: 24, 0b1000: 15}},
        4: {"L": {0b1: 187, 0b10: 114, 0b100: 78, 0b1000: 48}, "M": {0b1: 149, 0b10: 90, 0b100: 62, 0b1000: 38}, 
            "Q": {0b1: 111, 0b10: 67, 0b100: 46, 0b1000: 28}, "H": {0b1: 82, 0b10: 50, 0b100: 34, 0b1000: 21}},
        5: {"L": {0b1: 255, 0b10: 154, 0b100: 106, 0b1000: 65}, "M": {0b1: 202, 0b10: 122, 0b100: 84, 0b1000: 52}, 
            "Q": {0b1: 144, 0b10: 87, 0b100: 60, 0b1000: 37}, "H": {0b1: 106, 0b10: 64, 0b100: 44, 0b1000: 27}}
    }
    #prima e numeric dupa alfanumeric dupa byte

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

    ec_num     = "L"
    mask_num   = "3"
    tuplul = (ec_num, mask_num)

    #input_string=input("input")
    # input_string="Hello, world! 123"
    def check_input(input_string):
        if input_string.isdigit():
            return 0b0001  
        elif all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:" for c in input_string):  
            return 0b0010
        else:
            return 0b0100

    mod_num = check_input(input_string)

    char_count = len(input_string)

    def find_version():
        for current_version in range(1, len(qr_capacity)+1):
            if char_count <= qr_capacity[current_version][ec_num][mod_num]:
                return current_version
            

    version = find_version()

    size = 21+4*(version-1)
    mat = np.full((size, size), -1)

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
    place_qr_finder(mat, 0, 0)
    place_qr_finder(mat, 0, size-7)
    place_qr_finder(mat, size-7, 0)

    if version > 1:
        place_mini_qr_finder(mat, size-9, size-9)

    mat[size-7-1, 0:0+8] = 0
    mat[0:0+8, size-7-1] = 0


    for j in range(8, size-8):
        mat[6, j] = 1 - (j % 2)

    for i in range(8, size-8):
        mat[i, 6] = 1 - (i % 2)



    for i in range(8):
        mat[8, i if i < 6 else i + 1] = 0
        mat[8, size - 8 + i] = 0

    for i in range(8):
        mat[size - 1 - i if i < 7 else i + 1, 8] = 0
        mat[0 + i if i != 6 else i + 1 , 8] = 0
    mat[size - 8, 8] = 1

    mat_for_mask = mat.copy()
    print(mat_for_mask)

    binary_data=""

    # numeric
    if mod_num == 0b0001:
        binary_data = "0001"
        binary_data += format(len(input_string), '010b')
        l=len(input_string)
        for i in range(0, l//3*3, 3):
            binary_data+=(f"{int(input_string[i:i+3]):010b}")
        if l%3==1:
            binary_data+=(f"{int(input_string[-1]):04b}")
        if l%3==2:
            binary_data+=(f"{int(input_string[-2:]):07b}")

    # alpha
    # QR Alphanumeric Character Set Mapping
    ALPHANUMERIC_MAP = {char: i for i, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")}

    if mod_num == 0b0010:
        binary_data = "0010"
        binary_data += format(len(input_string), '09b')
        l = len(input_string)
        for i in range(0, l // 2 * 2, 2):
            c1, c2 = input_string[i], input_string[i+2-1]
            value = 45 * ALPHANUMERIC_MAP[c1] + ALPHANUMERIC_MAP[c2]
            binary_data += format(value, '011b')
        if l % 2 == 1:
            binary_data += format(ALPHANUMERIC_MAP[input_string[-1]], '06b')


    # byte
    if mod_num == 0b0100:
        binary_data="0100"
        binary_data += format(len(input_string), '08b')
        l=len(input_string)
        binary_data += ''.join(format(ord(char), '08b') for char in input_string)
    binary_data += "0000"
    binary_data += "0"* ((8-len(binary_data)%8) if len(binary_data)%8!=0 else 0)



    def fill_qr_matrix(matrix, size, binary_data):
        number = 0  # Start numbering from 0 (for binary data index)
        col = size - 1  # Start from the rightmost column
        row = size - 1  # Start from the bottom row

        while col >= 0:
            if col == 6:  # Skip the vertical timing pattern at column 6
                col -= 1  # Move an extra step left to avoid column 6
            
            # Zigzag upwards
            while row >= 0 and col >= 0:
                if matrix[row, col] == -1 and number < len(binary_data):
                    matrix[row, col] = int(binary_data[number])
                    number += 1
                if col > 0 and matrix[row, col - 1] == -1 and number < len(binary_data):
                    matrix[row, col - 1] = int(binary_data[number])
                    number += 1
                row -= 1  # Move UP

            col -= 2  # Move left by two columns
            row += 1  # Adjust row after reaching the top

            if col < 0:  # Prevent out-of-bounds
                break

            if col == 6:  # Ensure we skip column 6 again
                col -= 1

            # Zigzag downwards
            while row < size and col >= 0:
                if matrix[row, col] == -1 and number < len(binary_data):
                    matrix[row, col] = int(binary_data[number])
                    number += 1
                if col > 0 and matrix[row, col - 1] == -1 and number < len(binary_data):
                    matrix[row, col - 1] = int(binary_data[number])
                    number += 1
                row += 1  # Move DOWN

            col -= 2  # Move left by two columns
            row -= 1  # Adjust row after reaching the bottom

        return matrix



    global num_ec_codewords

    def padding(binary_string, version, ec_num):
        global num_ec_codewords

        total_codewords = {
            1: 26, 2: 44, 3: 70, 4: 100, 5: 108, 6: 136, 7: 156, 8: 194,
            9: 232, 10: 274
        }

        target_lengths = {
            1: {"L": 26 - 7, "M": 26 - 10, "Q": 26 - 13, "H": 26 - 17},
            2: {"L": 44 - 10, "M": 44 - 16, "Q": 44 - 22, "H": 44 - 28},
            3: {"L": 70 - 15, "M": 70 - 24, "Q": 70 - 34, "H": 70 - 44},
            4: {"L": 100 - 20, "M": 100 - 32, "Q": 100 - 49, "H": 100 - 64},
            5: {"L": 108 - 26, "M": 108 - 41, "Q": 108 - 62, "H": 108 - 80},
            6: {"L": 136 - 36, "M": 136 - 55, "Q": 136 - 82, "H": 136 - 102},
            7: {"L": 156 - 42, "M": 156 - 63, "Q": 156 - 94, "H": 156 - 120},
            8: {"L": 194 - 58, "M": 194 - 87, "Q": 194 - 130, "H": 194 - 162},
            9: {"L": 232 - 72, "M": 232 - 106, "Q": 232 - 151, "H": 232 - 194},
            10: {"L": 274 - 92, "M": 274 - 137, "Q": 274 - 202, "H": 274 - 251},
        }

        if version not in total_codewords:
            raise ValueError(f"Invalid QR version: {version}. Valid versions are from 1 to 40.")
        
        # Validate if the version exists in the target lengths for ECC levels
        if version not in target_lengths:
            raise ValueError(f"Invalid QR version: {version}. Valid versions are from 1 to 40.")
        
        # Check if the error correction level is valid
        if ec_num not in target_lengths[version]:
            raise ValueError("Invalid error correction level. Use 'L', 'M', 'Q', or 'H'.")

        # Get the total number of codewords for the specified version
        total_codewords_for_version = total_codewords[version]
        
        # Get the target length for the specified version and error correction level
        target_length = target_lengths[version].get(ec_num, 0)

        # Calculate the number of error correction codewords
        num_ec_codewords = total_codewords_for_version - target_length

        current_length = len(binary_string) // 8
        padding_needed = target_length - current_length
        print(target_length, current_length, padding_needed)

        # Generate padding bytes (alternating 0xEC and 0x11)
        padding_bytes = []
        for i in range(padding_needed):
            if i % 2 == 0:
                padding_bytes.append(0xEC)
            else:
                padding_bytes.append(0x11)

        # Convert padding bytes to binary
        padding_binary = "".join(f"{byte:08b}" for byte in padding_bytes)

        # Add padding to the original binary string
        padded_binary_string = binary_string + padding_binary

        return padded_binary_string

    final_binary_data = padding(binary_data, version, ec_num)
    def binary_to_hex(binary_string):
        # Ensure the binary string length is a multiple of 8
        if len(binary_string) % 8 != 0:
            raise ValueError("Binary string length must be a multiple of 8")

        # Convert each 8-bit chunk into a hex value
        hex_string = " ".join(f"{int(binary_string[i:i+8], 2):02X}" for i in range(0, len(binary_string), 8))
        
        return hex_string

    binary_string = final_binary_data

    hex_output = binary_to_hex(binary_string)
    print(hex_output)


    data_codewords = [int(final_binary_data[i:i+8], 2) for i in range(0, len(final_binary_data), 8)]

    rs = reedsolo.RSCodec(num_ec_codewords)
    full_codewords = rs.encode(data_codewords)
    ec_codewords = full_codewords[-num_ec_codewords:]

    final_binary_data += ''.join(format(byte, '08b') for byte in ec_codewords) + "0000000"
    print(final_binary_data)

    mat = fill_qr_matrix(mat, size, final_binary_data)

    # calculam masca

    # def apply_mask(matrix, mask_pattern):
    #     """Applies the given mask pattern to the QR matrix."""
    #     size = len(matrix)
    #     masked_matrix = np.copy(matrix)
        
    #     for i in range(size):
    #         for j in range(size):
    #             if mask_pattern == 0:
    #                 condition = (i + j) % 2 == 0
    #             elif mask_pattern == 1:
    #                 condition = i % 2 == 0
    #             elif mask_pattern == 2:
    #                 condition = j % 3 == 0
    #             elif mask_pattern == 3:
    #                 condition = (i + j) % 3 == 0
    #             elif mask_pattern == 4:
    #                 condition = (i // 2 + j // 3) % 2 == 0
    #             elif mask_pattern == 5:
    #                 condition = ((i * j) % 2) + ((i * j) % 3) == 0
    #             elif mask_pattern == 6:
    #                 condition = (((i * j) % 2) + ((i * j) % 3)) % 2 == 0
    #             elif mask_pattern == 7:
    #                 condition = (((i + j) % 2) + ((i * j) % 3)) % 2 == 0
                
    #             if condition:
    #                 masked_matrix[i, j] ^= 1  # Toggle bit
        
    #     return masked_matrix

    # def calculate_penalty(matrix):
    #     """Calculates the penalty scores for a given QR matrix."""
    #     size = len(matrix)
    #     RunP, BoxP, FindP, BalP = 0, 0, 0, 0
        
    #     # RunP: Consecutive runs of the same color
    #     for i in range(size):
    #         for j in range(size - 4):
    #             if all(matrix[i, j + k] == matrix[i, j] for k in range(5)):
    #                 RunP += 3  # Base penalty for 5-module run
    #                 k = 5
    #                 while j + k < size and matrix[i, j + k] == matrix[i, j]:
    #                     RunP += 1
    #                     k += 1
        
    #     for j in range(size):
    #         for i in range(size - 4):
    #             if all(matrix[i + k, j] == matrix[i, j] for k in range(5)):
    #                 RunP += 3
    #                 k = 5
    #                 while i + k < size and matrix[i + k, j] == matrix[i, j]:
    #                     RunP += 1
    #                     k += 1
        
    #     # BoxP: 2x2 blocks
    #     for i in range(size - 1):
    #         for j in range(size - 1):
    #             if (matrix[i, j] == matrix[i, j + 1] == matrix[i + 1, j] == matrix[i + 1, j + 1]):
    #                 BoxP += 3
        
    #     # FindP: Finder-like patterns (1011101)
    #     finder_pattern = np.array([1, 0, 1, 1, 1, 0, 1])
    #     for i in range(size):
    #         for j in range(size - 6):
    #             if np.array_equal(matrix[i, j:j + 7], finder_pattern) or np.array_equal(matrix[j:j + 7, i], finder_pattern):
    #                 FindP += 40
        
    #     # BalP: Dark/light balance
    #     dark_modules = np.sum(matrix)
    #     total_modules = size * size
    #     dark_ratio = (dark_modules / total_modules) * 100
        
    #     if 45 <= dark_ratio <= 55:
    #         BalP = 0
    #     elif 40 <= dark_ratio < 60:
    #         BalP = 10
    #     elif 35 <= dark_ratio < 65:
    #         BalP = 20
    #     elif 30 <= dark_ratio < 70:
    #         BalP = 30
    #     else:
    #         BalP = 40
        
    #     TotalP = RunP + BoxP + FindP + BalP
    #     return RunP, BoxP, FindP, BalP, TotalP

    # def main(qr_matrix):
    #     """Computes the penalty scores for all 8 masks."""
    #     results = []
    #     for mask in range(8):
    #         masked_matrix = apply_mask(qr_matrix, mask)
    #         RunP, BoxP, FindP, BalP, TotalP = calculate_penalty(masked_matrix)
    #         results.append((mask, RunP, BoxP, FindP, BalP, TotalP))
        
    #     # Sort by lowest total penalty
    #     results.sort(key=lambda x: x[-1])
        
    #     print("Mask | RunP | BoxP | FindP | BalP | TotalP")
    #     print("----------------------------------------")
    #     for res in results:
    #         print(f"  {res[0]}  |  {res[1]}  |  {res[2]}  |  {res[3]}  |  {res[4]}  |  {res[5]}")
        
    #     best_mask = results[0][0]
    #     print(f"\nBest mask: {best_mask} with penalty {results[0][-1]}")

    # # Example Usage
    # main(mat)

    def apply_mask(matrix, mask_pattern):
        """Applies the given mask pattern to the QR matrix."""
        size = len(matrix)
        masked_matrix = np.copy(matrix)
        
        for i in range(size):
            for j in range(size):
                if mask_pattern == 0:
                    condition = (i + j) % 2 == 0
                elif mask_pattern == 1:
                    condition = i % 2 == 0
                elif mask_pattern == 2:
                    condition = j % 3 == 0
                elif mask_pattern == 3:
                    condition = (i + j) % 3 == 0
                elif mask_pattern == 4:
                    condition = (i // 2 + j // 3) % 2 == 0
                elif mask_pattern == 5:
                    condition = ((i * j) % 2) + ((i * j) % 3) == 0
                elif mask_pattern == 6:
                    condition = (((i * j) % 2) + ((i * j) % 3)) % 2 == 0
                elif mask_pattern == 7:
                    condition = (((i + j) % 2) + ((i * j) % 3)) % 2 == 0
                
                if condition:
                    masked_matrix[i, j] ^= 1  # Toggle bit
        
        return masked_matrix


    def calculate_penalty(matrix):
        size = len(matrix)
        score = 0

        # RunP: Detect consecutive runs of the same color
        def calculate_run_penalty(line):
            run_length = 1
            run_score = 0
            for i in range(1, len(line)):
                if line[i] == line[i - 1]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        for i in range(i-run_length, i):
                            line[i]=-1
                        run_score += run_length - 2  # 3 points for 5, 4 points for 6, etc.
                    run_length = 1
            if run_length >= 5:
                run_score += run_length - 2
            return run_score
        for row in matrix:
            runP = calculate_run_penalty(row)
        for col in matrix.T:
            # runP += calculate_run_penalty(col)
            score += runP
        print(matrix)
        # BoxP: Count 2x2 blocks of the same color
        boxP = 0
        for i in range(size - 1):
            for j in range(size - 1):
                if matrix[i, j] == matrix[i, j + 1] == matrix[i + 1, j] == matrix[i + 1, j + 1]:
                    boxP += 3
        score += boxP

        # FindP: Detect finder patterns (approximated as 1-1-3-1-1 sequences)
        def detect_finder_pattern(line):
            count = 0
            pattern1 = [1,0,1,1,1,0,1,0]
            pattern2 = [0,1,0,1,1,1,0,1]
            for i in range(len(line) - 8):
                if list(line[i:i+9]) == pattern1 or list(line[i:i+9]) == pattern2:
                    count += 1
            return count
        findP = 0
        for row in matrix:
            findP += 40 * detect_finder_pattern(row)
        for col in matrix.T:
            findP += 40 * detect_finder_pattern(col)
        score += findP

        # BalP: Compute balance penalty
        dark_modules = np.sum(matrix)
        total_modules = size * size
        dark_ratio = (dark_modules / total_modules) * 100
        biggest_ratio = max(dark_ratio,100-dark_ratio)
        print("Biggest Ratio:",biggest_ratio)
        
        penalty_steps = [5, 10, 15, 20, 25, 30, 35, 40]  # Every 5% outside [45%, 55%]
        balP = 0
        if biggest_ratio - 55 > 0:
            difference = biggest_ratio - 55
            balP = math.ceil(difference/5)*10
            score += balP

        return runP, boxP, findP, balP, score

    def main(qr_matrix):
        """Computes the penalty scores for all 8 masks."""
        results = []
        for mask in range(8):
            masked_matrix = apply_mask(qr_matrix, mask)
            score = calculate_penalty(masked_matrix)
            results.append((mask, score[0], score[1], score[2], score[3], score[4]))
        
        # Sort by lowest total penalty
        results.sort(key=lambda x: x[-1])
        
        print("Mask | RunP | BoxP | FindP | BalP | TotalP")
        print("----------------------------------------")
        for res in results:
            print(f"  {res[0]}  |  {res[1]}  |  {res[2]}  |  {res[3]}  |  {res[4]}  |  {res[5]} ")
        
        best_mask = results[0][0]
        print(f"\nBest mask: {best_mask} with penalty {results[0][-1]}")

    # Example Usage
    main(mat)


    tuplul = (ec_num, mask_num)
    for i in range(8):
        mat[8, i if i < 6 else i + 1] = type_information_bits[tuplul][i]
        mat[8, size - 8 + i] = type_information_bits[tuplul][i+7]
    for i in range(8):
        mat[size - 1 - i if i < 7 else i + 1, 8] = type_information_bits[tuplul][i]
        mat[0 + i if i != 6 else i + 1 , 8] = type_information_bits[tuplul][14-i]
    mat[size - 8, 8] = 1
    # xoram

    def mask_bit(row, col, mat, mask_num):
        if mask_num == "0":
            mat[row][col] ^= (row + col) % 2 == 0
        elif mask_num == "1":
            mat[row][col] ^= row % 2 == 0
        elif mask_num == "2":
            mat[row][col] ^= col % 3 == 0
        elif mask_num == "3":
            mat[row][col] ^= (row + col) % 3 == 0
        elif mask_num == "4":
            mat[row][col] ^= (np.floor(row / 2) + np.floor(col / 3)) % 2 == 0
        elif mask_num == "5":
            mat[row][col] ^= ((row * col) % 2) + ((row * col) % 3) == 0
        elif mask_num == "6":
            mat[row][col] ^= (((row * col) % 2) + ((row * col) % 3)) % 2 == 0
        elif mask_num == "7":
            mat[row][col] ^= (((row + col) % 2) + ((row * col) % 3)) % 2 == 0

        
    for row in range(size):
        for col in range(size):
            if mat_for_mask[row, col] == -1:
                mask_bit(row,col, mat, mask_num)
    return mat
fig, ax = plt.subplots(figsize=(5, 3))
plt.subplots_adjust(bottom=0.3)  # Space for the text box
ax.axis('off')  # Hide axes

# Function to update text image
def update_text(text):
    ax.clear()  # Clear previous image
    ax.axis('off')  # Hide axes
    
    matrix = big_problem(text)  # Generate image matrix
    ax.imshow(matrix, cmap='gray_r', interpolation='nearest')  # Display image
    
    fig.canvas.draw_idle()  # Refresh the figure

# Create a text input box
axbox = plt.axes([0.2, 0.1, 0.6, 0.075])  # Position [left, bottom, width, height]
text_box = TextBox(axbox, "Enter Text: ")
text_box.on_submit(update_text)  # Call update_text when text is entered

plt.show()