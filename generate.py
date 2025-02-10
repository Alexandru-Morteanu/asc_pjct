import math
import numpy as np
import matplotlib.pyplot as plt
import reedsolo
from matplotlib.widgets import TextBox
from PIL import Image, ImageDraw
from matplotlib.widgets import TextBox, RadioButtons

num_ec_codewords = 0
mat=[]

def big_problem(input_string, ec_num):
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

    ec_num     = ec_num[0]

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
    if version == 3 and (ec_num == "Q" or ec_num == "H"):
        num_ec_codewords = num_ec_codewords // 2

        mid = len(final_binary_data) // 2
        first_half = final_binary_data[:mid] 
        second_half = final_binary_data[mid:] 

        even_codewords = [int(first_half[i:i+8], 2) for i in range(0, len(first_half), 8)]
        odd_codewords = [int(second_half[i:i+8], 2) for i in range(0, len(second_half), 8)]

        rs = reedsolo.RSCodec(num_ec_codewords)

        full_even_codewords = rs.encode(even_codewords)
        full_odd_codewords = rs.encode(odd_codewords)

        ec_even_codewords = full_even_codewords[-num_ec_codewords:]
        ec_odd_codewords = full_odd_codewords[-num_ec_codewords:]
        ec_even_binary = ' '.join(format(byte, '08b') for byte in ec_even_codewords)
        ec_odd_binary = ' '.join(format(byte, '08b') for byte in ec_odd_codewords)
        
        half_length = len(final_binary_data) // 2
        first_half = final_binary_data[:half_length]
        second_half = final_binary_data[half_length:]

        first_half_chunks = [first_half[i:i+8] for i in range(0, len(first_half), 8)]
        second_half_chunks = [second_half[i:i+8] for i in range(0, len(second_half), 8)]

        intersected = ""
        for bin1, bin2 in zip(first_half_chunks, second_half_chunks):
            intersected += bin1 + bin2

        for bin1, bin2 in zip(ec_even_binary.split(), ec_odd_binary.split()):
            intersected += bin1 + bin2
        final_binary_data = intersected + "0000000"

    if version != 3 or ec_num == "L" or ec_num == "M":
        final_binary_data += ''.join(format(byte, '08b') for byte in ec_codewords) + "0000000"

    mat = fill_qr_matrix(mat, size, final_binary_data)

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
                        run_score += run_length - 2  # 3 points for 5, 4 points for 6, etc.
                    run_length = 1
            if run_length >= 5:
                run_score += run_length - 2
            return run_score
        runP = 0
        for i in range(len(matrix)):
            runP += calculate_run_penalty(matrix[i])
        for i in range(len(matrix)):
            runP += calculate_run_penalty(matrix.T[i])
        score += runP
        #print(matrice_help)
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
            pattern1 = [0,0,0,0,1,0,1,1,1,0,1,0]
            pattern2 = [0,1,0,1,1,1,0,1,0,0,0,0]
            for i in range(len(line) - 11):
                if list(line[i:i+12]) == pattern1:
                    count += 1
                if list(line[i:i+12]) == pattern2:
                    count +=1
            return count
        padded_matrix = np.pad(matrix, pad_width=5, mode='constant', constant_values=0)
        findP = 0
        for row in padded_matrix:
            findP += 40 * detect_finder_pattern(row)
        for col in padded_matrix.T:
            findP += 40 * detect_finder_pattern(col)
        score += findP
        print("FindP: ",findP)

        # BalP: Compute balance penalty
        dark_modules = np.sum(matrix)
        total_modules = size * size
        dark_ratio = (dark_modules / total_modules) * 100
        biggest_ratio = max(dark_ratio,100-dark_ratio)
        print("Biggest Ratio:",biggest_ratio)
        
        balP = 0
        if biggest_ratio - 55 > 0:
            difference = biggest_ratio - 55
            balP = math.ceil(difference/5)*10
            score += balP
        print(score)
        return runP, boxP, findP, balP, score


    def calculate_final_mask(matrix, mat_for_mask, ec_num):
        """Computes the penalty scores for all 8 masks."""
        results = []
        for mask in range(8):
            matrice_help = matrix.copy()
            tuplul = (ec_num, str(mask))
            for i in range(8):
                matrice_help[8, i if i < 6 else i + 1] = type_information_bits[tuplul][i]
                matrice_help[8, size - 8 + i] = type_information_bits[tuplul][i+7]
            for i in range(8):
                matrice_help[size - 1 - i if i < 7 else i + 1, 8] = type_information_bits[tuplul][i]
                matrice_help[0 + i if i != 6 else i + 1 , 8] = type_information_bits[tuplul][14-i]
            matrice_help[size - 8, 8] = 1

            masked_matrix = mask_bit(matrice_help, str(mask), mat_for_mask)
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
        return best_mask

    maska_buna = calculate_final_mask(mat, mat_for_mask, ec_num)
    mat = mask_bit(mat, str(maska_buna), mat_for_mask)

    tuplul = (ec_num, str(maska_buna))
    for i in range(8):
        mat[8, i if i < 6 else i + 1] = type_information_bits[tuplul][i]
        mat[8, size - 8 + i] = type_information_bits[tuplul][i+7]
    for i in range(8):
        mat[size - 1 - i if i < 7 else i + 1, 8] = type_information_bits[tuplul][i]
        mat[0 + i if i != 6 else i + 1 , 8] = type_information_bits[tuplul][14-i]
    mat[size - 8, 8] = 1
    
    return mat

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.4)
ax.axis('off')

def update_text(text, label="Low"):
    ax.clear()
    ax.axis('off')
    mat = big_problem(text, label)
    ax.imshow(mat, cmap='gray_r', interpolation='nearest')
    
    fig.canvas.draw_idle()

def update_option(label):
    print(label)
    update_text(text_box.text, label)

axbox = plt.axes([0.2, 0.05, 0.6, 0.075])
text_box = TextBox(axbox, "Enter Text: ")
text_box.on_text_change(update_text)
ax_dropdown = plt.axes([0.2, 0.14, 0.2, 0.2])
dropdown = RadioButtons(ax_dropdown, ["Low", "Medium", "Quartile", "High"], 
                        activecolor="black")


for circle in dropdown.circles:
    circle.set_radius(0.03)
for label in dropdown.labels:
    label.set_fontsize(10)

dropdown.on_clicked(update_option)

plt.show()
