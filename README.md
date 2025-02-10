# Generator / Scanner de Coduri QR

## Descriere
Acest proiect este un generator și scanner de coduri QR facut cu scop academic.

## Caracteristici
- Introducerea unui text sau URL pentru generarea codului QR
- Generarea instantanee a codului QR
- Interfață simplă și intuitivă

## Constrangeri
- Generarea si scanarea merge pana la verisunea [1, 2, 3, 4-Low, 5-Low] am luat in considerarea ca error corection bitii se genereaza prin separarea pe indexi la data biti doar pentru versiunea 3 pentru quartile si high

  <img width="183" alt="Screenshot 2025-02-10 at 16 28 27" src="https://github.com/user-attachments/assets/9209f2e6-a614-4cc7-8d9c-b40a281b8ddd" />

- Totodata alignment patternu' este pus static la pozitia (size - 9, size - 9) pentru versiuni >=2 deci de la versiune 7 nu mai merge si ar trebui sa iei valori din tabel
- Marimea pentru Numeric - 10 biti AlphaNumeric - 9 biti Byte - 8 biti este statica deci de la versiunea >=9 nu va mai merge
- Nu am luat in considerare tipul kanji

## Cerințe
Pentru a rula acest proiect, este necesar să aveți instalat:
- Python 3.x
- Bibliotecile necesare (specificate în `requirements.txt`)

## Instalare
1. Clonați acest repository:
   ```sh
   git clone https://github.com/Alexandru-Morteanu/asc_pjct.git
   ```
2. Accesați directorul proiectului:
   ```sh
   cd asc_pjct
   ```
3. Instalați dependențele necesare:
   ```sh
   pip install -r requirements.txt
   ```

## Utilizare Generator
1. Rulați scriptul de generare:
   ```sh
   python generate.py
   ```
3. Introduceți textul sau URL-ul dorit
4. Selectati calitatea QR codului

## Utilizare Scanner
1. Rulați scriptul de scanare:
   ```sh
   python scan.py
   ```
2. Selectati poza cu QR code
3. Cititi mesajul in consola

## Cum functioneaza Generarea
1. Prima oara aflam pe ce versiune incape acel input in functie de format si calitate
   ```sh
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
   ```
2. Dupa punem toti pixelii care sunt statici pentru fiecare versiune si salvam si o copie pentru a o folosi ulterior la maska si la adaugarea sirului binar

   <img width="306" alt="Screenshot 2025-02-08 at 23 54 41" src="https://github.com/user-attachments/assets/39294ab1-6660-4380-8054-d567aaf3c5e2" />

   ```sh
    size = 21+4*(version-1)
    # mat = np.full((size, size), -1)
   
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
   ```
4. Incepem sa cream sirul binar si adaugam Mode-ul dar si numarul de caractere in functie de Mode ( Numeric, Alphanumeric, Byte ), dupa adaugam terminatorul 0000 si la final numarul de 0 - uri pentru a se imparti exact la 8

   <img width="312" alt="Screenshot 2025-02-09 at 00 04 40" src="https://github.com/user-attachments/assets/016c3b63-1e6c-44e1-ba10-b55f1ac0a996" />

   ```sh
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
   ```
6. Acum adaugam bite padding pana ajungem la numarul de bytes alocat saptiul de date in functie de versiune si calitate

   <img width="295" alt="Screenshot 2025-02-09 at 00 08 07" src="https://github.com/user-attachments/assets/b3f87222-8cd4-4549-a334-3b3e67294cda" />

   ```sh
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
   ```
8. Si asta e data encodata pe 19 bytes (totalul de 26 pentru versiunea 1 )
   ```sh
   01000000010101001000011001010110110001101100011011110000111
   01100000100011110110000010001111011000001000111101100000100
   0111101100000100011110110000010001
   
   40 54 86 56 C6 C6 F0 EC 11 EC 11 EC 11 EC 11 EC 11 EC 11
   ```
9. Deci au ramas 7 bytes alocat pentru error corection bits
   ```sh
   rs = reedsolo.RSCodec(num_ec_codewords)
   full_codewords = rs.encode(data_codewords)
   ec_codewords = full_codewords[-num_ec_codewords:]

   final_binary_data += ''.join(format(byte, '08b') for byte in ec_codewords) + "0000000"
   ```
10. Si acum avem sirul final de biti

    <img width="295" alt="Screenshot 2025-02-09 at 00 10 02" src="https://github.com/user-attachments/assets/3af5d8e5-7406-45fb-8486-ca65f5abbda4" />

   ```sh
   01000000010101001000011001010110110001101100011011110000111
   01100000100011110110000010001111011000001000111101100000100
   01111011000001000111101100000100011010110101111010100101100
   0111101100001011001001101011110
   
   40 54 86 56 C6 C6 F0 EC 11 EC 11 EC 11 EC 11 EC 11 EC 11 AD 7A 96 3D 85 93 5E
   ```
11. Hai sa-i punem pe QR code, pentru asta avem codul
   ```sh
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
   ```
11. Bun, iar acum trebuie sa facem un tabel cu penalty points pentru fiecare maska si sa o alegem pe cea cu cel mai bun scor

    <img width="440" alt="Screenshot 2025-02-09 at 00 16 54" src="https://github.com/user-attachments/assets/f2405b3e-15bb-4b51-b641-c09a3b3a9133" />

   ```sh
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
   ```
12. Si ne-a mai ramas sa aplicam masca buna pe QR code ul nostru
   ```sh
    mat = mask_bit(mat, str(maska_buna), mat_for_mask)

    tuplul = (ec_num, str(maska_buna))
    for i in range(8):
        mat[8, i if i < 6 else i + 1] = type_information_bits[tuplul][i]
        mat[8, size - 8 + i] = type_information_bits[tuplul][i+7]
    for i in range(8):
        mat[size - 1 - i if i < 7 else i + 1, 8] = type_information_bits[tuplul][i]
        mat[0 + i if i != 6 else i + 1 , 8] = type_information_bits[tuplul][14-i]
    mat[size - 8, 8] = 1
   ```
13. Felicitari! Asta este!

    <img width="297" alt="Screenshot 2025-02-09 at 00 18 01" src="https://github.com/user-attachments/assets/8179b974-e48f-43db-97d1-25c7488d31e6" />

## Cum functioneaza Scanarea
   Hai sa il scanam acum
   
1. Prima oara trebuie sa calculam hamming distance ul intre bitii de format din QR code si toti bitii de format din lookup table apoi selectam bitii de format cu cea mai mica hamming distance
   
   <img width="445" alt="Screenshot 2025-02-10 at 16 32 19" src="https://github.com/user-attachments/assets/627c721b-0286-4c76-b584-10b313f72a99" />

3. Iar acum avem masca, deci putem sa o aplicam peste QR code

   <img width="295" alt="Screenshot 2025-02-09 at 00 10 02" src="https://github.com/user-attachments/assets/3af5d8e5-7406-45fb-8486-ca65f5abbda4" />
   
5. Si acum incercam sa scapam de errori prin reedsolomon library, luand in considerare pentru version 3 pe high si quartile ca se face split pe indexi la data biti
   ```sh
   01000000010101001000011001010110110001101100011011110000111
   01100000100011110110000010001111011000001000111101100000100
   0111101100000100011110110000010001
   
   40 54 86 56 C6 C6 F0 EC 11 EC 11 EC 11 EC 11 EC 11 EC 11
   ```
6. Procesam bitii: 4 biti de format cu [8, 9, 10] biti numarul de caractere si restul este efectiv mesajul
   ```sh
   Hello
   ```


## Contribuții
1. Morteanu Alexandru
2. Bejan-Topse Denis-Marian
3. Blezniuc Teoodor Mihai
4. Dragan Mario Alexandru


