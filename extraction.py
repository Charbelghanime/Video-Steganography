import sqlite3

def map_retrieval_info_to_bytes(db_path, retrieval_info):
    """
    Map retrieval information to the corresponding byte sequences.
    
    Args:
        db_path (str): Path to the retrieval database.
        retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).

    Returns:
        list: List of byte sequences corresponding to the retrieval information.
    """
    byte_sequence = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for info in retrieval_info:
        video_id, feature_id, position_id = info
        cursor.execute(
            "SELECT ByteSequence FROM RetrievalDatabase WHERE VideoID = ? AND FeatureID = ? AND PositionID = ?",
            (video_id, feature_id, position_id)
        )
        result = cursor.fetchone()
        if result:
            byte_sequence.append(result[0])
        else:
            print(f"[WARNING] No byte sequence found for retrieval info: {info}")

    conn.close()
    return byte_sequence

def remove_padding(byte_sequence):
    """
    Remove the padding from the byte sequence to recover the original secret information.

    Args:
        byte_sequence (list): List of binary strings (8-bit) representing the byte sequence.

    Returns:
        str: Original secret information without padding.
    """
    if not byte_sequence:
        print("[ERROR] Byte sequence is empty.")
        return ""

    # Check the last byte to determine padding
    last_byte = byte_sequence[-1]
    if last_byte == '00000000':
        # No padding
        byte_sequence = byte_sequence[:-1]
    else:
        # Padding exists, remove the padded zeros
        padding_length = int(last_byte, 2)
        byte_sequence = byte_sequence[:-1]  # Remove the padding indicator byte
        if padding_length > 0:
            byte_sequence[-1] = byte_sequence[-1][:-padding_length]

    # Combine the byte sequence into the original secret information
    secret_info = ''.join(byte_sequence)
    return secret_info

def binary_to_text(binary_data):
    """
    Convert binary string data to a readable text message.

    Args:
        binary_data (str): Binary string representing the secret information.

    Returns:
        str: Decoded text message.
    """
    chars = [chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)]
    return ''.join(chars)

def extract_secret_info(db_path, retrieval_info):
    """
    Extract the secret information from the retrieval information and database.

    Args:
        db_path (str): Path to the retrieval database.
        retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).

    Returns:
        str: Extracted secret information.
    """
    # Step 1: Map retrieval info to byte sequence
    byte_sequence = map_retrieval_info_to_bytes(db_path, retrieval_info)
    print(f"[INFO] Byte sequence retrieved: {byte_sequence}")

    # Step 2: Remove padding to recover secret information
    binary_data = remove_padding(byte_sequence)
    print(f"[INFO] Extracted secret information (binary): {binary_data}")

    # Step 3: Convert binary data to readable text
    message = binary_to_text(binary_data)

    return message

if __name__ == "__main__":
    # Example usage
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database
    retrieval_info = [(9, '21', 89), (7, '01', 1682), (4, '10', 1), (5, '21', 9), (7, '01', 3074), (16, '11', 9), (4, '20', 15), (12, '20', 54), (4, '10', 1), (11, '10', 10), (16, '11', 9), (1, '01', 549), (5, '10', 2), (5, '21', 2), (7, '01', 2774), (1, '01', 310), (1, '01', 316), (8, '10', 0), (12, '10', 12), (7, '01', 3102), (7, '00', 2320), (7, '01', 1421), (5, '10', 2), (1, '00', 485), (5, '01', 796), (12, '20', 20), (8, '10', 0), (7, '01', 1650), (16, '11', 9), (10, '21', 11), (7, '01', 2093), (7, '00', 1081), (1, '00', 487), (16, '11', 9), (16, '11', 15), (7, '01', 2141), (7, '21', 24), (7, '00', 1068), (7, '01', 2633), (7, '21', 27), (5, '10', 2), (11, '20', 33), (7, '01', 1627), (16, '10', 8), (16, '11', 15), (5, '21', 2), (5, '01', 792), (9, '11', 3), (7, '01', 2658), (5, '10', 2), (7, '21', 27), (5, '01', 792), (8, '10', 0), (7, '01', 2188), (5, '01', 794), (4, '20', 8), (5, '10', 2), (7, '01', 1121), (11, '21', 57), (5, '10', 2), (12, '21', 9), (5, '01', 798), (5, '10', 2), (7, '21', 10), (7, '01', 3074), (7, '01', 1419), (5, '10', 2), (7, '01', 1637), (10, '10', 11), (7, '01', 3414), (7, '01', 2948), (11, '21', 57), (5, '10', 2), (7, '21', 24), (7, '01', 2080), (7, '01', 2628), (1, '11', 19), (7, '01', 2080), (7, '01', 2178), (7, '01', 2633), (8, '10', 0), (5, '01', 795), (7, '01', 931), (8, '10', 0), (5, '01', 793), (12, '20', 3), (7, '00', 1068), (5, '10', 2), (11, '21', 5), (10, '20', 16), (7, '01', 2555), (10, '21', 11), (7, '01', 1646), (12, '21', 60), (7, '01', 2139), (5, '10', 2), (5, '21', 2), (7, '01', 2131), (7, '00', 1080), (8, '10', 0), (16, '11', 15), (7, '01', 2132), (7, '00', 1068), (7, '00', 2317), (7, '01', 2638), (7, '00', 1068), (4, '10', 10), (9, '20', 60), (9, '20', 124), (9, '20', 220), (5, '10', 2), (7, '01', 2608), (4, '11', 12), (7, '00', 1081), (7, '01', 2140), (16, '11', 9), (7, '01', 2497), (16, '11', 15), (16, '11', 9), (7, '21', 44), (7, '01', 3103), (7, '01', 2132), (5, '10', 2), (11, '21', 5), (7, '01', 1655), (16, '10', 8), (16, '10', 8), (5, '21', 12), (5, '01', 793), (7, '00', 1081), (7, '01', 1411), (16, '11', 9), (12, '21', 40), (16, '11', 9), (1, '01', 199), (9, '21', 158), (12, '20', 51), (2, '20', 0), (9, '21', 35), (7, '00', 1190), (9, '20', 61), (7, '00', 1268), (13, '20', 15), (7, '21', 18), (9, '20', 0), (12, '20', 16), (5, '00', 867)]
    # Extract the secret information
    message = extract_secret_info(db_path, retrieval_info)
    print("[INFO] Final message:", message)
