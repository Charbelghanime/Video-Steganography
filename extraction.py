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
    secret_info = remove_padding(byte_sequence)
    print(f"[INFO] Extracted secret information: {secret_info}")

    return secret_info

if __name__ == "__main__":
    # To do: implement obtaining the hashes on the receiver end through generating the hashes on the receiver end based on auxiliary information
    # Not ot forget that the videos must transmitted to the receiver based on the auxiliary information
    # Example usage
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database
    retrieval_info = [(1, '01', 430), (4, '10', 10), (1, '00', 36)]  # Example retrieval information

    # Extract the secret information
    secret_info = extract_secret_info(db_path, retrieval_info)
    print("[INFO] Final secret information:", secret_info)
