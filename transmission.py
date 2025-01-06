import math
import sqlite3
from establishment_of_db import video_retrieval_database_construction

def segment_and_pad(secret_info):
    """
    Segment the secret information into 8-bit chunks (bytes) and pad if necessary.
    """
    byte_sequence = []
    length = len(secret_info)
    mod_value = length % 8

    if mod_value != 0:
        # Pad the sequence with zeros to make its length a multiple of 8
        padding = 8 - mod_value
        secret_info += '0' * padding
        byte_sequence = [secret_info[i:i + 8] for i in range(0, len(secret_info), 8)]
        # Add an additional byte to indicate the number of zeros padded
        byte_sequence.append(format(padding, '08b'))
    else:
        byte_sequence = [secret_info[i:i + 8] for i in range(0, len(secret_info), 8)]
        # Add a byte of '00000000' to indicate no padding
        byte_sequence.append('00000000')

    return byte_sequence

def retrieve_from_database(db_path, byte_sequence):
    """
    Retrieve the corresponding retrieval information for each byte from the database.
    """
    retrieval_info = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for byte in byte_sequence:
        cursor.execute("SELECT VideoID, FeatureID, PositionID FROM RetrievalDatabase WHERE ByteSequence = ?", (byte,))
        result = cursor.fetchone()
        if result:
            retrieval_info.append(result)
        else:
            print(f"[WARNING] No match found for byte sequence: {byte}")

    conn.close()
    return retrieval_info

def transmit_secret_info(db_path, secret_info):
    """
    Transmit the secret information using the constructed retrieval database and carrier videos.
    """
    # Step 1: Segment and pad the secret information
    byte_sequence = segment_and_pad(secret_info)
    print(f"[INFO] Byte sequence after segmentation and padding: {byte_sequence}")

    # Step 2: Retrieve information from the database
    retrieval_info = retrieve_from_database(db_path, byte_sequence)
    print(f"[INFO] Retrieved retrieval information: {retrieval_info}")

    # Step 3: Send the retrieval information and carrier videos (here just simulate sending)
    print(f"[INFO] Transmitting retrieval information and carrier videos...")

    return retrieval_info

# Example usage
videos_dir = "Videos"  # Directory containing carrier videos
db_path = "retrieval_database.sqlite"  # Path to the retrieval database
db_path, carrier_videos = video_retrieval_database_construction(videos_dir, db_path)
# Example secret information (binary string)
secret_info = "1101011100101110"  # Replace with actual secret information

retrieval_info = transmit_secret_info(videos_dir, db_path, secret_info)
print("[INFO] Transmission complete.")
