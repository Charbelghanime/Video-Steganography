import sqlite3
import json


class SecretExtractor:
    def __init__(self, db_path):
        """
        Initialize the SecretExtractor with the path to the retrieval database.

        Args:
            db_path (str): Path to the retrieval database.
        """
        self.db_path = db_path

    def map_retrieval_info_to_bytes(self, retrieval_info):
        """
        Map retrieval information to the corresponding byte sequences.

        Args:
            retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).

        Returns:
            list: List of byte sequences corresponding to the retrieval information.
        """
        byte_sequence = []

        conn = sqlite3.connect(self.db_path)
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

    def remove_padding(self, byte_sequence):
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

    def binary_to_text(self, binary_data):
        """
        Convert binary string data to a readable text message.

        Args:
            binary_data (str): Binary string representing the secret information.

        Returns:
            str: Decoded text message.
        """
        chars = [chr(int(binary_data[i:i + 8], 2)) for i in range(0, len(binary_data), 8)]
        return ''.join(chars)

    def extract_secret_info(self, retrieval_info):
        """
        Extract the secret information from the retrieval information and database.

        Args:
            retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).

        Returns:
            str: Extracted secret information.
        """
        # Step 1: Map retrieval info to byte sequence
        byte_sequence = self.map_retrieval_info_to_bytes(retrieval_info)
        print(f"[INFO] Byte sequence retrieved: {byte_sequence}")

        # Step 2: Remove padding to recover secret information
        binary_data = self.remove_padding(byte_sequence)
        print(f"[INFO] Extracted secret information (binary): {binary_data}")

        # Step 3: Convert binary data to readable text
        message = self.binary_to_text(binary_data)

        return message


if __name__ == "__main__":
    # Example usage
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database

    # Load retrieval information from the file
    try:
        with open("retrieval_info.json", "r") as f:
            retrieval_info = json.load(f)
        print(f"[INFO] Retrieval information loaded: {retrieval_info}")
    except FileNotFoundError:
        print("[ERROR] Retrieval information file not found. Run transmission.py first.")
        exit(1)

    # Create a SecretExtractor object
    extractor = SecretExtractor(db_path)

    # Extract the secret information
    message = extractor.extract_secret_info(retrieval_info)
    print("[INFO] Final message:", message)
