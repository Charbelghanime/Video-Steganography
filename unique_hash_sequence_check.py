import sqlite3

def count_unique_hash_sequences(db_path):
    """
    Count the number of unique hash sequences in the database.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        int: Number of unique hash sequences.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # SQL query to count distinct ByteSequence
    cursor.execute("SELECT COUNT(DISTINCT ByteSequence) FROM RetrievalDatabase")
    unique_count = cursor.fetchone()[0]
    
    conn.close()
    return unique_count

db_path = "retrieval_database.sqlite"  # Path to your database
unique_hash_count = count_unique_hash_sequences(db_path)
print(f"[INFO] Number of unique hash sequences in the database: {unique_hash_count}")