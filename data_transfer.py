from pymongo import MongoClient
from pymongo.errors import BulkWriteError

def transfer_database(src_uri, dest_uri, db_name):
    """
    Transfers all collections and documents from one MongoDB database to another.

    Parameters:
        src_uri (str): MongoDB connection URI for source.
        dest_uri (str): MongoDB connection URI for destination.
        db_name (str): Database name to transfer.
    """

    # Connect to source and destination clients
    src_client = MongoClient(src_uri)
    dest_client = MongoClient(dest_uri)

    src_db = src_client[db_name]
    dest_db = dest_client[db_name]

    # List all collections
    collections = src_db.list_collection_names()
    print(f"Found collections: {collections}")

    for col in collections:
        print(f"\nTransferring collection: {col}")
        src_collection = src_db[col]
        dest_collection = dest_db[col]

        # Drop destination collection before inserting fresh data
        dest_collection.drop()

        # Fetch documents in batches (to avoid memory overload)
        batch_size = 1000
        docs = src_collection.find()
        batch = []

        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_size:
                try:
                    dest_collection.insert_many(batch, ordered=False)
                except BulkWriteError as bwe:
                    print(f"Bulk write error: {bwe.details}")
                batch = []

        # Insert remaining docs
        if batch:
            try:
                dest_collection.insert_many(batch, ordered=False)
            except BulkWriteError as bwe:
                print(f"Bulk write error: {bwe.details}")

        print(f"✅ Finished transferring {col}")

    print("\n🎉 Database transfer completed successfully!")


if __name__ == "__main__":
    # Example usage
    src_uri = input("Enter source MongoDB URI: ").strip()
    dest_uri = input("Enter destination MongoDB URI: ").strip()
    db_name = input("Enter database name to transfer: ").strip()

    transfer_database(src_uri, dest_uri, db_name)
