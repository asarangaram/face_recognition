from lancedb.pydantic import LanceModel, Vector
import uuid
import numpy as np
from typing import List, Dict
import lancedb
import logging

logging.basicConfig(
    level=logging.WARNING,  # global default
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class FaceRecognitionSchema(LanceModel):
    id: str  # Unique identifier for each entry
    vector: Vector(512)  # Face embeddings, fixed size of 512

class FaceVectorStore:
    def __init__(self, db, table_name: str):
        # Initialize the table
        if table_name not in db.table_names():
            tbl = db.create_table(table_name, schema=FaceRecognitionSchema)
        else:
            tbl = db.open_table(table_name)
            schema_fields = [field.name for field in tbl.schema]
            if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
                raise RuntimeError(f"Table {table_name} has a different schema.")
        self.tbl = tbl


if __name__ == "__main__":
    logging.info("WHERE AM I GOING")
    # Database and table setup
    uri = "./face_database_test.vec.db"
    table_name = "face"

    # Connect to the database
    db = lancedb.connect(uri=uri)

    vector_store =  FaceVectorStore(db=db, table_name=table_name)

    print(vector_store.tbl.schema)
