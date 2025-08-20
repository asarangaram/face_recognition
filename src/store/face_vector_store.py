from dataclasses import dataclass
from lancedb.pydantic import LanceModel, Vector
import uuid
import numpy as np
from typing import Annotated, List, Dict


class FaceRecognitionSchema(LanceModel):
    id: str
    vector: Annotated[List[float], Vector(512)]


class LanceTableManager:
    def __init__(self, db, table_name: str, schema: type[LanceModel]):
        if table_name not in db.table_names():
            self.tbl = db.create_table(table_name, schema=schema)
        else:
            self.tbl = db.open_table(table_name)
        schema_fields = [field.name for field in self.tbl.schema]
        if schema_fields != list(schema.model_fields.keys()):
            raise RuntimeError(f"Table {table_name} has a different schema.")


class FaceVectorStore(LanceTableManager):
    def __init__(self, db, table_name: str):
        super().__init__(db, table_name=table_name, schema=FaceRecognitionSchema)

    def add(self, id: str, vector: Annotated[List[float], Vector(512)]):
        self.tbl.add([FaceRecognitionSchema(id=id, vector=vector)])

    def remove(self, id: str):
        self.tbl.delete(f"id = '{id}'")
        pass

    def searchByVector(
        self,
        vector: Annotated[List[float], Vector(512)],
        threshold: float = 0.85,
        count: int = 1,
    ):
        items_found = (
            self.tbl.search(vector, vector_column_name="vector")
            .metric("cosine")
            .limit(count)
            .to_list()
        )

        result = []
        for item in items_found:
            similarity_score = round(1 - item["_distance"], 2)
            if similarity_score >= threshold:
                result.append((item["id"], similarity_score))
        return result

    def searchById(self, id: str):
        result = self.tbl.search().where(f"id = {id}").limit(1).to_list()
        if result:
            return result[0]
        else:
            return None
