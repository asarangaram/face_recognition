from dataclasses import dataclass
import logging
import math
import os
from typing import Dict, Optional, List, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np

from face_recognition.store.registered_faces import db_create_table_registered_faces
from face_recognition.store.registered_person import db_create_table_registered_person
from face_recognition.store.face_vector_store import (
    FaceRecognitionSchema,
    FaceVectorStore,
)

from face_recognition.face import DetectedFace, KnownFace
from face_recognition.proc.face_detection import DetectionModel, EmbeddingModel
from face_recognition.proc.align_and_crop import align_and_crop


@dataclass
class RegisteredPerson:
    id: int
    name: str
    key_face_id: int


@dataclass
class RegisteredFace:
    id: str
    person_id: int
    person_name: str


class FaceRecognizer:
    face_table_name = "face"
    face_vector_table = "face"
    person_table_name = "person"

    @classmethod
    def vector_tables(cls):
        return [cls.face_vector_table]

    @classmethod
    def tables(cls):
        return [cls.face_table_name, cls.person_table_name]

    def __init__(
        self, db, dbModel, vectordb, face_dir: str, is_interactive: bool = False
    ):
        self.db = db  # to debug
        self.dbModel = dbModel  # to debug
        self.vectordb = vectordb  # to debug
        self.face_dir = face_dir
        self.is_interactive = is_interactive

        self.RegisteredFace = db_create_table_registered_faces(db, dbModel)
        self.RegisteredPerson = db_create_table_registered_person(db, dbModel)

        self.faceVectorStore = FaceVectorStore(
            vectordb, table_name=self.face_vector_table
        )

    def register_face(
        self, path: str, person_id: int = None, person_name: str = None
    ) -> Optional[RegisteredFace]:
        """
        POST /faces/register

        """
        logging.info(f'register_face: {f"id={person_id}" if person_id else f"name={person_name}"} -> {path}')
        detector = DetectionModel()
        detected_faces = detector.scan(path=path)

        num_faces = len(detected_faces.results)
        if num_faces > 1:
            logging.warning(
                f"Skipped {detected_faces.info} as it contains more than one face ({num_faces} faces detected)."
            )
            return None
        elif num_faces == 0:
            logging.warning(f"Skipped {detected_faces.info} as no faces were detected.")
            return None

        result = detected_faces.results[0]

        aligned_img, _ = align_and_crop(
            detected_faces.image,
            [landmark["landmark"] for landmark in result["landmarks"]],
        )
        embedding_model = EmbeddingModel()
        face_embedding = embedding_model.extract_face_embedding(aligned_img)
        result = detected_faces.results[0]

        face = self.save_face(
            identity=person_id if person_id else person_name,
            aligned_img=aligned_img,
            face_embedding=face_embedding,
        )

        if self.is_interactive:
            self.show_face(face)

        return face

    def register_faces_no_batch(self, faces: List[Tuple[Union[int, str], str]]):
        registerd_faces = []
        for identity, path in faces:
            face = None
            if isinstance(identity, int):
                face = self.register_face(path=path, person_id=identity)
            elif isinstance(identity, str):
                face = self.register_face(path=path, person_name=identity)
            if face:
                registerd_faces.append(face)
        return registerd_faces

    def register_faces(
        self, faces: List[Tuple[Union[int, str], str]]
    ) -> List[RegisteredFace]:
        """
        Register multiple faces at once.
        - known_faces: map of existing person_id to list of face image files
        - new_faces: map of new person_name to list of face image files
        Returns: all registered faces
        """
        identities = [t[0] for t in faces]
        image_files = [t[1] for t in faces]

        detector = DetectionModel()
        detected_faces_batch = list(detector.batch_scan(path=image_files))

        embedding_model = EmbeddingModel()
        embedding = []
        saved_faces = []
        for identity, detected_faces in zip(identities, detected_faces_batch):
            num_faces = len(detected_faces.results)
            if num_faces > 1:
                logging.warning(
                    f"Skipped {detected_faces.info} as it contains more than one face ({num_faces} faces detected)."
                )
                continue
            elif num_faces == 0:
                logging.warning(
                    f"Skipped {detected_faces.info} as no faces were detected."
                )
                continue

            result = detected_faces.results[0]

            aligned_img, _ = align_and_crop(
                detected_faces.image,
                [landmark["landmark"] for landmark in result["landmarks"]],
            )
            face_embedding = embedding_model.extract_face_embedding(aligned_img)
            result = detected_faces.results[0]
            embedding.append((identity, aligned_img, face_embedding))
            face = self.save_face(
                identity=identity,
                aligned_img=aligned_img,
                face_embedding=face_embedding,
            )

            if face:
                saved_faces.append(face)

        if self.is_interactive:
            self.show_faces(saved_faces)

        return saved_faces

    def save_face(self, identity, aligned_img, face_embedding) -> RegisteredFace:
        person = None
        if isinstance(identity, int):  # Id is provided.
            person = self.RegisteredPerson.get_person(id=identity)
        elif isinstance(identity, str):
            person = self.RegisteredPerson.create(identity)
            pass

        if not person:
            logging.warning(f"failed to get person with identity: {identity}")
            return None
        file_name = self.save_file(name=f"{person.name}_{person.id}", img=aligned_img)
        face = self.RegisteredFace.create(person_id=person.id, path=file_name)
        if not face:
            logging.warning(f"failed to get save face for identity {identity}")
            os.unlink(Path.joinpath(self.face_dir, f"{file_name}.png"))
            return None
        self.faceVectorStore.add(id=face.id, vector=face_embedding)

        return face

    def save_file(
        self, name: str, img: Union[np.ndarray, Image.Image], ext="png"
    ) -> Path:
        """
        Save an image to `folder` with a unique name in the format <name>_n.ext.
        Returns the saved Path.
        """
        folder = Path(self.face_dir)
        folder.mkdir(parents=True, exist_ok=True)

        counter = 1
        while True:
            file_name = f"{name}_{counter}"
            file_path = folder / f"{file_name}.{ext}"
            if not file_path.exists():
                break
            counter += 1

        if isinstance(img, Image.Image):
            img.save(file_path)
        elif isinstance(img, np.ndarray):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(file_path)
        else:
            raise TypeError("img must be a PIL Image or NumPy array")

        return file_name

    def forget_face(self, face_id: str) -> bool:
        """
        DELETE /faces/{id}
        """
        face = self.RegisteredFace.get_face(id=id)

        face.delete()
        person = self.RegisteredPerson.get_person(face.person_id)
        if person.faces < 1:
            person.delete()
        return True

    def forget_person(self, person_id: int) -> bool:
        """
        DELETE /persons/{id}
        """
        person = self.RegisteredPerson.get_person(id=person_id)
        person.delete()
        return True

    def get_all_persons(self) -> List[RegisteredPerson]:
        """
        GET /persons
        """
        all = self.RegisteredPerson.get_persons()
        persons = []
        for item in all:
            persons.append(
                RegisteredPerson(
                    id=item.id, name=item.name, key_face_id=item.key_face_id
                )
            )
        return persons

    def get_person(self, id: int) -> RegisteredPerson:
        """
        GET /persons/{id}
        """
        item = self.RegisteredPerson.get_person(id=id)
        return RegisteredPerson(
            id=item.id, name=item.name, key_face_id=item.key_face_id
        )

    def get_face(self, id: int) -> str:
        """
        GET /faces/{id}
        - returns the image
        """
        face = self.RegisteredFace.get_face(id=id)
        return face.path

    def get_person_by_face(self, id: int) -> RegisteredPerson:
        """
        GET /faces/{id}/person
        - returns the image
        """
        face = self.RegisteredFace.get_face(id=id)
        item = face.person

        return RegisteredPerson(
            id=item.id, name=item.name, key_face_id=item.key_face_id
        )

    def update_person(
        self,
        id: int,
        new_name: str = None,
        is_hidden: bool = None,
        key_face_id: int = None,
    ) -> RegisteredPerson:
        """
        PUT	/persons/{person_id}
        """
        current = self.RegisteredPerson.get_person(id=id)
        item = current.update(
            name=new_name, is_hidden=is_hidden, key_face_id=key_face_id
        )
        return RegisteredPerson(
            id=item.id, name=item.name, key_face_id=item.key_face_id
        )

    def reassign_to_person(
        self, face_id: int, new_person_id: int = None, new_person_name: str = None
    ) -> RegisteredFace:
        """
        PUT /faces/{id}/reassign
        """
        if not new_person_id:
            if new_person_name:
                person = self.RegisteredPerson.create(name=new_person_name)
                new_person_id = person.id
            else:
                raise Exception("Name this")

        current = self.RegisteredFace.get_face(id=id)
        f = current.update(person_id=new_person_id)
        return RegisteredFace(id=f.id, person_id=person.id, person_name=person.name)

    def detect_and_align_faces(
        self, path: str
    ) -> List[Tuple[np.array, list, DetectedFace]]:
        detector = DetectionModel()
        detected_faces = detector.detectFaces(path=path)

        aligned_faces = []
        for face in detected_faces.results:
            x1, y1, x2, y2 = map(int, face["bbox"])
            cropped_face = detected_faces.image[y1:y2, x1:x2]
            landmarks = [landmark["landmark"] for landmark in face["landmarks"]]
            aligned_face, _ = align_and_crop(
                detected_faces.image, landmarks
            )  # Align and crop face
            aligned_faces.append(
                (aligned_face, landmarks, DetectedFace(bbox=(x1, y1, x2, y2)))
            )
        return aligned_faces

    def show_image(self, image, title="Images", figsize=(15, 5)):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.axis("off")  # optional: hides the axes
        plt.show()

    def get_face_path(self, face):
        file_name = f"{face.person.name}_{face.person.id}"
        return Path.joinpath(self.face_dir, f"{file_name}.png")

    def show_faces(self, faces: List, per_row=8):
        n_images = len(faces)
        n_rows = math.ceil(n_images / per_row)

        fig, axes = plt.subplots(n_rows, per_row, figsize=(per_row * 2, n_rows * 2))
        axes = axes.flatten()  # flatten in case of multiple rows

        for i in range(len(axes)):
            axes[i].axis("off")  # hide axes
            if i < n_images:
                img = Image.open(self.get_face_path(faces[i]))

                # If image is BGR (from OpenCV), convert to RGB
                if img.shape[-1] == 3 and img.dtype == "uint8":
                    img = img[..., ::-1]  # simple BGR -> RGB
                axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[i].set_title(faces[i].person.name, fontsize=8)

        plt.tight_layout()
        plt.show()
