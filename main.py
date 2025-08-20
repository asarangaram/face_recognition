from pathlib import Path
from store_init import init_store


def available_faces(path: str):
    return [
        (file.stem.split("_")[0], str(file))
        for file in Path(path).rglob("*")
        if file.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]


class Config:
    FACE_STORE_DIR = "face_store"


if __name__ == "__main__":
    rebuild_store = True
    preserve_past = not rebuild_store
    recogniser = init_store(Config.FACE_STORE_DIR, preserve_past=False)
    if rebuild_store:
        faces = available_faces(
            "/home/anandas/demos/degirum_hailo_examples/assets/Friends_dataset"
        )
        face_ids = recogniser.register_faces_no_batch(faces)
        print(f"Available faces: {len(faces)}")
        print(f"Registerred faces: {len(face_ids)}")
