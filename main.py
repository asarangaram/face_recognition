import argparse
from pathlib import Path
from store_init import init_store



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Store Initializer")
    parser.add_argument(
        "--rebuild-store",
        action="store_true",
        help="Rebuild the face store from dataset (default: False)",
    )
    parser.add_argument(
        "--face-store-dir",
        type=str,
        default=f"{Path.home()}/.local/share/colan_apps/store/face_db",
        help="Directory to store faces (default: face_store)",
    )
    parser.add_argument(
        "--faces",
        nargs="+",
        required=False,
        help="One or more dataset directories containing face images",
    )
    args = parser.parse_args()

    rebuild_store = args.rebuild_store
    preserve_past = not rebuild_store
    recogniser = init_store(args.face_store_dir, preserve_past=preserve_past)

    if rebuild_store and args.faces:
        all_faces = []
        for path in args.faces:
            for file in Path(path).rglob("*"):
                if file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    all_faces.append((file.stem.split("_")[0], str(file)))

        face_ids = recogniser.register_faces_no_batch(all_faces)
        print(f"Available faces: {len(all_faces)}")
        print(f"Registered faces: {len(face_ids)}")
