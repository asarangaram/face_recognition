from typing import List
import degirum as dg
import degirum_tools


class HostedModel:
    def __init__(self):
        self.inference_host_address = "@local"
        self.zoo_url = "./zoo"
        self.token = '' 
        self.device_type = "HAILORT/HAILO8"


class DetectionModel(HostedModel):
    def __init__(self):
        self.face_det_model_name = "retinaface_mobilenet--736x1280_quant_hailort_hailo8_1"
        super().__init__()


        self.model = dg.load_model(
            model_name=self.face_det_model_name,
            inference_host_address=self.inference_host_address,
            zoo_url=self.zoo_url,
            token=self.token, 
            overlay_color=(0, 255, 0)  # Green color for bounding boxes
        )
        pass

    def scan(self, path: str):
        detected_faces = self.model(path)
        return detected_faces

    def batch_scan(self, path: List[str]):
        detected_faces = self.model.predict_batch(path)
         
        return detected_faces
    

class EmbeddingModel(HostedModel):
    def __init__(self):
        self.face_rec_model_name = "arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
        super().__init__()
        # Load the face recognition model
        self.model = dg.load_model(
            model_name=self.face_rec_model_name,
            inference_host_address=self.inference_host_address,
            zoo_url=self.zoo_url,
            token=self.token
        )
    
    def extract_face_embedding(self, image):
        face_embedding = self.model(image).results[0]["data"][0]
