from ultralytics import YOLO

PRESET_NAME = "yolo11n.pt"


def get_base_model():
    """create base yolo v11 model and return it"""
    model = YOLO(PRESET_NAME)
    return model
