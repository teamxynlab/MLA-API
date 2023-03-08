from mrcnn.config import Config
from mrcnn.visualize import save_image
from mrcnn.model import MaskRCNN
import cv2
from mla.utils import get_random_filename


def IS():
    class PredictionConfig(Config):
        NAME = "manga_cfg"
        NUM_CLASSES = 1 + 3
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="models", config=cfg)
    model.load_weights("models/IS.h5", by_name=True)
    class_names = ["BG", "face", "text", "frame"]

    image = cv2.imread("static/input.png")
    results = model.detect([image], verbose=1)
    r = results[0]

    filename = get_random_filename(10)
    
    save_image(image, filename, r["rois"], r["masks"], r["class_ids"],r["scores"], class_names, save_dir="static",scores_thresh=0.9,mode=0)
    
    return {"success": True, "filename": filename}
