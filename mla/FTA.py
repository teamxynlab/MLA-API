from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as pyplot
from matplotlib.patches import Rectangle, Arrow
import math
import cv2
from mla.utils import get_random_filename
import matplotlib

matplotlib.use("agg")


def arrow_face_text(image, filename, cfg, model):
    results = model.detect([image], verbose=1)
    r = results[0]

    pyplot.figure(figsize=(10,10))
    pyplot.imshow(image)
    pyplot.title("Face to Text")

    ax = pyplot.gca()

    face_centers = []
    text_centers = []
    frame_corners = []
    count = 0

    for id in list(r["class_ids"]):
        box = list(r["rois"])[count]
        count += 1

        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1

        if id == 1:
            face_x = (x1 + x2)//2
            face_y = (y1 + y2)//2
            face_center = [face_x, face_y]
            face_centers.append(face_center)

            rect = Rectangle((x1, y1), width, height, fill=False, color="red")
            ax.add_patch(rect)

        elif id == 2:
            text_x = (x1 + x2)//2
            text_y = (y1 + y2)//2
            text_center = [text_x, text_y]
            text_centers.append(text_center)

            rect = Rectangle((x1, y1), width, height, fill=False, color="yellow")
            ax.add_patch(rect)

        elif id == 3:
            frame_corners.append([x1, x2, y1, y2])
            rect = Rectangle((x1, y1), width, height, fill=False, color="violet")
            ax.add_patch(rect)

    faces_to_texts = []

    for frame in frame_corners:
        x1, x2, y1, y2 = frame
        face_centers_filtered, text_centers_filtered = [], []
        num_faces, num_text = 0, 0

        for face in face_centers:
            if face[0] < x2 and face[0] > x1 and face[1] < y2 and face[1] > y1:
                num_faces += 1
                face_centers_filtered.append(face)

        for text in text_centers:
            if text[0] < x2 and text[0] > x1 and text[1] < y2 and text[1] > y1:
                num_text += 1
                text_centers_filtered.append(text)
        
        if num_faces >= num_text:
            for face in face_centers_filtered:
                if text_centers_filtered:
                    nearest_text = text_centers_filtered[0]
                    shortest_x = abs(face[0] - nearest_text[0])
                    shortest_y = abs(face[1] - nearest_text[1])
                    shortest_distance = math.sqrt(shortest_x**2 + shortest_y**2)

                    for text in text_centers_filtered:
                        distance_x = abs(face[0] - text[0])
                        distance_y = abs(face[1] - text[1])
                        distance = math.sqrt(distance_x**2 + distance_y**2)

                        if distance < shortest_distance:
                            shortest_distance = distance
                            nearest_text = text

                    face_to_text = [face, nearest_text]
                    faces_to_texts.append(face_to_text)

        elif num_faces < num_text:
            for text in text_centers_filtered:
                if face_centers_filtered:
                    nearest_face = face_centers_filtered[0]
                    shortest_x = abs(text[0] - nearest_face[0])
                    shortest_y = abs(text[1] - nearest_face[1])
                    shortest_distance = math.sqrt(shortest_x**2 + shortest_y**2)

                    for face in face_centers_filtered:
                        distance_x = abs(face[0] - text[0])
                        distance_y = abs(face[1] - text[1])
                        distance = math.sqrt(distance_x**2 + distance_y**2)

                        if distance < shortest_distance:
                            shortest_distance = distance
                            nearest_face = face

                    face_to_text = [nearest_face, text]
                    faces_to_texts.append(face_to_text)

    for face_to_text in faces_to_texts:
        face, text = face_to_text

        face_x, face_y = face
        text_x, text_y = text

        length_x = abs(face_x - text_x)
        length_y = abs(face_y - text_y)

        if face_x > text_x: #face is to the right of text
            length_x *= -1
        
        if face_y > text_y: #face is below text
            length_y *= -1

        arrow = Arrow(face_x, face_y, length_x, length_y, color="cornflowerblue")
        ax.add_patch(arrow)

    pyplot.savefig(f"static/{filename}.png", bbox_inches="tight", pad_inches=-0.5, orientation= "landscape")
    pyplot.close()


def FTA():
    image = cv2.imread("static/input.png")
    width = image.shape[1]
    filename = get_random_filename(10)

    class PredictionConfig(Config):
        NAME = "manga_cfg"
        NUM_CLASSES = 1 + 3
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir="models", config=cfg)
    model.load_weights("models/IS.h5", by_name=True)

    arrow_face_text(image, filename, cfg, model)
    
    return {"success": True, "filename": filename}
