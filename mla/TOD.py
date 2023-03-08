from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as pyplot
from matplotlib.patches import Rectangle, Arrow
import math
import cv2
from mla.utils import get_random_filename
import matplotlib

matplotlib.use('agg')


def order_frame_halves(unordered_frames):
    ordered_frames = []

    while unordered_frames:
        for frame in unordered_frames:
            for other_frame in unordered_frames:
                if other_frame[3] < frame[1]: # if there is a frame above the current frame
                    break # the current frame is not the next frame to read
            else:
                ordered_frames.append(frame[:-1]) # the current frame is the next frame
                unordered_frames.remove(frame)

    return ordered_frames


def order_text(image, img_width, filename, cfg, model):
    results = model.detect([image], verbose=1)
    r = results[0]

    pyplot.figure(figsize=(10,10))
    pyplot.imshow(image)
    pyplot.title("Order Text")

    ax = pyplot.gca()

    text_centers = []
    total_unordered_frames = []
    count = 0

    for id in list(r['class_ids']):
        box = list(r['rois'])[count]
        count += 1

        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1

        if id == 2:
            text_x = (x1 + x2)//2
            text_y = (y1 + y2)//2
            text_center = [text_x, text_y]
            text_centers.append(text_center)

            rect = Rectangle((x1, y1), width, height, fill=False, color="cornflowerblue")
            ax.add_patch(rect)

        elif id == 3:
            frame_x_center = (x1 + x2)//2
            total_unordered_frames.append([x1, y1, x2, y2, frame_x_center]) # the corners are ordered this way to improve sorting later on

    total_unordered_frames.sort(reverse=True)
    ordered_frames = [] # 1st, 2nd, 3rd, ...

    half_line = img_width//2

    unordered_frames_first = [frame for frame in total_unordered_frames if frame[-1] > half_line] # frames to the right of half line
    unordered_frames_second = [frame for frame in total_unordered_frames if frame[-1] <= half_line] # frames to the left of half line

    ordered_frames_first = order_frame_halves(unordered_frames_first)
    ordered_frames_second = order_frame_halves(unordered_frames_second)

    index = 1

    for frame in ordered_frames_first:
        x1, y1, x2, y2 = frame
        text_centers_filtered = []

        for text in text_centers:
            if text[0] < x2 and text[0] > x1 and text[1] < y2 and text[1] > y1:
                text_centers_filtered.append([text[0], -text[1]])
        
        text_centers_filtered.sort(reverse=True)

        for text in text_centers_filtered:
            pyplot.text(text[0], -text[1], index, color="red")
            index += 1

    for frame in ordered_frames_second:
        x1, y1, x2, y2 = frame
        text_centers_filtered = []

        for text in text_centers:
            if text[0] < x2 and text[0] > x1 and text[1] < y2 and text[1] > y1:
                text_centers_filtered.append([text[0], -text[1]])

        text_centers_filtered.sort(reverse=True)

        for text in text_centers_filtered:
            pyplot.text(text[0], -text[1], index, color="red")
            index += 1

    pyplot.savefig(f"static/{filename}.png", bbox_inches="tight", pad_inches=-0.5, orientation= "landscape")
    pyplot.close()


def TOD():
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

    order_text(image, width, filename, cfg, model)
    
    return {"success": True, "filename": filename}
