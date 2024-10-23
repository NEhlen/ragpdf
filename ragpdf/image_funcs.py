from ultralytics import YOLO
from dataclasses import dataclass
from PIL import Image


@dataclass
class Cropped:
    img: Image.Image
    label: str


def cutout_schemas_single_image(
    img: Image.Image, model: YOLO, confidence_threshold: float = 0.1
):
    results = model(img, show=False, conf=confidence_threshold)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    clss = results.boxes.cls.cpu().tolist()
    cropped_imgs = []
    if boxes is not None:
        for box, _cls in zip(boxes, clss):
            cropped = img.crop(box)
            cropped_imgs.append(Cropped(img=cropped, label=_cls))
    return cropped_imgs, results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = Image.open("data/test/montage_7740995-0.png")
    model = YOLO("models/yolo11n_best.pt")
    crops, results = cutout_schemas_single_image(img, model, confidence_threshold=0.8)
    plt.imshow(results.plot())
