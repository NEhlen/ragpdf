from raggifypdf.image_funcs import CropperModelYOLO
from ultralytics import YOLO
import pymupdf
from PIL import Image

# Example on how to crop images from pdf using a YOLO model

# load a cropper to find image patches in pd
# you can use your own trained YOLO model
cropper = CropperModelYOLO(
    confidence_threshold=0.2, model=YOLO("raggifypdf/model/yolo11n_fallback.pt")
)

pdf = pymupdf.open("examples/example_data/1809.01886v1.pdf")

# load page
page = pdf[3]

# convert page to pixmap
pix = page.get_pixmap(dpi=200)
# convert pixmap to PIL.Image
with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as img:
    crops = cropper.crop(img)

for count, crop in enumerate(crops):
    crop.img.save(f"examples/example_data/1809.01886v1_crops_page3/crop_{count}.png")
