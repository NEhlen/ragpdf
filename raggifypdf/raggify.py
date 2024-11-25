import importlib.resources as pkg_resources
import io
import logging
import os
from typing import Union

import pymupdf
import tqdm
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from dataclasses import dataclass

from raggifypdf.crop_funcs import CropAnalyzerGeneral, Cropped
from raggifypdf.image_funcs import YOLO, CropperModel, compute_overlap


@dataclass
class PageCrop:
    crops: list[Cropped]
    descriptions: list[str]


class PDFModifier:
    def __init__(
        self,
        pdf: pymupdf.Document,
        cropper: CropperModel,
        analyzer: CropAnalyzerGeneral,
    ):
        self.cropper = cropper
        self.analyzer = analyzer
        self.pdf = pdf

    def get_image_crops(self, page_num: int) -> list[Cropped]:
        page = self.pdf[page_num]
        images = page.get_images(full=True)

        crops_imgs = []
        for img in images:
            im = Image.open(io.BytesIO(page.parent.extract_image(img[0])["image"]))
            bbox = page.get_image_bbox(img)
            crops_imgs.append(Cropped(img=im, label="<orig_img>", bbox=bbox))

        return crops_imgs

    def get_detected_crops(self, page_num: int) -> list[Cropped]:
        page = self.pdf[page_num]
        pix = page.get_pixmap(dpi=200)  # get image of page
        page_mat = pymupdf.IRect(pix.irect).torect(
            page.rect
        )  # get matrix to convert image coords to pdf coords

        # get crops
        logging.debug("getting crops")
        with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as img:
            crops_ml = self.cropper.crop(img)

        # convert bbox to pdf frame of reference
        for crop in crops_ml:
            crop.bbox = pymupdf.Rect(*crop.bbox) * page_mat
        return crops_ml

    def get_all_crops(self, page_num: int) -> list[Cropped]:
        page = self.pdf[page_num]
        crops_imgs = self.get_image_crops(page_num)
        crops_ml = self.get_detected_crops(page_num)
        # throw out ml crops that are already covered by image crops
        logging.debug("filtering crops")
        if crops_imgs:
            crops_ml = list(
                filter(
                    lambda crop: max(
                        [
                            compute_overlap(crop.bbox, img_crop.bbox)[1]
                            for img_crop in crops_imgs
                        ]
                    )
                    < 0.5,
                    crops_ml,
                )
            )
        # throw out ml crops that are covered by text boxes
        if page.get_text("blocks"):
            crops_ml = list(
                filter(
                    lambda crop: max(
                        [
                            compute_overlap(
                                crop.bbox, [block[0], block[1], block[2], block[3]]
                            )[1]
                            for block in page.get_text("blocks")
                        ]
                    )
                    < 0.5,
                    crops_ml,
                )
            )
        # combine into one list
        crops_total = crops_imgs
        crops_total.extend(crops_ml)
        return crops_total

    def evaluate_crops(self, crops: list[Cropped]) -> list[str]:
        descriptions = []
        logging.info("Getting descriptions for crops")
        for crop in tqdm.tqdm(crops):
            descriptions.append(self.analyzer.describe_crop(crop))

        return descriptions

    def modify_page(self, page_num: int, crops: list[Cropped], descriptions: list[str]):
        page = self.pdf[page_num]
        logging.info(f"modifying page {page_num}")
        for crop, desc in zip(crops, descriptions):
            page.draw_rect(
                crop.bbox, color=(1, 1, 1), fill=1
            )  # Drawing a white rectangle over the image
            # insert textbox at old image location with image description
            # with elongated box along y-direction to ensure the text fits
            crop_bbox = pymupdf.Rect(
                crop.bbox[0], crop.bbox[1], crop.bbox[2], crop.bbox[3] + 1000
            )
            if crop_bbox.is_valid() and (not crop_bbox.is_empty()):
                inserted = page.insert_textbox(
                    crop_bbox,
                    "[IMG]" + desc + "[/IMG]",
                    fontsize=2,
                    color=(0, 0, 0),
                    align=0,
                )
                if inserted < 0:
                    raise Warning(
                        f"text could not be inserted on page {page_num}, bounding_box {crop.bbox}"
                    )
            else:
                raise Warning(
                    f"Invalid or empty bounding box {crop_bbox} for crop on page {page_num}"
                )
        return True

    def modify(
        self, pages: Union[str, list[int]] = "full"
    ) -> tuple[pymupdf.Document, dict[str, PageCrop]]:
        logging.info("Modifying PDF for RAG use")
        if pages == "full":
            pages = range(len(self.pdf))
        # go through pages
        data_dict = {}
        for page_num in pages:
            # get all crops for page
            crops = self.get_all_crops(page_num)
            # evaluate crops
            descriptions = self.evaluate_crops(crops)
            # add to data dict
            data_dict[page_num] = PageCrop(
                crops=crops,
                descriptions=descriptions,
            )
            # modify page
            self.modify_page(page_num, crops, descriptions)

        return self.pdf, data_dict

    def save_pdf(self, save_path: str) -> pymupdf.Document:
        full_path = os.path.abspath(save_path)
        logging.info(f"Saving PDF to {full_path}")
        self.pdf.save(full_path)
        return self.pdf
