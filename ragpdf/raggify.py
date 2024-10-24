import importlib.resources as pkg_resources
import io
import logging

import pymupdf
from PIL import Image
import os
import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ragpdf.crop_funcs import Cropped, CropAnalyzerGeneral
from ragpdf.image_funcs import (
    YOLO,
    CropperModel,
    compute_overlap,
)


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
            inserted = page.insert_textbox(
                pymupdf.Rect(crop.bbox[0], crop.bbox[1], crop.bbox[2], 1000),
                desc,
                fontsize=5,
                color=(0, 0, 0),
                align=0,
            )
            if inserted < 0:
                raise Warning(
                    f"text could not be inserted on page {page_num}, bounding_box {crop.bbox}"
                )
        return True

    def modify(self):
        logging.info("Modifying PDF for RAG use")
        # go through pages
        for page_num, page in enumerate(self.pdf):
            # get all crops
            crops = self.get_all_crops(page_num)
            # evaluate crops
            descriptions = self.evaluate_crops(crops)
            # modify page
            self.modify_page(page_num, crops, descriptions)

        return self.pdf

    def save_pdf(self, save_path: str):
        full_path = os.path.abspath(save_path)
        logging.info(f"Saving PDF to {save_path}")
        self.pdf.save(save_path)
        return self.pdf


def _evaluate_page(
    page: pymupdf.Page,
    cropper: CropperModel,
    **kwargs,
):
    # get actual images
    logger.debug("Getting images on page")
    images = page.get_images(full=True)
    crops_imgs = []
    for img in images:
        im = Image.open(io.BytesIO(page.parent.extract_image(img[0])["image"]))
        bbox = page.get_image_bbox(img)
        crops_imgs.append(Cropped(img=im, label="<orig_img>", bbox=bbox))
    ## find other schematics via ml
    logger.debug("Finding schematics via ML")
    pix = page.get_pixmap(dpi=200)  # get image of page
    page_mat = pymupdf.IRect(pix.irect).torect(
        page.rect
    )  # get matrix to convert image coords to pdf coords

    # get crops
    logging.debug("getting crops")
    with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as img:
        crops_ml = cropper(img)

    # convert bbox to pdf frame of reference
    for crop in crops_ml:
        crop.bbox = pymupdf.Rect(*crop.bbox) * page_mat

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
    crops_imgs.extend(crops_ml)
    return crops_imgs


def replace_images_with_text_in_pdf(
    pdf_path: str,
    method="florence",
    yolo_model: YOLO = None,
    evaluation: str = "OpenAI",
    **kwargs,
):
    pdf = pymupdf.open(pdf_path)
    if evaluation == "OpenAI":
        from ragpdf.crop_funcs import CropAnalyzerOpenAI

        analyzer = CropAnalyzerOpenAI()
    elif evaluation == "GCP":
        from ragpdf.crop_funcs import CropAnalyzerGCP

        analyzer = CropAnalyzerGCP()
    else:
        raise ValueError("<evaluation> needs to be set to a valid Analyzer")

    if method == "florence":
        logger.debug("using florence")
        cropper = CropperModel(type="florence")

    elif method == "YOLO":
        logger.debug("using YOLO")
        if not yolo_model:
            logger.debug("loading fallback YOLO")
            with pkg_resources.path(
                "ragpdf.model", "yolo11n_fallback.pt"
            ) as model_path:
                yolo_model = YOLO(str(model_path))
        cropper = CropperModel(type="YOLO", confidence_threshold=0.8, model=yolo_model)
    else:
        raise ValueError("Method needs to be YOLO or florence")

    for page_num, page in enumerate(pdf):
        crops = _evaluate_page(page, cropper, **kwargs)

        for crop in crops:
            desc = analyzer.describe_crop(crop)
            # desc = "test"

            page.draw_rect(
                crop.bbox, color=(1, 1, 1), fill=1
            )  # Drawing a white rectangle over the image
            inserted = page.insert_textbox(
                pymupdf.Rect(crop.bbox[0], crop.bbox[1], crop.bbox[2], 1000),
                desc,
                fontsize=5,
                color=(0, 0, 0),
                align=0,
            )
            if inserted < 0:
                logging.warning(
                    f"text could not be inserted on page {page_num}, bounding_box {crop.bbox}"
                )

    return pdf
