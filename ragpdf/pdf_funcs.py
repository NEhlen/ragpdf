import pymupdf
import sys
import os
import glob
from pathlib import Path
from PIL import Image


def convert_pdf_to_image(
    pdf_path: str,
    saved_path: Path,
    remove_old_files: bool = True,
    dpi: int = None,
) -> list[Image.Image]:
    os.makedirs(saved_path, exist_ok=True)

    files = saved_path.glob("./*")
    if remove_old_files:
        for f in files:
            os.remove(f)

    try:
        pymupdf.TOOLS.mupdf_warnings()
        doc = pymupdf.open(pdf_path)
        warnings = pymupdf.TOOLS.mupdf_warnings()
        if warnings:
            print(warnings)
            raise RuntimeError()

        pictures = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            pix.save(saved_path / f"{pdf_path.with_suffix("").name}-{page.number}.png")
            pictures.append(pix)
        return pictures

    except Exception as e:
        print("error when opening the pdf file {}".format(pdf_path.name))
        return e


def get_images_from_pdf(pdf_path: str) -> dict[str, Image.Image]:
    try:
        pymupdf.TOOLS.mupdf_warnings()
        doc = pymupdf.open(pdf_path)
        warnings = pymupdf.TOOLS.mupdf_warnings()
        if warnings:
            print(warnings)
            raise RuntimeError()
        images = {}
        for count, page in enumerate(doc):
            imgs = page.get_images()
            images[f"page_{count}"] = imgs
        return images

    except Exception as e:
        print(f"error when extracting images from pdf at {pdf_path}")
        return e
