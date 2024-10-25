from raggifypdf.raggify import PDFModifier
from raggifypdf.crop_funcs import CropAnalyzerOpenAI
from raggifypdf.image_funcs import CropperModelYOLO
import pymupdf

# load a cropper to find image patches in pdf
cropper = CropperModelYOLO(confidence_threshold=0.8)

# load analyzer that turns cropped images into descriptions
analyzer = CropAnalyzerOpenAI(model_id="gpt-4o-mini")

# load pdf
pdf = pymupdf.open("examples/test_01.pdf")

# initialize modifier
modifier = PDFModifier(pdf, cropper, analyzer)

# modify full pdf
modified_pdf = modifier.modify()

# save modified pdf
modifier.save_pdf("examples/test_01-modified.pdf")
