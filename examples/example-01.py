from raggifypdf.raggify import PDFModifier
from raggifypdf.crop_funcs import CropAnalyzerOpenAI
from raggifypdf.image_funcs import CropperModelYOLO
import pymupdf

# load a cropper to find image patches in pdf
cropper = CropperModelYOLO(confidence_threshold=0.8)

# load analyzer that turns cropped images into descriptions
# to use the OpenAI Analyzer, the OPENAI_API_KEY enviroment variable needs to be set
analyzer = CropAnalyzerOpenAI(model_id="gpt-4o-mini")

# load pdf
pdf = pymupdf.open("examples/test_01.pdf")

# initialize modifier
modifier = PDFModifier(pdf, cropper, analyzer)

# modify full pdf
# if you just want to modify some pages, use the pages argument with a
# list of pages to
modified_pdf = modifier.modify(pages="full")

# save modified pdf
modifier.save_pdf("examples/test_01-modified.pdf")
