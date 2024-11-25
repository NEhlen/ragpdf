from raggifypdf.raggify import PDFModifier
from raggifypdf.crop_funcs import CropAnalyzerOpenAI, CropAnalyzerGCP, CropAnalyzerHF
from raggifypdf.image_funcs import CropperModelYOLO
import pymupdf

# load a cropper to find image patches in pdf
cropper = CropperModelYOLO(confidence_threshold=0.8)

# load analyzer that turns cropped images into descriptions
# to use the OpenAI Analyzer, the OPENAI_API_KEY enviroment variable needs to be set

# analyzer = CropAnalyzerOpenAI(model_id="gpt-4o-mini")
# analyzer = CropAnalyzerGCP()
analyzer = CropAnalyzerHF()

# load pdf
pdf = pymupdf.open("examples/example_data/1809.01886v1.pdf")

# initialize modifier
modifier = PDFModifier(pdf, cropper, analyzer)

# modify full pdf
# if you just want to modify some pages, use the pages argument with a
# list of pages to
modified_pdf = modifier.modify(pages="full")

# save modified pdf
modifier.save_pdf("examples/example_data/1809.01886v1_modified.pdf")
