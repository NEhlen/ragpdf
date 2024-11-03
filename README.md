# raggifyPDF

This repository helps with preparing a PDF file for RAG applications.
In the current state it is mostly used to detect images/schematics and replace them with a text description using various LLM options like openAI, GCP or open source models from huggingface. With modern multimodal models it is probably not needed vs just sending an image of a PDF page but this repo enables the use of multimodal models to modify the PDFs and then use them in a pure text RAG-pipeline.

## How to install
In your project folder, create a virtual environment.
If you are using Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```
With Windows use
```
python -m venv .venv
.venv/Scripts/activate
```

Clone the repository with
```git clone https://github.com/NEhlen/ragpdf.git```

Install the repository in your virtual enviroment
```
pip install ./ragpdf[full]
```

## How to use / Quickstart
The package includes several basic Classes used to modify a PDF.
A `CropperModel` used to find and crop schematics in an image of a PDF page, a `CropAnalyzer` model used to turn the cropped schematics into text descriptions and a `PDFModifier` to handle the PDF io, page extraction etc.
There are multiple types of CropperModels and CropAnalyzers depending on what vendor you want to use for the analysis and what kind of Cropping Model you want to use. (see below)

In a python file do the following
```
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
```

## Package Structure
### CropperModels

Currently there are `CropperModelYOLO` and `CropperModelFlorence`. CropperModelYOLO uses a Yolo11 Model for object detection of schematics on the pages. You can supply your own trained model or use the very basic one included with this repo.
CropperModelFlorence uses Microsofts Florence2 to detect schematics on supplied images.
If you want to supply your own CropperModel just inherit from the CropperModel Base class in `raggifypdf.image_funcs` and supply your own initializiation and crop methods (you can also supply your own cleanup method for any postprocessing of the crops).

### CropAnalyzers

There are currently three supported Crop Analyzers `CropAnalyzerOpenAI`, `CropAnalyzerGCP`, `CropAnalyzerHF`.
To use the OpenAI Analyzer, you need an OpenAI API-Key. To use the GCP Analyzer, you need gcloud-cli installed and configured to handle the authorization.
`CropAnalyzerHF` uses Microsofts Phi-3-vision Model.
If you want to supply your own CropAnalyzer you can inherit from CropAnalyzer in `raggifypdf.crop_funcs` and supply your own init and describe_crop methods. Just make sure to return the same types as given in the method definition to ensure compatibility.

### PDFModifier
The PDFModifier class in `raggifypdf.raggify` is meant to manage the PDF i/o and the modification. You can also do this by hand using the CropperModels and CropAnalyzers but the PDFModifier will take care of necessary coordinate transformations of the bounding boxes, overlap checks of detected schematics by the ML CropperModels and actual image-bounding boxes as well as text bounding boxes (to get rid of false detections) and the insert of the text at the correct locations on the pages.

### Cropped

The Package is based on a basic dataclass called `Cropped` in `raggifypdf.image_funcs`. This `Cropped` class holds the cropped image as a PIL Image object, the label given to the crop by the CropperModels as well as the bounding box of the cropped image in relation to the full page (depending on context either in PDF coordinates or in image/pixel coordinates).