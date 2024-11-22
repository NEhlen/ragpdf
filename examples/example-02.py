from raggifypdf.crop_funcs import (
    Cropped,
    CropAnalyzerHF,
    CropAnalyzerGCP,
    CropAnalyzerOpenAI,
)

from PIL import Image

# Example on how to get description of an image
# for this example you need to have the gcloud-cli initialized with a gcp project
# if you don't have one, change CropAnalyzerGCP to CropAnalyzerOpenAI if you have
# an openai API-Key or CropAnalyzerHF is you want to run phi3-vision locally
test = Cropped(
    img=Image.open("examples/example_data/1809.01886v1_crops_page3/crop_0.png"),
    label="schematic",
    bbox=[0, 0, 1, 1],
)
analyzer = CropAnalyzerGCP()
print(analyzer.describe_crop(test))

# you can also change the prompt used to generate an image description
analyzer = CropAnalyzerHF(
    prompt="Please give bulletpoints of what's shown in the image"
)
print(analyzer.describe_crop(test))
