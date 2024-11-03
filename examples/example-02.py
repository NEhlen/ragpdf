from raggifypdf.crop_funcs import Cropped, CropAnalyzerHF

from PIL import Image

# Example on how to get description of an image

test = Cropped(
    img=Image.open("examples/test-image-01.png"),
    label="schematic",
    bbox=[0, 0, 1, 1],
)
analyzer = CropAnalyzerHF()
print(analyzer.describe_crop(test))
