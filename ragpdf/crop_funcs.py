from PIL import Image
from .image_funcs import Cropped


# the analyzer for the crops
# model-dependent class
# needs to be implemented per vendor
class CropAnalyzerOpenAI:
    def __init__(self, prompt: str = None):
        self.prompt = prompt

    # get a description of the crop
    def describe_crop(self, crop: Cropped):
        pass

    # check if crop is meaningful
    def meaningful_crop(self, crop: Cropped):
        pass


class CropAnalyzerOpenAI:
    def __init__(self, prompt: str = None):
        self.prompt = prompt

    # get a description of the crop
    def describe_crop(self, crop: Cropped):
        pass

    # check if crop is meaningful
    def meaningful_crop(self, crop: Cropped):
        pass
