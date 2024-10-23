from PIL import Image
from .image_funcs import Cropped
import os
from openai import OpenAI
import base64, io


def convert_crop_to_png(crop: Cropped):
    with io.BytesIO() as output:
        crop.img.save(output, format="PNG")
        return output.getvalue()


def convert_crop_to_base64(crop: Cropped):
    png_img = convert_crop_to_png(crop)
    return base64.b64encode(png_img).decode("utf-8")


# the analyzer for the crops
# model-dependent class
# needs to be implemented per vendor
class CropAnalyzerOpenAI:
    def __init__(self, prompt: str = None, **kwargs):
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
            )
        if "api_key" not in kwargs:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            self.client = OpenAI(api_key=kwargs["api_key"])

    # get a description of the crop
    def describe_crop(self, crop: Cropped):
        encoded_image = convert_crop_to_base64(crop)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            model="gpt-4o",
        )
        return chat_completion.choices[0].message.content


class CropAnalyzerGeneral:
    def __init__(self, prompt: str = None):
        self.prompt = prompt

    # get a description of the crop
    def describe_crop(self, crop: Cropped):
        pass
