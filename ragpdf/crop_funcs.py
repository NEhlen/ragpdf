import base64
import io
import os
from getpass import getpass

from .image_funcs import Cropped


def convert_crop_to_png(crop: Cropped):
    with io.BytesIO() as output:
        crop.img.save(output, format="PNG")
        return output.getvalue()


def convert_crop_to_base64(crop: Cropped):
    png_img = convert_crop_to_png(crop)
    return base64.b64encode(png_img).decode("utf-8")


class CropAnalyzerGeneral:
    def __init__(self, prompt: str = None):
        self.prompt = prompt

    # get a description of the crop
    def describe_crop(self, crop: Cropped):
        pass


# the analyzer for the crops
# model-dependent class
# needs to be implemented per vendor
class CropAnalyzerOpenAI(CropAnalyzerGeneral):
    def __init__(self, prompt: str = None, **kwargs):
        from openai import OpenAI

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
            )
        if "api_key" not in kwargs:
            if "OPENAI_API_KEY" in os.environ:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            else:
                print(
                    "No OpenAI api key found in environment, please enter in terminal"
                )
                api_key = getpass("OpenAI api key: ")
                self.client = OpenAI(api_key=api_key)
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
