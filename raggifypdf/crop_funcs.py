import base64
import io
import os
from getpass import getpass

from raggifypdf.image_funcs import Cropped


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
class CropAnalyzerGeneral:
    def __init__(self, prompt: str = None):
        self.prompt = prompt

    # get a description of the crop
    def describe_crop(self, crop: Cropped) -> str:
        pass


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
    def describe_crop(self, crop: Cropped) -> str:
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


class CropAnalyzerGCP(CropAnalyzerGeneral):
    def __init__(self, prompt: str = None, **kwargs):
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part, SafetySetting

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
            )
        if "location" not in kwargs:
            location = "europe-west3"
        else:
            location = kwargs["location"]

        if "project" not in kwargs:
            print("No GCP project-id given in kwarg 'project', please enter below")
            project = getpass("GCP-project-id: ")
        else:
            project = kwargs["project"]
        vertexai.init(project=project, location=location)

        if "model" in kwargs:
            self.model = GenerativeModel(kwargs["model"])
        else:
            self.model = GenerativeModel("gemini-1.5-flash-002")

        if "generation_config" in kwargs:
            self.generation_config = kwargs["generation_config"]
        else:
            self.generation_config = {
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
            }

    # get a description of the crop
    def describe_crop(self, crop: Cropped) -> str:
        from vertexai.generative_models import Part

        encoded_image = Part.from_data(
            mime_type="image/png", data=convert_crop_to_base64(crop)
        )
        response = self.model.generate_content(
            [encoded_image, self.prompt],
            generation_config=self.generation_config,
            stream=False,
        )
        return response.text


class CropAnalyzerHF(CropAnalyzerGeneral):
    def __init__(self, prompt: str = None, **kwargs):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
            )

    # get a description of the crop
    def describe_crop(self, crop: Cropped) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=crop.img, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_texts[0]


if __name__ == "__main__":
    from PIL import Image

    test = Cropped(
        img=Image.open(
            "/home/niels/dev/pdf_analyzer/analyze_pdfs/data/test/description.png"
        ),
        label="schematic",
        bbox=[0, 0, 1, 1],
    )
    analyzer = CropAnalyzerHF()
    print(analyzer.describe_crop(test))
