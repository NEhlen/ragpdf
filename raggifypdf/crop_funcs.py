import base64
import io
import os
from getpass import getpass

from raggifypdf.image_funcs import Cropped


def convert_crop_to_png(crop: Cropped):
    with io.BytesIO() as output:
        crop.img.convert("RGB").save(output, format="PNG")
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
                "You might also get additional context for interpreting the image, but I cannot guarantee that."
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

        if "model_id" in kwargs:
            self.model = kwargs["model_id"]
        else:
            self.model = "gpt-4o-mini"

    # get a description of the crop
    def describe_crop(self, crop: Cropped, context: str = None) -> str:
        encoded_image = convert_crop_to_base64(crop)
        txt = self.prompt
        if context:
            txt += "\n\nCONTEXT:\n" + context
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": txt,
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
            model=self.model,
        )
        return chat_completion.choices[0].message.content


class CropAnalyzerGCP(CropAnalyzerGeneral):
    def __init__(self, prompt: str = None, **kwargs):
        import vertexai
        from vertexai.generative_models import GenerativeModel

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
                "You might also get additional context for interpreting the image, but I cannot guarantee that."
            )
        if "location" not in kwargs:
            location = "europe-west3"
        else:
            location = kwargs["location"]

        if "project" not in kwargs:
            print("No GCP project-id given in kwarg 'project', using None instead")
            project = None
        else:
            project = kwargs["project"]
        vertexai.init(project=project, location=location)

        if "model_id" in kwargs:
            self.model = GenerativeModel(kwargs["model_id"])
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
    def describe_crop(self, crop: Cropped, context: str = None) -> str:
        from vertexai.generative_models import Part

        encoded_image = Part.from_data(
            mime_type="image/png", data=convert_crop_to_base64(crop)
        )

        txt = self.prompt
        if context:
            txt += "\n\nCONTEXT:\n" + context

        response = self.model.generate_content(
            [encoded_image, txt],
            generation_config=self.generation_config,
            stream=False,
        )
        return response.text


class CropAnalyzerHF(CropAnalyzerGeneral):
    def __init__(self, prompt: str = None, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct",
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True
        )

        if prompt:
            self.prompt = prompt
        else:
            self.prompt = (
                "I'm sending you a cropped image from a PDF. "
                "Please try to describe in detail what is shown in the image. "
                "In particular if the image is a schematic from a manual describe the measurements shown and what they are measuring."
                "You might also get additional context for interpreting the image, but I cannot guarantee that."
            )

    # get a description of the crop
    def describe_crop(self, crop: Cropped, context: str = None) -> str:
        txt = self.prompt
        if context:
            txt += "\n\nCONTEXT:\n" + context
        messages = [
            {
                "role": "user",
                "content": "<|image_1|>\n" + txt,
            }
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[crop.img], return_tensors="pt").to(
            self.device
        )
        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "do_sample": False,
        }

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return generated_texts[0]
