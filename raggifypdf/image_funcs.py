import importlib.resources as pkg_resources
import logging
from dataclasses import dataclass
from typing import Union

from PIL import Image
from pymupdf import Rect
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Cropped:
    img: Image.Image
    label: str
    bbox: list[float | int]


class CropperModel:
    def __init__(
        self,
        **kwargs,
    ):
        pass

    def crop(self, img: Image.Image) -> list[Cropped]:
        """
        Cropper specific crop function.
        Input:
            img: pillow Image
        Output:
            list[Cropped]: list of Cropped object data classes holding cropped image, bounding box and label
        """
        pass

    def cleanup(
        self, boxes: list[list[float | int] | Rect], clss: list[str]
    ) -> tuple[list[list[float | int] | Rect], list[str]]:
        """
        Cropper specific cleanup function.
        Input:
            boxes: list of bounbing boxes (either list of ints/floats of four corners or Rect Object from pymudpdf)
            clss: list of label strings
        Output:
            (boxes, clss): filtered lists
        """
        return boxes, clss


class CropperModelFlorence(CropperModel):
    def __init__(
        self,
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large-ft",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large-ft",
            trust_remote_code=True,
        )
        self.task = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.prompt = "schematics, text"

    def cleanup(self, boxes, clss):
        # filter out boxes that are text by checking overlap
        box_tup = list(zip(boxes, clss))
        box_tup_schematics = list(filter(lambda x: x[1] == "schematics", box_tup))
        box_tup_text = list(filter(lambda x: x[1] == "text", box_tup))
        if box_tup_text:
            box_tup_schematics = filter(
                lambda x: max([compute_overlap(x[0], bt[0])[0] for bt in box_tup_text])
                < 0.9,
                box_tup_schematics,
            )
        boxes, clss = zip(*box_tup_schematics)
        return boxes, clss

    def crop(self, img: Image.Image, **kwargs) -> list[Cropped]:
        # process image with task and prompt
        inputs = self.processor(
            text=self.task + self.prompt,
            images=img,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)
        # generate tokens
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **kwargs,
        )
        # decode generated tokens
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        # parse the generated answer
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=(img.width, img.height),
        )

        boxes = parsed_answer[self.task]["bboxes"]
        clss = parsed_answer[self.task]["labels"]

        # clean up by removing doubled labels (in particular images that are also labeled
        # as text)
        boxes, clss = self.cleanup(boxes, clss)
        # standard cleanup, see below
        boxes, clss = cleanup_crops(img, list(boxes), list(clss))

        # if there are any found bboxes, return list of Cropped objects
        if boxes:
            crops = []
            for box, label in zip(boxes, clss):
                crops.append(Cropped(img=img.crop(box), bbox=box, label=label))
            return crops
        else:
            return []


class CropperModelYOLO(CropperModel):
    def __init__(
        self,
        confidence_threshold: float = 0.1,
        **kwargs,
    ):
        self.conf_thresh = confidence_threshold
        if "model" not in kwargs:
            print("loading fallback YOLO")
            with pkg_resources.path(
                "raggifypdf.model", "yolo11n_fallback.pt"
            ) as model_path:
                self.model = YOLO(str(model_path))
        else:
            self.model: YOLO = kwargs["model"]

    def crop(self, img: Image.Image) -> list[Cropped]:
        results = self.model(img, show=False, conf=self.conf_thresh)[0]
        boxes = results.boxes.xyxy.cpu().tolist()
        clss = results.boxes.cls.cpu().tolist()
        boxes, clss = cleanup_crops(img, boxes, clss)

        if boxes:
            crops = []
            for box, label in zip(boxes, clss):
                crops.append(Cropped(img=img.crop(box), bbox=box, label=label))
            return crops
        else:
            return []


def cleanup_crops(
    img: Image.Image,
    boxes: list[Union[int, float]],
    clss: list[str],
) -> tuple[list[Union[int, float]], list[str]]:
    """
    Cleans up a list of bounding boxes and their labels for a given image.
    Ensures bounding boxes that stretch across most of the image are filtered out.
    Ensures bounding boxes that are nearly fully covered by other bounding boxes
    are filtered out.
    Returns filtered lists of bounding boxes and their corresponding lavels
    """
    # filter out boxes that are mostly the whole image
    temp = list(
        zip(
            *filter(
                lambda x: compute_overlap(
                    x[0],
                    [0, 0, img.size[0], img.size[1]],
                )[0]
                < 0.7,
                zip(boxes, clss),
            )
        )
    )
    if not temp:
        return [], []

    boxes, clss = temp

    # sort boxes by size from largest to smallest
    _, boxes, clss = zip(
        *sorted(
            zip(
                [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes],
                boxes,
                clss,
            ),
            reverse=True,
        )
    )
    boxes = list(boxes)
    clss = list(clss)
    if len(boxes) <= 1:
        return boxes, clss

    # filter out boxes that are mostly inside bigger boxes
    kept_boxes, kept_clss = [], []
    while boxes:

        test_box = boxes.pop(0)
        test_label = clss.pop(0)
        kept_boxes.append(test_box)
        kept_clss.append(test_label)
        if boxes:
            p = list(
                zip(
                    *filter(
                        lambda x: compute_overlap(test_box, x[0])[2] < 0.8,
                        zip(boxes, clss),
                    )
                )
            )
            if p:
                boxes, clss = p
                boxes = list(boxes)
                clss = list(clss)
            else:
                boxes = []
                clss = []
    return kept_boxes, kept_clss


# compute overlap between two boxes,
# return relative overlaps
# intersection/union, intersection/area_box0, intersection/area_box1
def compute_overlap(
    box0: Union[Rect, list[Union[int, float]]],
    box1: Union[Rect, list[Union[int, float]]],
) -> tuple[float, float, float]:
    """
    Compute Overlap between wo bounding boxes.
    Give the bounding boxes in xyxy-Format, ideally as pymupdf Rects,
    alternatively as lists of ints/floats
    returns the relative overlaps

    Returns: (intersection/union, intersecion/area_box0, intersection/area_box1)
    """
    XA1, YA1, XA2, YA2 = box0
    XB1, YB1, XB2, YB2 = box1
    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    SA = (XA2 - XA1) * (YA2 - YA1)
    SB = (XB2 - XB1) * (YB2 - YB1)
    SU = SA + SB - SI
    return SI / SU, SI / SA, SI / SB
