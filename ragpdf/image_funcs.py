from ultralytics import YOLO
from dataclasses import dataclass
from PIL import Image


@dataclass
class Cropped:
    img: Image.Image
    label: str


class CropperModel:
    def __init__(
        self,
        type: str = "florence",
        confidence_threshold: float = 0.1,
        **kwargs,
    ):
        self.conf_thresh = confidence_threshold
        self.type = type
        if self.type == "YOLO":
            if "model" not in kwargs:
                raise AttributeError("If type is YOLO, model needs to be given")
            else:
                self.model = kwargs["model"]
        elif self.type == "florence":
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
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
        else:
            raise ValueError("<type> must be either YOLO or florence")

    def crop_yolo(self, img: Image.Image):
        results = self.model(img, show=False, conf=self.conf_thresh)[0]
        boxes = results.boxes.xyxy.cpu().tolist()
        clss = results.boxes.cls.cpu().tolist()
        return boxes, clss, results

    def crop_florence(self, img: Image.Image):
        inputs = self.processor(
            text=self.task + self.prompt,
            images=img,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=(img.width, img.height),
        )

        boxes = parsed_answer[self.task]["bboxes"]
        clss = parsed_answer[self.task]["labels"]

        # filter out boxes that are text
        box_tup = list(zip(boxes, clss))
        box_tup_schematics = list(filter(lambda x: x[1] == "schematics", box_tup))
        box_tup_text = list(filter(lambda x: x[1] == "text", box_tup))
        box_tup_schematics = filter(
            lambda x: max([compute_overlap(x[0], bt[0])[0] for bt in box_tup_text])
            < 0.9,
            box_tup_schematics,
        )
        boxes, clss = zip(*box_tup_schematics)
        return list(boxes), list(clss), parsed_answer

    def crop(self, img: Image.Image):
        if self.type == "YOLO":
            boxes, clss, result = self.crop_yolo(img)
            return *self.cleanup_crops(img, boxes, clss), result
        elif self.type == "florence":
            boxes, clss, result = self.crop_florence(img)
            return *self.cleanup_crops(img, boxes, clss), result

    def cleanup_crops(self, img, boxes, clss):
        # remove crops with too high overlap
        # remove crops that are whole page if there are smaller
        # crops present

        # filter out boxes that are mostly the whole image
        boxes, clss = zip(
            *filter(
                lambda x: compute_overlap(
                    x[0],
                    [0, 0, img.size[0], img.size[1]],
                )[0]
                < 0.7,
                zip(boxes, clss),
            )
        )

        # sort boxes by size from largest to smallest
        _, boxes, clss in zip(
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
                p = zip(
                    *filter(
                        lambda x: compute_overlap(test_box, x[0])[2] < 0.8,
                        zip(boxes, clss),
                    )
                )
                boxes, clss = p
                boxes = list(boxes)
                clss = list(clss)
        return kept_boxes, kept_clss


# compute overlap between two boxes,
# return relative overlaps
# intersection/union, intersection/area_box0, intersection/area_box1
def compute_overlap(box0, box1):
    XA1, YA1, XA2, YA2 = box0
    XB1, YB1, XB2, YB2 = box1
    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    SA = (XA2 - XA1) * (YA2 - YA1)
    SB = (XB2 - XB1) * (YB2 - YB1)
    SU = SA + SB - SI
    return SI / SU, SI / SA, SI / SB


def cutout_schemas_single_image(img: Image.Image, cropper: CropperModel):
    boxes, clss, results = cropper.crop(img)
    cropped_imgs = []
    if boxes is not None:
        for box, _cls in zip(boxes, clss):
            cropped = img.crop(box)
            cropped_imgs.append(Cropped(img=cropped, label=_cls))
    return cropped_imgs, results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = Image.open(
        "../analyze_pdfs/data/images/train/gebrauchsanweisung_7700359-4.png"
    )
    model = YOLO("../analyze_pdfs/models/yolo11n_best.pt")
    cropper = CropperModel(type="YOLO", confidence_threshold=0.8, model=model)
    # cropper = CropperModel(type="florence", confidence_threshold=0.8)
    crops, results = cutout_schemas_single_image(img, cropper)
