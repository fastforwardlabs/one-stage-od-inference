import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.retinanet import retinanet_resnet50_fpn

COCO_LABELS = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def predict(model, image, transform, detection_threshold):
    """
    Use a trained Pytorch detection model to make inference on an input image

    Args:
        model - Pytorch detetection model
        image - PIL image as RGB format
        transform - torchvision Compose object
        detection_threshold - confidence score for anchorbox predictions to be kept

    Returns:
        outputs - dict containing boxes, scores, labels for predictions
    """

    if model.training:
        model.eval()

    image = transform(image).unsqueeze(0)
    outputs = model(image)[0]

    idxs = np.where(outputs["scores"] > detection_threshold)

    boxes = outputs["boxes"][idxs]
    scores = outputs["scores"][idxs]
    labels = outputs["labels"][idxs]
    outputs = {"boxes": boxes, "scores": scores, "labels": labels}

    return outputs


def get_inference_artifacts(img_path, nms_off=False):
    """
    Given an image path, this function makes inference on the image and returns
    both the outputs and the model (with saved artifacts)
    """

    retinanet = retinanet_resnet50_fpn(
        pretrained=True, pretrained_backbone=True, nms_off=nms_off
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    img = Image.open(img_path)

    # resize image to fullsize according to RetinaNet min/max transform
    full_size = retinanet.transform(transform(img).unsqueeze(0))[0].tensors.size()[2:][
        ::-1
    ]
    img = img.resize(full_size)

    outputs = predict(
        model=retinanet,
        image=img,
        transform=transform,
        detection_threshold=0.8,
    )

    inference_artifacts = {"outputs": outputs, "model": retinanet, "image": img}

    return inference_artifacts