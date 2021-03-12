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
    "backpack",
    "umbrella",
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
    "dining table",
    "toilet",
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


def get_inference_artifacts(img_path):
    """
    Given an image path, this function makes inference on the image and returns
    both the outputs and the model (with saved artifacts)
    """

    retinanet = retinanet_resnet50_fpn(
        pretrained=True, pretrained_backbone=True, nms_off=False
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    img = Image.open(img_path)

    # resize image to fullsize
    full_size = retinanet.transform(transform(img).unsqueeze(0))[0].tensors.size()[2:][
        ::-1
    ]
    img = img.resize(full_size)

    outputs = predict(
        model=retinanet, image=img, transform=transform, detection_threshold=0.9
    )

    return outputs, retinanet, img