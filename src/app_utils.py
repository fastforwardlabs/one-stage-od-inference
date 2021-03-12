import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.retinanet import retinanet_resnet50_fpn


PRESET_IMAGES = {
    "giraffe": "images/giraffe.jpg",
    "soccer": "images/soccer_img.jpg",
    "snowboard": "images/snowboard.jpg",
}

APP_PAGES = [
    "0. Welcome",
    "1. Feature Extraction",
    "2. Region Proposal Network",
    "3. Post Processing",
]


def convert_bb_spec(xmin, ymin, xmax, ymax):
    """
    Convert a bounding box representation

    """

    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin

    return x, y, width, height


def plot_predictions(image, outputs, label_map, nms_off=False):
    """
    Overlay bounding box predictions on an image

    Args:
        image - PIL image as RGB format
        outputs - boxes, scores, labels output from predict()
        label_map - list mapping of idx to label name
        nms_off - indicates if visualization is with or without NMS

    """

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    np.random.seed(24)
    colors = np.random.uniform(size=(len(label_map), 3))
    boxes, scores, labels = outputs.values()

    for i, box in enumerate(boxes):

        x, y, width, height = convert_bb_spec(*box)
        top = y + height

        patch = patches.Rectangle(
            (x, y),
            width,
            height,
            edgecolor=colors[labels[i]],
            linewidth=1,
            facecolor="none",
        )
        ax.add_patch(patch)

        if nms_off:
            continue

        ax.text(
            x,
            y,
            label_map[labels[i]],
            color=colors[labels[i]],
            fontweight="semibold",
            horizontalalignment="left",
            verticalalignment="bottom",
        )
        ax.text(
            x + width,
            y + height,
            round(scores[i].item(), 2),
            color=colors[labels[i]],
            fontweight="bold",
            horizontalalignment="right",
            verticalalignment="top",
        )

    plt.axis("off")
    plt.show()


def sample_feature_maps(features, n):
    """
    Given a list of features from model.backbone, this function randomly samples N
    feature maps from each FPN layer.

    Args:
        features (List[Tensor]) - list of features from model.backbone
        n (int) - number of samples per layer

    Returns:
        samples (dict) - a dict containing N feature maps per layer

    """
    np.random.seed(42)

    samples = {}
    for i, plevel in enumerate(features):

        maps = plevel.squeeze(0).detach().numpy()
        channel_idx = np.random.choice(range(maps.shape[0]), n)
        samples[i] = maps[channel_idx, :, :]

    return samples


def plot_feature_samples(samples):
    """
    Given a dict of samples from sample_feature_maps(), this function plots
    them in a organized columnar fashion.

    """

    rows = samples[0].shape[0]
    columns = len(samples)

    fig = plt.figure(figsize=(16, 18))
    grid = plt.GridSpec(nrows=rows, ncols=columns, figure=fig, wspace=0.3, hspace=0.1)

    for i, layer in samples.items():
        for j in range(rows):
            ax = plt.subplot(grid[j, i])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

            if j == 0:
                ax.set_title(f"P{i+3}: {layer.shape[1]} x {layer.shape[2]}")

            ax.imshow(layer[j, :, :], aspect="auto")

    return fig


def get_feature_map_plot(model):
    """
    Uses the computed features saved in retinanet.model to plot samples
    of the feature map.

    Note - must pass the output of this function to st.pyplot() to render in streamlit

    """

    samples = sample_feature_maps(model.viz_artifacts["features"], 7)
    fig = plot_feature_samples(samples)

    return fig


def plot_anchors(image, boxes, sample):
    """
    Overlay anchor boxes on the original image

    Args:
        image - PIL image as RGB format
        boxes - Tensor(N,4) of all anchor box specs
        sample - percentage of anchorboxes to randomly select for visualization

    """

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    num_samples = int(sample * boxes.shape[0])
    sample_idxs = torch.randint(low=0, high=boxes.shape[0], size=(num_samples,))
    boxes = boxes[sample_idxs]

    for i, box in enumerate(boxes):

        x, y, width, height = convert_bb_spec(*box)
        top = y + height

        patch = patches.Rectangle(
            (x, y),
            width,
            height,
            edgecolor="red",
            linewidth=1,
            facecolor="none",
        )
        ax.add_patch(patch)

    plt.axis("off")

    return fig


def plot_pyramid_level_anchors(
    pyramid_level_idx, img, image_size, strides, cell_anchors, pred_boxes
):
    """
    This function overlays a full set of anchor boxes (all aspect ratios and sizes) on a given image
    centered atop each object detected in the image. Specifiying the pyramid level allows you to
    visualize the size of the anchor boxes relative to the feature map resolution (shown by grid size)

    Args:
        pyramide_level_index (int)
        img (PIL.Image.Image)
        image_size (torch.Size)
        strides (List[List[torch.Tensor]])
        cell_anchors (List[torch.Tensor])
        pred_boxes (torch.Tensor)

    Returns:
        fig - matplotlib figure

    """

    figsize = [round(i / 100) for i in image_size]
    fig, ax = plt.subplots(1, figsize=(figsize[::-1]))
    ax.grid(True)
    ax.set_xticks(np.arange(0, image_size[1], strides[pyramid_level_idx][1]))
    ax.set_yticks(np.arange(0, image_size[0], strides[pyramid_level_idx][0]))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.title(f"P{pyramid_level_idx+3} - ")
    ax.imshow(img.resize(image_size[::-1]), aspect="auto", alpha=0.4)

    box_centers = [
        ((box[2] - box[0]) / 2 + box[0], (box[3] - box[1]) / 2 + box[1])
        for box in pred_boxes.tolist()
    ]

    for box_center in box_centers:
        for i, box in enumerate(cell_anchors[pyramid_level_idx]):
            x, y, width, height = convert_bb_spec(*box)

            x_offset = box_center[0]
            y_offset = box_center[1]

            patch = patches.Rectangle(
                (x + x_offset, y + y_offset),
                width,
                height,
                alpha=0.5,
                edgecolor="red",
                linewidth=2,
                facecolor="none",
            )
            ax.add_patch(patch)

    return fig
