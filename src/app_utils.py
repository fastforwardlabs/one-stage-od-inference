import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.retinanet import retinanet_resnet50_fpn


PRESET_IMAGES = {
    "giraffe": "data/giraffe/giraffe.jpg",
    "snowboard": "data/snowboard/snowboard.jpg",
}

APP_PAGES = [
    "0. Welcome",
    "1. Feature Extraction",
    "2. Region Proposal Network",
    "3. Non-Maximum Suppression",
    "4. References",
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

    Returns:
        maplotlib Figure

    """

    fig, ax = plt.subplots(1)
    ax.imshow(image, aspect="auto")

    np.random.seed(24)
    colors = np.random.uniform(size=(len(label_map), 3))
    boxes, scores, labels = outputs.values()

    for i, box in enumerate(boxes.detach().numpy()):

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
    plt.tight_layout()

    return fig


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

    fig = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(nrows=rows, ncols=columns, figure=fig, wspace=0.15, hspace=0.1)

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

    Returns:
        matplotlib Figure

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


def get_anchor_plots(image, anchor_generator, pred_boxes, features):

    anchor_plots = {}
    for i in range(5):
        fig, fig_stats = plot_pyramid_level_anchors(
            i,
            img=image,
            image_size=anchor_generator.anchor_artifacts["image_size"],
            strides=anchor_generator.anchor_artifacts["strides"],
            grid_sizes=anchor_generator.anchor_artifacts["grid_sizes"],
            cell_anchors=anchor_generator.cell_anchors.copy(),
            pred_boxes=pred_boxes,
            features=features,
            anchor_sizes=anchor_generator.sizes,
        )
        anchor_plots[f"P{i+3}"] = {"fig": fig, "fig_stats": fig_stats}

    return anchor_plots


def plot_pyramid_level_anchors(
    pyramid_level_idx,
    img,
    image_size,
    strides,
    grid_sizes,
    cell_anchors,
    pred_boxes,
    features,
    anchor_sizes,
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
        features (List[torch.Tensor])
        anchors_sizes (List[tuple])

    Returns:
        fig - matplotlib figure

    """

    figsize = [round(i / 100) for i in image_size]
    figsize[0] = figsize[0] * 2
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(figsize[::-1]))
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, image_size[1], strides[pyramid_level_idx][1]))
    ax1.set_yticks(np.arange(0, image_size[0], strides[pyramid_level_idx][0]))
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax1.title.set_text(f"P{pyramid_level_idx+3} - Anchor Grid with Anchor Box Overlay")
    ax1.imshow(img.resize(image_size[::-1]), aspect="auto", alpha=0.4)

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
            ax1.add_patch(patch)

    # normalize and resize a feature map for visualization
    fm = features[pyramid_level_idx][:, 66, :, :].detach().numpy()
    fm = np.rollaxis(fm, 0, 3)
    fm_norm = (
        ((fm - fm.min()) * (1 / (fm.max() - fm.min()) * 255)).astype("uint8").squeeze(2)
    )
    fm_img = Image.fromarray(fm_norm).resize(image_size[::-1])
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, image_size[1], strides[pyramid_level_idx][1]))
    ax2.set_yticks(np.arange(0, image_size[0], strides[pyramid_level_idx][0]))
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.title.set_text(f"P{pyramid_level_idx+3} - Sample Feature Map")

    ax2.imshow(
        fm_img,
        aspect="auto",
    )
    plt.tight_layout()

    fig_stats = {
        "image_size": list(image_size),
        "stride": [stride.item() for stride in strides[pyramid_level_idx]],
        "grid_size": list(grid_sizes[pyramid_level_idx]),
        "anchor_sizes": anchor_sizes[pyramid_level_idx],
    }

    return fig, fig_stats
