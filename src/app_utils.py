import os
import numpy as np
import matplotlib.pyplot as plt

from src.retinanet import retinanet_resnet50_fpn


PRESET_IMAGES = {
    "giraffe": "images/giraffe.jpg",
    "soccer": "images/soccer_img.jpg",
    "snowboard": "images/snowboard.jpg",
}

APP_PAGES = [
    "0. Welcome",
    "1. Feature Extraction",
    "2. Anchor Box Generation",
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