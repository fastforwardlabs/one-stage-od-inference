import os
import pickle

from src.model_utils import COCO_LABELS
from src.model_utils import get_inference_artifacts
from src.app_utils import get_feature_map_plot, get_anchor_plots, plot_predictions


def create_directory_structure(dirname):
    """ Builds skeleton folder structure to hold artifacts used in the app"""

    os.makedirs(f"data/{dirname}")
    for subdir in ["fpn", "rpn", "nms"]:
        os.makedirs(f"data/{dirname}/{subdir}")


def gather_data_artifacts(img_path):
    """
    Uses specified image path to load the image and gather all data artifacts to be used
    throughout the app.

    Args:
        img_path

    Returns:
        data_artifacts

    """

    inference_artifacts = get_inference_artifacts(img_path, False)
    feature_map_figure = get_feature_map_plot(inference_artifacts["model"])
    anchor_plots = get_anchor_plots(
        inference_artifacts["image"],
        inference_artifacts["model"].anchor_generator,
        inference_artifacts["outputs"]["boxes"],
        inference_artifacts["model"].viz_artifacts["features"],
    )
    prediction_figures = {
        k: plot_predictions(
            image=inference_artifacts["image"],
            outputs=inference_artifacts["outputs"],
            label_map=COCO_LABELS,
            nms_off=False if k == "with_nms" else True,
        )
        for k in ["with_nms", "without_nms"]
    }

    data_artifacts = {
        "outputs": inference_artifacts["outputs"],
        "image": inference_artifacts["image"],
        "viz_artifacts": inference_artifacts["model"].viz_artifacts,
        "feature_map_fig": feature_map_figure,
        "anchor_plots": anchor_plots,
        "prediction_figures": prediction_figures,
    }

    return data_artifacts


def create_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj