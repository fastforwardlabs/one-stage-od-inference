import os
import pickle

import app.SessionState
from app.SessionState import SessionState
from src.model_utils import COCO_LABELS
from src.model_utils import get_inference_artifacts
from src.app_utils import get_feature_map_plot, get_anchor_plots, plot_predictions


def gather_data_artifacts(img_path):
    """
    Uses specified image path to load the image and gather all data artifacts to be used
    throughout the app.

    Args:
        img_path

    Returns:
        data_artifacts

    """

    with_nms, without_nms = [
        get_inference_artifacts(img_path, nms_setting) for nms_setting in [False, True]
    ]
    feature_map_figure = get_feature_map_plot(with_nms["model"])
    anchor_plots = get_anchor_plots(
        with_nms["image"],
        with_nms["model"].anchor_generator,
        with_nms["outputs"]["boxes"],
        with_nms["model"].viz_artifacts["features"],
    )
    prediction_figures = {
        k: plot_predictions(
            image=v["image"],
            outputs=v["outputs"],
            label_map=COCO_LABELS,
            nms_off=False if k == "with_nms" else True,
        )
        for k, v in {"with_nms": with_nms, "without_nms": without_nms}.items()
    }

    data_artifacts = {
        "outputs": with_nms["outputs"],
        "image": with_nms["image"],
        "viz_artifacts": with_nms["model"].viz_artifacts,
        "feature_map_fig": feature_map_figure,
        "anchor_plots": anchor_plots,
        "prediction_figures": prediction_figures,
    }

    return data_artifacts


def save_figure_images(session_state):

    DIR = f"data/{session_state.img_option}"
    img_paths = {}

    # save feature map figure
    fpn_img_path = os.path.join(DIR, "fpn", "feature_map_fig.png")
    img_paths["fpn"] = fpn_img_path
    session_state.data_artifacts["feature_map_fig"].savefig(fpn_img_path)

    # save anchor box figures
    rpn = {}
    for pyramid_level, data in session_state.data_artifacts["anchor_plots"].items():
        rpn_img_path = os.path.join(DIR, "rpn", f"{pyramid_level}.png")
        rpn[pyramid_level] = rpn_img_path
        data["fig"].savefig(rpn_img_path)

    img_paths["rpn"] = rpn

    # save final prediction figures
    nms = {}
    for nms_setting, fig in session_state.data_artifacts["prediction_figures"].items():
        nms_img_path = os.path.join(DIR, "nms", f"{nms_setting}.png")
        nms[nms_setting] = nms_img_path
        fig.savefig(nms_img_path)

    img_paths["nms"] = nms

    return img_paths


def create_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj