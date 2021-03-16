import os
import pickle

import app.SessionState
from app.SessionState import SessionState
from src.model_utils import get_inference_artifacts
from src.app_utils import get_feature_map_plot, get_anchor_plots


def gather_data_artifacts(img_path):
    """
    Uses specified image path to load the image and gather all data artifacts to be used
    throughout the app.

    Args:
        img_path

    Returns:
        data_artifacts

    """

    outputs, model, image = get_inference_artifacts(img_path)
    feature_map_figure = get_feature_map_plot(model)
    anchor_plots = get_anchor_plots(
        image, model.anchor_generator, outputs["boxes"], model.viz_artifacts["features"]
    )

    data_artifacts = {
        "outputs": outputs,
        "image": image,
        # "model": model,
        "viz_artifacts": model.viz_artifacts,
        "feature_map_fig": feature_map_figure,
        "anchor_plots": anchor_plots,
    }

    return data_artifacts


def save_figure_images(session_state):

    DIR = f"data/{session_state.img_option}"
    img_paths = {}

    # save feature maps
    fpn_img_path = os.path.join(DIR, "fpn", "feature_map_fig.png")
    img_paths["fpn"] = fpn_img_path
    session_state.data_artifacts["feature_map_fig"].savefig(fpn_img_path)

    # save anchorbox images
    rpn = {}
    for pyramid_level, data in session_state.data_artifacts["anchor_plots"].items():
        rpn_img_path = os.path.join(DIR, "rpn", f"{pyramid_level}.png")
        rpn[pyramid_level] = rpn_img_path
        data["fig"].savefig(rpn_img_path)

    img_paths["rpn"] = rpn

    return img_paths


def create_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj