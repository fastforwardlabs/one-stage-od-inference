import pickle

from src.model_utils import get_inference_artifacts
from src.app_utils import get_feature_map_plot


def gather_data_artifacts(img_path):
    """
    Uses specified image path to load the image and gather all data artifacts to be used
    throughout the app.

    Args:
        img_path

    Returns:
        data_artifacts

    """

    outputs, model = get_inference_artifacts(img_path)
    feature_map_figure = get_feature_map_plot(model)

    data_artifacts = {
        "outputs": outputs,
        "model": model,
        "feature_map_fig": feature_map_figure,
    }

    return data_artifacts


def create_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj