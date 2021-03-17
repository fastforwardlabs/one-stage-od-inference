import os
import sys
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import SessionState
from welcome import welcome
from fpn import fpn
from rpn import rpn
from nms import nms
from src.data_utils import (
    gather_data_artifacts,
    create_pickle,
    load_pickle,
    save_figure_images,
)
from src.app_utils import (
    PRESET_IMAGES,
    APP_PAGES,
    get_feature_map_plot,
)

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():

    session_state = SessionState.get(img_option=None, img_path=None)

    step_option = st.sidebar.selectbox(
        label="Step through the app here",
        options=APP_PAGES,
    )

    if step_option == APP_PAGES[0]:
        img_option, img_path = welcome(session_state, PRESET_IMAGES)

        session_state.img_option = img_option
        session_state.img_path = img_path

    elif step_option == APP_PAGES[1]:

        # # for saving preset objects only
        # session_state.data_artifacts = gather_data_artifacts(session_state.img_path)
        # session_state.img_paths = save_figure_images(session_state)
        # create_pickle(
        #     session_state,
        #     f"data/{session_state.img_option}/{session_state.img_option}.pkl",
        # )

        if not session_state.img_option in PRESET_IMAGES.keys():
            session_state.data_artifacts = gather_data_artifacts(session_state.img_path)
            session_state.img_paths = save_figure_images(session_state)
        else:
            session_state = load_pickle(
                f"data/{session_state.img_option}/{session_state.img_option}.pkl"
            )

        fpn(session_state)

    elif step_option == APP_PAGES[2]:
        session_state = load_pickle(
            f"data/{session_state.img_option}/{session_state.img_option}.pkl"
        )
        rpn(session_state)

    elif step_option == APP_PAGES[3]:
        session_state = load_pickle(
            f"data/{session_state.img_option}/{session_state.img_option}.pkl"
        )

        nms(session_state)


if __name__ == "__main__":
    main()