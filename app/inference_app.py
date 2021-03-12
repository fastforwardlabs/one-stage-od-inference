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
from fpn import feature_extraction
from rpn import anchor_boxs
from src.data_utils import gather_data_artifacts, create_pickle, load_pickle
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
        # create_pickle(session_state, f"data/{session_state.img_option}.pkl")

        if not session_state.img_option in PRESET_IMAGES.keys():
            session_state.data_artifacts = gather_data_artifacts(session_state.img_path)
        else:
            session_state = load_pickle(f"data/{session_state.img_option}.pkl")

        feature_extraction(session_state)

    elif step_option == APP_PAGES[2]:
        session_state = load_pickle(f"data/{session_state.img_option}.pkl")
        anchor_boxs(session_state)

    elif step_option == APP_PAGES[3]:
        pass


if __name__ == "__main__":
    main()