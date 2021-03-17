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
)
from src.app_utils import (
    PRESET_IMAGES,
    APP_PAGES,
    get_feature_map_plot,
)

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():

    step_option = st.sidebar.selectbox(
        label="Step through the app here",
        options=APP_PAGES,
    )
    session_state = SessionState.get()

    if step_option == APP_PAGES[0]:

        session_state = welcome(session_state, PRESET_IMAGES)
        session_state._set_path_attributes()

    elif step_option == APP_PAGES[1]:

        if session_state.img_option not in PRESET_IMAGES.keys():

            if not hasattr(session_state, "data_artifacts"):
                session_state._prepare_data_assets()

        else:
            # uncomment these two lines to build preset data pickles
            # also need to manually add sub-directories
            # session_state._prepare_data_assets()
            # create_pickle(
            #     session_state,
            #     f"{session_state.ROOT_PATH}/{session_state.img_option}.pkl",
            # )

            session_state = load_pickle(session_state.pkl_path)

        fpn(session_state)

    elif step_option == APP_PAGES[2]:
        rpn(session_state)

    elif step_option == APP_PAGES[3]:
        nms(session_state)


if __name__ == "__main__":
    main()