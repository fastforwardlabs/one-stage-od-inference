import os
import sys
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import time

st.set_option("deprecation.showPyplotGlobalUse", False)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import SessionState
from app_pages import welcome, fpn, rpn, nms, references
from src.model_utils import COCO_LABELS
from src.app_utils import PRESET_IMAGES, APP_PAGES
from src.data_utils import create_pickle, load_pickle


def main():
    """This function acts as the scaffolding to operate the multi-page Streamlit App"""

    step_option = st.sidebar.selectbox(
        label="Step through the app here:",
        options=APP_PAGES,
    )
    session_state = SessionState.get()

    if step_option == APP_PAGES[0]:

        session_state = welcome(session_state, PRESET_IMAGES)
        session_state._set_path_attributes()

    elif step_option == APP_PAGES[1]:

        with st.spinner("Hang tight while your image is processed!"):

            if session_state.img_option not in PRESET_IMAGES.keys():
                if not hasattr(session_state, "data_artifacts"):
                    session_state._prepare_data_assets()

                    if len(session_state.data_artifacts["outputs"]["boxes"]) == 0:
                        st.error(
                            f"Sorry! The image you uploaded doesn't contain any recognizable objects. \
                            Please try another image that contains one of the following classes: \
                            \n\n {', '.join([label for label in COCO_LABELS if label not in ['N/A', '__background__']])}"
                        )

            else:
                # uncomment these two lines to build preset data pickles
                # # also need to manually add sub-directories
                # session_state._prepare_data_assets()
                # create_pickle(
                #     session_state,
                #     f"{session_state.ROOT_PATH}/{session_state.img_option}.pkl",
                # )

                session_state = load_pickle(session_state.pkl_path)

        fpn(session_state)

    elif step_option == APP_PAGES[2]:

        if session_state.img_option in PRESET_IMAGES.keys():
            session_state = load_pickle(session_state.pkl_path)

        rpn(session_state)

    elif step_option == APP_PAGES[3]:

        if session_state.img_option in PRESET_IMAGES.keys():
            session_state = load_pickle(session_state.pkl_path)

        nms(session_state)

    elif step_option == APP_PAGES[4]:

        references()


if __name__ == "__main__":
    main()