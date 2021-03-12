import os
import sys
import streamlit as st

from src.app_utils import plot_anchors


def anchor_boxs(session_state):

    st.title("Efficient Region Proposal Network")

    st.pyplot(
        plot_anchors(
            image=session_state.data_artifacts["image"],
            boxes=session_state.data_artifacts["viz_artifacts"]["anchors"][0],
            sample=0.003,
        )
    )

    st.image("images/anchor_box_grid.jpg")

    return
