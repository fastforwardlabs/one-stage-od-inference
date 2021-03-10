import os
import sys
import streamlit as st

from src.app_utils import get_feature_map_plot


def feature_extraction(session_state):
    """
    Step 1 - Feature Extraction with FPN

    NOTE TO SELF: show FPN diagram, explanatory text (resolution vs. semantic value), then
    use columns (?) to animate the visualization of feature maps, then text explaining it

    https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c

    """
    st.title("Rich, Multi-scale Feature Extraction")
    # NOTE TO SELF - need to open up with more direct verbiage...RetinaNet uses FPN, here's what it is.
    st.write(
        "Feature extraction, or the refinement of raw images into semantically rich, low dimensional representations is central to any computer vision \
        pipeline. In classification tasks, low dimensional feature maps are desireable as they enable efficient computation. However, in object detection, \
        simply extracting low dimension features sacrifices the granular resolution needed to detect small objects."
    )
    st.write(
        "Therefore, it is critical for feature representations to \
        remain scale invariant because objects may be present at different sizes in an image."
    )
    with st.beta_expander("Visualized Feature Maps"):
        st.image(session_state.img_path, caption="Original Image")
        st.subheader("Feature Maps per FPN Level")
        # st.pyplot(get_feature_map_plot(session_state.data_artifacts["model"]))
        st.pyplot(session_state.data_artifacts["feature_map_fig"])

    # with st.beta_expander("ResNet"):
    #     st.write(
    #         "A deep Residual Network (ResNet) is a CNN architecture that introduces shortcut connections between layers that facilitate effective training of deep neural networks. \
    #          While RetinaNet makes use of a ResNet backbone, the type of backbone network is not critical"
    #     )
    #     st.image(
    #         "images/vgg_diagram.png",
    #         caption="Feature Extraction from CNN",
    #     )
    #     st.write(
    #         "[Image Credit](https://www.jeremyjordan.me/object-detection-one-stage/)"
    #     )
    with st.beta_expander("Feature Pyramid Network (FPN) Backbone"):
        st.write(
            "To overcome this challenge, RetinaNet makes use of a Feature Pyramid Network (FPN) on top of an off-the-shelf CNN feature extractor, typically ResNet."
        )
        st.image(
            "images/retinanet_architecture.png",
            caption="The one-stage RetinaNet network architecture",
        )

    # st.write(session_state.img_option)
    # st.image(session_state.img_path)

    return