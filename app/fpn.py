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
    st.title("1. Multi-scale Feature Extraction")
    st.write(
        "Feature extraction is central to any computer vision pipeline and is traditionally performed using deep networks of stacked convolutional layers \
        that refine raw images into a semantically rich, low dimensional representations. This approach is transferable to object detection, however it must \
        be augmented to maintain scale invariant image representations because in the real-world, objects from the same class can exist at a wide range of sizes depending on their depth in an image. \
        Recognizing objects a varying scales, particularly small objects is a fundamental challenge in object detection. \
        RetinaNet uses a [Feature Pyramid Network (FPN)](https://arxiv.org/pdf/1612.03144.pdf) to solve this problem by extracting feature maps from multiple \
        levels of a [ResNet](https://arxiv.org/pdf/1512.03385.pdf) backbone."
    )

    with st.beta_expander("How Do FPNs Work?", expanded=False):
        st.write(
            "Feature Pyramid Networks exploit the inherent mulit-scale, pyramidal hierarchy of deep CNNs to detect objects at different scales by augmenting a CNN's default, \
            bottom-up composition with a top-down pathway and lateral connections."
        )
        st.image(
            "images/fpn_diagram.png",
            caption="Feature Pyramid Network Architecture",
        )
        st.write(
            "**a. Bottom-up Pathway:** An FPN can be constructed from any deep CNN, but RetinaNet chooses a ResNet architecture. In ResNet, convolutional layers are grouped together \
                into stages by their output size. The bottom-up pathway of the FPN simply extracts a feature map as the output from the last layer of each stage called a pyramid level. \
                RetinaNet constructs a pyramid with levels P$_3$ - P$_7$, where P$_l$ indicates pyramid level and has resolution 2$^l$ lower than the input image. Each pyramid level holds 256 channels, \
                and P$_0$ - P$_2$ are omitted from the FPN because their high dimensionality has substantial impact on memory and computation speed."
        )
        st.write(
            "**b. Top-down Pathway** The top-down pathway regenerates higher resolution features by upsampling spatially coarser, but semantically stronger feature maps from higher pyramid levels. \
            Each feature map is upsampled by a factor of 2 using nearest neighbor upsampling."
        )
        st.write(
            "**c. Lateral Connections** Lateral connections between the two pathways are used to merge feature maps of the same spatial size by element-wise addition. These lateral connections combine \
                the semanticically rich, upsampled feature map from the top-down pathway with accurately localized activations from the bottom-up pathway creating robust, multi-scale features maps \
                to use for inference."
        )

    with st.beta_expander("RetinaNet Feature Maps", expanded=True):
        st.image(session_state.img_path, caption="Original Image")
        st.subheader("Feature Maps per FPN Level")
        st.write(
            "The visual below depicts seven sampled feature maps from each of the five levels in the FPN. We see that features from the third pyramid level (P3) \
            maintain higher resolution, but semantically weaker attributes which are useful for detecting small objects. In contrast, features from the seventh level \
            (P7) hold much lower resolution, but semantically stronger attributes, making them effective for identifying larger objects."
        )
        # st.pyplot(session_state.data_artifacts["feature_map_fig"])
        st.image(session_state.img_paths["fpn"])

    return
