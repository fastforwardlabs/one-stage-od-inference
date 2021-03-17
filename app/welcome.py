import os
import sys
import shutil
import streamlit as st
from PIL import Image

from src.data_utils import create_directory_structure


def welcome(session_state, preset_images):
    st.title("Object Detection Inference: _Visualized_")
    st.write(
        "Object detection is a critical task in computer vision - powering use cases such as autonomous driving, surveillance, \
        defect detection in manufacturing, medical image analysis, and more. This application offers a step-by-step walkthrough to help \
        visualize the inference workflow of a single-stage object detector. Specifically, we'll see how a pre-trained [RetinaNet](https://arxiv.org/abs/1708.02002) \
        model processes an image to quickly and accurately detect objects while also exploring fundamental object detection concepts along the way."
    )

    with st.beta_expander("Object Detection - A Brief Overview", expanded=True):
        st.write(
            "In the field of computer vision, object detection refers to the task of classifying and localizing distinct objects of interest within an image. \
            Traditionally, state-of-the-art object detectors have been based on a two-stage architecture, where the first stage narrows the search space by \
            generating a sparse set of candidate object location proposals, and the second stage then classifies the narrowed down list of proposals. While this approach produces \
            models with high accuracy, there is a significant tradeoff in speed, making these detectors impractical for real time use cases."
        )
        st.write(
            "In contrast, one-stage detectors must localize and classify a much larger set of densely sampled candidate object locations all in one pass. By design, these detectors can \
            attain faster prediction speeds, but must overcome the inherent challenge of efficiently disambiguating between background noise and actual \
            object signal *without* the privilege of a independent proposal system."
        )

    with st.beta_expander("RetinaNet"):
        st.image(
            "images/retinanet_architecture.png",
            caption="The one-stage RetinaNet network architecture",
        )
        st.write(
            "RetinaNet was the first one-stage object detection model to uphold the speed benefits of a one-stage detector, while surpassing the accuracy of (at the time) all existing \
            state-of-the-art two-stage detectors. This was achieved by piecing together standard components like a Feature Pyramid Network (FPN) backbone, a Region Proposal Network (RPN), dedicated classification \
            and box regression sub networks, and introducing a novel loss function called Focal Loss."
        )
        st.write(
            "In this application, we'll step through the inference process highlighting the inner working \
            of these fundamental concepts - select an image below to get started!"
        )

    with st.beta_expander("Let's Get Started"):

        st.write(
            "To get started, select one of the preset images _or_ upload your own image to use throughout the application:"
        )

        col1, col2 = st.beta_columns([1, 2])

        with col1:
            img_setting = st.radio(
                label="Select an image setting",
                options=("Preset image", "Upload your own"),
            )

        with col2:
            if img_setting == "Preset image":
                img_option = st.selectbox(
                    "Select an preset image from the list below",
                    [key.capitalize() for key in preset_images.keys()],
                    index=0,
                )

                # display selected image
                img_option = img_option.lower()
                img_path = preset_images[img_option]
                st.image(img_path)

                session_state.img_option = img_option
                session_state.img_path = img_path

            elif img_setting == "Upload your own":
                uploaded_image = st.file_uploader(
                    label="Upload your image here",
                    type=["png", "jpg"],
                    accept_multiple_files=False,
                )

                if uploaded_image is not None:

                    # display image
                    img = Image.open(uploaded_image)
                    st.image(img, caption="Uploaded Image")

                    # with col1:
                    #     st.text("")
                    #     st.text("")
                    #     st.text("")
                    #     confirm = st.button("Confirm this image")

                    # if confirm:

                    # create folder directory to hold assets
                    img_option = "custom"
                    parent_dir = f"data/{img_option}"

                    if os.path.exists(parent_dir):
                        shutil.rmtree(parent_dir)

                    create_directory_structure(img_option)

                    # save image to directory
                    file_type = uploaded_image.name.split(".")[-1]

                    if file_type == "png":
                        img = img.convert("RGB")

                    img_path = f"data/{img_option}/{img_option}.jpg"
                    img.save(img_path, "jpeg")

                    session_state.img_option = img_option
                    session_state.img_path = img_path

    return session_state
