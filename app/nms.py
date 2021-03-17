import os
import streamlit as st


def nms(session_state):

    st.title("Object Detection Post Processing")

    total_anchors = sum(
        [
            grid["grid_size"][0] * grid["grid_size"][1] * 9
            for grid in [
                session_state.data_artifacts["anchor_plots"][f"P{i+3}"]["fig_stats"]
                for i in range(5)
            ]
        ]
    )

    st.write(
        f"From the previous step, we know that RetinaNet proposed and simultaneously made inference on a total of **{total_anchors:,}** anchor boxes across all pyramid levels. For this reason, it is \
        likely that objects in the image may be positively predicted by multiple anchor boxes from different pyramid levels causing duplicative detections. To refine the overlapping detections, RetinaNet \
        applies a series of post-processing steps to the model ouput."
    )

    st.write(
        "First, RetinaNet selects up to 1,000 anchor boxes from each feature pyramid level that have the highest predicted probability of any class after thresholding detector confidence at 0.05. \
        Then, the top predictions from all levels are merged together, and an algorithm called *Non-Maximum Suppression (NMS)* is applied with a threshold of 0.5 to yield the final set of non-redundant detections."
    )

    with st.beta_expander("How does NMS work?"):
        st.write(
            "Non-Maximum Suppression takes in the refined list of predicted _bounding boxes_ across all FPN levels along with the corresponding class prediction and confidence score. Then for each class independently:"
        )

        st.write("1. Select the bounding box with the highest confidence score.")
        st.write(
            "2. Compare the _Intersection over Union (IoU)_ of that bounding box with all other bounding boxes."
        )
        col1, col2, col3 = st.beta_columns(3)
        # with col1:
        #     st.image("images/iou.png")
        with col2:
            with st.beta_container():
                st.image("images/iou.png")
                st.write(
                    "      [Image Credit](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)"
                )
        st.write("3. For each box, if IoU is > 0.5, drop that bounding box.")
        st.write(
            "4. For the remaining boxes, repeat steps 1 - 3 until all bounding boxes are accounted for."
        )
        st.text("")

    with st.beta_expander("Post-process detections", expanded=True):
        st.write(
            "The image displayed below shows the top predictions from each FPN level _before_ non-max suppression is applied. Click the \
                checkbox to see the final detections:"
        )

        cola, colb, colc = st.beta_columns([1, 2, 1])
        with colb:
            nms_checkbox = st.checkbox("Apply Non-Max Suppression")

        if nms_checkbox:
            st.image(session_state.img_paths["nms"]["with_nms"])
        else:
            st.image(session_state.img_paths["nms"]["without_nms"])

    return
