# This module is adapted from the original Github Gist module
# found at https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92

"""Hack to add per-session state to Streamlit.

Usage
-----

>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'

Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'

"""

import os

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server

from src.data_utils import gather_data_artifacts, create_pickle


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _set_path_attributes(self):

        ROOT_PATH = f"data/{self.img_option}"

        self.img_option = self.img_option
        self.ROOT_PATH = ROOT_PATH
        self.pkl_path = f"{ROOT_PATH}/{self.img_option}.pkl"

    def _save_figure_images(self):

        fig_paths = {}

        # save feature map figure
        fpn_img_path = os.path.join(self.ROOT_PATH, "fpn", "feature_map_fig.png")
        fig_paths["fpn"] = fpn_img_path
        self.data_artifacts["feature_map_fig"].savefig(fpn_img_path)

        # save anchor box figures
        rpn = {}
        for pyramid_level, data in self.data_artifacts["anchor_plots"].items():
            rpn_img_path = os.path.join(self.ROOT_PATH, "rpn", f"{pyramid_level}.png")
            rpn[pyramid_level] = rpn_img_path
            data["fig"].savefig(rpn_img_path)

        fig_paths["rpn"] = rpn

        # save final prediction figures
        nms = {}
        for nms_setting, fig in self.data_artifacts["prediction_figures"].items():
            nms_img_path = os.path.join(self.ROOT_PATH, "nms", f"{nms_setting}.png")
            nms[nms_setting] = nms_img_path
            fig.savefig(nms_img_path)

        fig_paths["nms"] = nms

        return fig_paths

    def _prepare_data_assets(self):

        self.data_artifacts = gather_data_artifacts(img_path=self.img_path)
        self.fig_paths = self._save_figure_images()
        self.has_detections = (
            True if len(self.data_artifacts["outputs"]["boxes"]) > 0 else False
        )
        self.data_artifacts.pop("outputs")


def get(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, "_session_infos"):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, "_main_dg") and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, "_main_dg") and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (
                not hasattr(s, "_main_dg")
                and s._uploaded_file_mgr == ctx.uploaded_file_mgr
            )
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object. "
            "Are you doing something fancy with threads?"
        )

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, "_custom_session_state"):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state
