import os
import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "frontend")
_slider = components.declare_component("continuous_slider", path=_COMPONENT_DIR)

def continuous_slider(label, min_value=0, max_value=100, value=0, step=1, key=None):
    return _slider(label=label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
