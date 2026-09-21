"""Smoke test for Streamlit import."""
import streamlit as st

def test_streamlit_installed():
    assert hasattr(st, 'write')

