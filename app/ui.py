"""UI helpers for Streamlit dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if not css_path.exists():
        return
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: Optional[str] = None) -> None:
    help_html = f"<div class=\"metric-help\">{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    st.markdown(f"<div class=\"section-title\">{text}</div>", unsafe_allow_html=True)


def content_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
