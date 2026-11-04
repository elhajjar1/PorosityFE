"""Helpers shared by the tabs.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import streamlit as st


def _show_figure(fig, file_stem: str, key: str) -> None:
    """Render ``fig`` in the page and offer it as a PNG download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.download_button(
        "Download PNG", data=buf.getvalue(), file_name=f"{file_stem}.png",
        mime="image/png", key=key,
    )


def _placeholder_tab():
    st.info("Run an analysis to populate this tab.")


def _metric_row(rows) -> None:
    """``rows`` of ``(label, value, help)`` as one row of metrics."""
    if rows:
        for col, (label, value, help_text) in zip(st.columns(len(rows)), rows,
                                                  strict=True):
            col.metric(label, value, help=help_text)
