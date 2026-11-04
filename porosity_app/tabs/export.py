"""The Export tab: result downloads and the NCR form.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import datetime

import streamlit as st

from porosity_fe.reporting import (
    INDICATION_TYPES,
    STRUCTURAL_CLASSES,
    DispositionPolicy,
    _sanitise_filename_component,
    _serialise_payload_csv,
    _serialise_payload_json,
    build_export_payload,
    build_ncr_record,
    download_filename_stem,
    serialise_ncr_json,
    serialise_ncr_markdown,
    serialise_ncr_pdf,
)

from ..config import _config_to_key
from ..runners import fe_vtu_bytes_cached
from ..summaries import _load_factor_text, ncr_delamination_available
from ..tabs.common import _placeholder_tab


def _build_export_tab(result: dict | None, layup_for_title: str):
    if result is None:
        _placeholder_tab()
        return
    payload = build_export_payload(result)
    export_stem = download_filename_stem(payload)
    payload_json = _serialise_payload_json(payload)
    st.download_button(
        "Download JSON",
        data=payload_json,
        file_name=f"{export_stem}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_export_json",
    )
    st.download_button(
        "Download CSV",
        data=_serialise_payload_csv(payload),
        file_name=f"{export_stem}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_export_csv",
    )
    if result.get("fe_field") is not None:
        fe_slug = result["fe_loading"]
        if (result.get("fe_thermal") or {}).get("mode") == "combined":
            fe_slug += "_residual"
        st.download_button(
            "Download FE results (VTU)",
            data=fe_vtu_bytes_cached(_config_to_key(result["config"]), result),
            file_name=f"{export_stem}_fe_{fe_slug}.vtu",
            mime="application/octet-stream",
            use_container_width=True,
            key="dl_export_vtu",
            help=(
                "Binary VTK XML (.vtu) of the FE mesh for ParaView: "
                "displacement, porosity, element stress and strain, failure "
                "index and ply data, nodal stress and strain recovered "
                "within each ply, and the residual stress of a thermal run."
            ),
        )
    else:
        st.caption("No FE results to download as VTU: the FE solve was skipped.")
    with st.expander("Preview JSON"):
        st.code(payload_json, language="json")

    st.divider()
    st.subheader("NCR validation summary")
    st.caption(
        "Generate a concise analysis summary to **attach to an NCR**. "
        "It carries the porosity validation and a *recommended* "
        "disposition path for the MRB — it is not a full NCR form and "
        "does not issue a final disposition. Part/serial/work-order "
        "identification stays on the parent NCR."
        + (" A delamination gets a fracture assessment and no recommended "
           "path: the MRB dispositions it."
           if ncr_delamination_available(result) else "")
    )

    # A delamination of the part with a fracture load factor gets a fracture
    # assessment, never a recommended path (the porosity alone otherwise).
    delamination_ncr = ncr_delamination_available(result)
    if result.get("delamination") is not None and not delamination_ncr:
        st.caption(
            "The delamination has no fracture load factor (no toughness "
            "data), so this summary covers the porosity only.")

    with st.form("ncr_form"):
        c1, c2 = st.columns(2)
        with c1:
            prepared_by = st.text_input(
                "Prepared by", value="",
                help="Engineer preparing this analysis summary.",
            )
            ncr_reference = st.text_input(
                "Parent NCR reference (optional)", value="",
                help="Cross-reference to the NCR this attaches to.",
            )
        with c2:
            structural_class = st.selectbox(
                "Structural classification",
                options=list(STRUCTURAL_CLASSES),
                index=0,
                help=(
                    "Drives required substantiation. Primary "
                    "structure escalates to customer/DER concurrence "
                    "for any Use-As-Is."
                ),
            )
        indication, required = "porosity", None
        if delamination_ncr:
            c3, c4 = st.columns(2)
            with c3:
                indication = st.selectbox(
                    "Indication type", options=list(INDICATION_TYPES),
                    index=list(INDICATION_TYPES).index("combined"),
                    key="ncr_indication",
                    help=("porosity: a recommended path from the void "
                          "content and knockdown. delamination: a fracture "
                          "assessment, λ_c against the required load factor, "
                          "with no recommended path: the MRB dispositions it. "
                          "combined: both; the recommended path covers the "
                          "porosity only."),
                )
            with c4:
                required = st.number_input(
                    "Required load factor", min_value=0.0001, value=None,
                    step=0.1, format="%.4g", key="ncr_required_load_factor",
                    placeholder="required",
                    help=(f"The multiple of the analysis load "
                          f"({_load_factor_text(result, 1.0)}) the part must "
                          f"carry without delamination growth, from your "
                          f"structural substantiation; there is no default. "
                          f"The predicted λ_c multiplies the same load."),
                )
        note = st.text_area(
            "Engineer note (optional)", value="",
            help="Any observation context to record on the summary.",
        )
        make_ncr = st.form_submit_button(
            "Generate summary", type="primary",
            use_container_width=True,
        )

    if not make_ncr:
        return
    if indication != "porosity" and required is None:
        st.error("Enter the required load factor to assess the "
                 "delamination's fracture margin.")
        return

    meta = {
        "prepared_by": prepared_by,
        "ncr_reference": ncr_reference,
        "structural_class": structural_class,
        "note": note,
        "date": datetime.date.today().isoformat(),
        "layup": layup_for_title,
    }
    if indication != "porosity":
        meta["indication"] = indication
        meta["required_load_factor"] = float(required)
    ncr = build_ncr_record(result, meta)
    dp = ncr["recommended_disposition"]
    assessment = ncr.get("delamination", {}).get("assessment")
    if dp is not None:
        scope = " (porosity only)" if assessment is not None else ""
        st.warning(
            f"**Recommended disposition path{scope}:** {dp['path']}  \n"
            f"{dp['rationale']}"
        )
        st.caption(
            "Disposition policy: "
            + DispositionPolicy.from_dict(dp["policy"]).describe())
        st.info(dp["disclaimer"])
    if assessment is not None:
        st.warning(
            f"**Delamination: no recommended disposition path.** "
            f"{assessment['statement']}"
        )
        st.info(assessment["disclaimer"])

    ncr_md = serialise_ncr_markdown(ncr)
    stem = (
        _sanitise_filename_component(ncr_reference.strip())
        or export_stem
    )
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download PDF",
            data=serialise_ncr_pdf(ncr),
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_ncr_pdf",
        )
    with dl2:
        st.download_button(
            "Download Markdown",
            data=ncr_md,
            file_name=f"{stem}.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_ncr_md",
        )
    with dl3:
        st.download_button(
            "Download JSON",
            data=serialise_ncr_json(ncr),
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_ncr_json",
        )
    with st.expander("Preview summary"):
        st.markdown(ncr_md)
