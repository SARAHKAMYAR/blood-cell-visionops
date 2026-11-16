"""Blood Cell VisionOps: Streamlit review dashboard.

Talks to the FastAPI service for live analysis and reads stored analyses
straight from PostgreSQL. Deliberately has no torch/ultralytics dependency so
the dashboard container stays small.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg2
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
IMAGE_DIR = Path(os.getenv("BCV_IMAGES_DIR", ROOT / "BCCD_Dataset" / "BCCD" / "JPEGImages"))
METRICS_DIR = ROOT / "reports" / "metrics"
EVAL_DIR = ROOT / "runs" / "eval"

COLORS = {"RBC": "#2a78d6", "WBC": "#eb6834", "Platelets": "#1baf7a"}  # CVD-validated
QC_COLORS = {"PASS": "#16A34A", "REVIEW": "#F59E0B", "FAIL": "#DC2626"}

st.set_page_config(page_title="Blood Cell VisionOps", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _db_params() -> dict | str | None:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    try:
        return dict(st.secrets["postgres"])
    except Exception:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def load_analyses() -> pd.DataFrame:
    params = _db_params()
    if params is None:
        raise RuntimeError("No DATABASE_URL or [postgres] secrets configured")
    conn = psycopg2.connect(params) if isinstance(params, str) else psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT *, image_jpeg IS NOT NULL AS has_image FROM blood_cell_detections "
                "ORDER BY created_at DESC, id DESC"
            )
            cols = [c[0] for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols).drop(columns=["image_jpeg"], errors="ignore")
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def load_image(analysis_id: int) -> bytes | None:
    params = _db_params()
    conn = psycopg2.connect(params) if isinstance(params, str) else psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT image_jpeg FROM blood_cell_detections WHERE id = %s", (int(analysis_id),))
            row = cur.fetchone()
            return bytes(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


@st.cache_data(ttl=10, show_spinner=False)
def api_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return None


def _as_obj(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        label = det.get("class", "?")
        color = COLORS.get(label, "#FFFFFF")
        draw.rectangle(bbox, outline=color, width=3)
        text = f"{label} {det.get('confidence', 0):.2f}"
        tx, ty = bbox[0], max(0, bbox[1] - 12)
        tw = draw.textlength(text, font=font)
        draw.rectangle([tx, ty, tx + tw + 4, ty + 12], fill=color)
        draw.text((tx + 2, ty), text, fill="white", font=font)
    return annotated


def count_metrics(counts: dict, total: int) -> None:
    cols = st.columns(4)
    cols[0].metric("Total cells", total)
    cols[1].metric("Red blood cells", counts.get("RBC", 0))
    cols[2].metric("White blood cells", counts.get("WBC", 0))
    cols[3].metric("Platelets", counts.get("Platelets", 0))


def qc_panel(qc: dict) -> None:
    # Only known values reach the raw-HTML badge below.
    status = qc.get("status") if qc.get("status") in QC_COLORS else "N/A"
    st.markdown(
        f"**QC status:** <span style='background:{QC_COLORS.get(status, '#6B7280')};"
        f"color:white;padding:2px 10px;border-radius:12px'>{status}</span>",
        unsafe_allow_html=True,
    )
    avg = qc.get("average_confidence")
    cols = st.columns(4)
    cols[0].metric("Mean confidence", f"{avg:.1%}" if avg is not None else "N/A")
    cols[1].metric("Low-confidence boxes", qc.get("low_confidence_count", "N/A"))
    cols[2].metric("Overlapping pairs", qc.get("overlapping_pair_count", "N/A"))
    cols[3].metric("Border-truncated", qc.get("edge_truncated_count", "N/A"))
    for flag in qc.get("flags", []):
        st.warning(flag.replace("_", " ").title(), icon="⚠️")


def image_viewer(image: Image.Image, detections: list[dict], key: str) -> None:
    c1, c2 = st.columns([3, 1])
    with c2:
        threshold = st.slider(
            "Min confidence", 0.0, 1.0, 0.0, 0.05, key=f"thr-{key}",
            help="Extra filter on top of the model's calibrated per-class thresholds",
        )
        classes = st.multiselect("Classes", list(COLORS), default=list(COLORS), key=f"cls-{key}")
    filtered = [
        d for d in detections
        if d.get("confidence", 0) >= threshold and d.get("class") in classes
    ]
    with c2:
        st.caption(f"Showing **{len(filtered)}** of {len(detections)} detections")
        for name, color in COLORS.items():
            n = sum(d.get("class") == name for d in filtered)
            st.markdown(f"<span style='color:{color}'>■</span> {name}: **{n}**", unsafe_allow_html=True)
    with c1:
        a, b = st.columns(2)
        a.image(image, caption="Original", width="stretch")
        b.image(draw_detections(image, filtered), caption="Detections", width="stretch")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🔬 Blood Cell VisionOps")
st.caption("YOLO11 blood-smear cell detection · automated QC · Gemini technical reports")
st.info("Research and education use only. Not intended for clinical diagnosis.", icon="ℹ️")

health = api_health()
with st.sidebar:
    st.header("System")
    if health:
        st.success(f"API online · {health.get('model')} on {health.get('device')}")
        st.caption(
            f"DB: {'✅' if health.get('database') else '❌'} · "
            f"Gemini: {'✅' if health.get('llm_reports') else 'template fallback'}"
        )
    else:
        st.error(f"API offline ({API_URL})")
    if st.button("Refresh data", width="stretch"):
        load_analyses.clear()
        api_health.clear()
        st.rerun()

tab_analyze, tab_review, tab_analytics, tab_model = st.tabs(
    ["🧪 Analyze", "🗂️ Review", "📊 Analytics", "🏆 Model performance"]
)

# ---- Analyze ------------------------------------------------------------- #
with tab_analyze:
    st.subheader("Analyze a new microscopy image")
    upload = st.file_uploader("Blood smear image", type=["jpg", "jpeg", "png"])
    c1, c2 = st.columns(2)
    persist = c1.toggle("Save to database", value=True)
    use_llm = c2.toggle("Gemini report", value=True)
    if upload and st.button("Run analysis", type="primary", disabled=health is None):
        with st.spinner("Detecting cells and writing report..."):
            try:
                resp = requests.post(
                    f"{API_URL}/analyze",
                    params={"persist": persist, "llm": use_llm},
                    files={"file": (upload.name, upload.getvalue(), upload.type)},
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state["last_analysis"] = resp.json()
                st.session_state["last_image"] = upload.getvalue()
                load_analyses.clear()
            except requests.RequestException as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                st.error(f"Analysis failed (HTTP {code})." if code else "Analysis failed: the API could not be reached.")
    result = st.session_state.get("last_analysis")
    if result:
        st.divider()
        st.caption(
            f"`{result['filename']}` · {result['model_name']} · "
            f"{result['inference_ms']:.0f} ms inference · report: {result['report_source']}"
            + (f" · saved as #{result['id']}" if result.get("id") else "")
        )
        count_metrics(result["counts"], result["total_detections"])
        qc_panel(result["quality_control"])
        image_viewer(Image.open(io.BytesIO(st.session_state["last_image"])), result["detections"], "live")
        st.subheader("Technical report")
        val = result.get("report_validation") or {}
        if val.get("status") == "verified":
            st.success(
                f"Gemini narrative passed the claim check: {val.get('numbers_checked', 0)} numeric values are "
                "traceable to the detector output and cell counts were cross-checked. Interpretations are not verified.",
                icon="✅",
            )
        elif val.get("status") == "rejected":
            st.warning("Gemini narrative rejected by verification; showing the deterministic report.", icon="🛡️")
            with st.expander("Why it was rejected"):
                for issue in val.get("issues", []):
                    st.write(f"- {issue}")
        st.markdown(result["ai_report"])

# ---- Shared DB load ------------------------------------------------------ #
db_error = None
try:
    df = load_analyses()
except Exception as exc:
    df, db_error = pd.DataFrame(), exc

# ---- Review -------------------------------------------------------------- #
with tab_review:
    if db_error:
        # Driver errors can name hosts, users or databases; keep them out of the page.
        st.error("Could not load analyses from PostgreSQL. Check that the database is running and DATABASE_URL is set.")
    elif df.empty:
        st.warning("No analyses stored yet. Run `make ingest` or use the Analyze tab.")
    else:
        f1, f2 = st.columns([2, 3])
        statuses = f1.multiselect("QC status", ["PASS", "REVIEW", "FAIL"], default=["PASS", "REVIEW", "FAIL"])
        view = df[df["qc_status"].isin(statuses)].reset_index(drop=True)
        if view.empty:
            st.info("No analyses match the filter.")
        else:
            idx = f2.selectbox(
                "Analysis",
                range(len(view)),
                format_func=lambda i: f"#{view.at[i, 'id']} · {view.at[i, 'filename']} · {view.at[i, 'qc_status']}",
            )
            rec = view.iloc[idx]
            qc = _as_obj(rec["quality_control"]) or {}
            dets = _as_obj(rec["detections"]) or []
            count_metrics(
                {"RBC": rec["rbc_count"], "WBC": rec["wbc_count"], "Platelets": rec["platelet_count"]},
                int(rec["total_detections"]),
            )
            qc_panel(qc)
            stored = load_image(rec["id"]) if rec.get("has_image") else None
            path = IMAGE_DIR / Path(rec["filename"]).name
            if stored:
                image_viewer(Image.open(io.BytesIO(stored)), dets, f"rev-{rec['id']}")
            elif path.is_file():
                image_viewer(Image.open(path), dets, f"rev-{rec['id']}")
            else:
                st.warning("Source image not stored and not found in the image folder.")
            st.subheader("Technical report")
            st.caption(f"Source: {rec.get('report_source') or 'n/a'} · model {rec['model_name']}")
            st.markdown(rec["ai_report"] or "No report available.")

# ---- Analytics ----------------------------------------------------------- #
with tab_analytics:
    if db_error or df.empty:
        st.info("Analytics appear once analyses are stored.")
    else:
        k = st.columns(4)
        k[0].metric("Images analysed", len(df))
        k[1].metric("QC pass rate", f"{(df['qc_status'] == 'PASS').mean():.0%}")
        k[2].metric("Median latency", f"{df['inference_ms'].median():.0f} ms")
        k[3].metric("Mean RBC / image", f"{df['rbc_count'].mean():.1f}")

        long = df.melt(
            id_vars=["id"], value_vars=["rbc_count", "wbc_count", "platelet_count"],
            var_name="cell", value_name="count",
        )
        long["cell"] = long["cell"].map({"rbc_count": "RBC", "wbc_count": "WBC", "platelet_count": "Platelets"})
        c1, c2 = st.columns(2)
        c1.altair_chart(
            alt.Chart(long).mark_boxplot(extent="min-max").encode(
                x=alt.X("cell:N", title=None),
                y=alt.Y("count:Q", title="Cells per image"),
                color=alt.Color("cell:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None),
            ).properties(title="Cell count distribution", height=320),
            width="stretch",
        )
        qc_counts = df["qc_status"].value_counts().rename_axis("status").reset_index(name="n")
        c2.altair_chart(
            alt.Chart(qc_counts).mark_arc(innerRadius=60).encode(
                theta="n:Q",
                color=alt.Color("status:N", scale=alt.Scale(domain=list(QC_COLORS), range=list(QC_COLORS.values()))),
                tooltip=["status", "n"],
            ).properties(title="QC outcomes", height=320),
            width="stretch",
        )
        flags = pd.Series(
            [f for q in df["quality_control"].map(_as_obj) for f in (q or {}).get("flags", [])],
            dtype="object",
        )
        if not flags.empty:
            st.altair_chart(
                alt.Chart(flags.value_counts().rename_axis("flag").reset_index(name="n")).mark_bar().encode(
                    x=alt.X("n:Q", title="Images"), y=alt.Y("flag:N", sort="-x", title=None),
                ).properties(title="Most common QC flags", height=220),
                width="stretch",
            )
        st.subheader("History")
        st.dataframe(
            df[["id", "filename", "model_name", "total_detections", "rbc_count", "wbc_count",
                "platelet_count", "qc_status", "inference_ms", "report_source", "created_at"]],
            hide_index=True, width="stretch",
        )
        st.download_button(
            "Download CSV", df.drop(columns=["detections", "has_image"], errors="ignore").to_csv(index=False), "analyses.csv"
        )

# ---- Model performance --------------------------------------------------- #
with tab_model:
    # Only per-model result files (they carry a "run" key); skip benchmark.json, bootstrap.json, etc.
    files = [
        p for p in (sorted(METRICS_DIR.glob("*.json")) if METRICS_DIR.exists() else [])
        if isinstance(json.loads(p.read_text()), dict) and "run" in json.loads(p.read_text())
    ]
    if not files:
        st.info("Run `make evaluate` to populate held-out test metrics.")
    else:
        rows = []
        for p in files:
            m = json.loads(p.read_text())
            rows.append({
                "model": m["run"], "params (M)": m["params_m"],
                **{k: v for k, v in m["overall"].items()},
                **{f"{k} count MAE": v for k, v in m["counting"]["count_mae"].items()},
            })
        st.subheader("Held-out test set")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        bench = METRICS_DIR / "benchmark.json"
        if bench.exists():
            st.subheader("Inference latency")
            st.dataframe(pd.DataFrame(json.loads(bench.read_text())), hide_index=True, width="stretch")
        run = st.selectbox("Diagnostic plots for", [json.loads(p.read_text())["run"] for p in files])
        plots = [("confusion_matrix_normalized.png", "Normalized confusion matrix"), ("BoxPR_curve.png", "Precision-recall curve")]
        cols = st.columns(len(plots))
        for col, (fname, title) in zip(cols, plots):
            path = EVAL_DIR / run / fname
            if path.exists():
                col.image(str(path), caption=title, width="stretch")
