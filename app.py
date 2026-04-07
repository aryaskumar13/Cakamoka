import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cakamoka — TPA Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- Typography & base ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Main background ---- */
    .stApp { background-color: #f5f5f7; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* ---- Cards ---- */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }

    /* ---- Section headers ---- */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6e6e73;
        margin-bottom: 6px;
    }

    /* ---- Big rank label ---- */
    .rank-badge {
        display: inline-block;
        background: #1d1d1f;
        color: #ffffff;
        border-radius: 100px;
        padding: 4px 16px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-bottom: 12px;
    }

    .score-big {
        font-size: 40px;
        font-weight: 300;
        color: #1d1d1f;
        line-height: 1.1;
    }

    .score-label {
        font-size: 13px;
        color: #6e6e73;
        margin-top: 2px;
    }

    /* ---- Tag pills ---- */
    .tag-major {
        display: inline-block;
        background: #ffeaea;
        color: #c0392b;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-minor {
        display: inline-block;
        background: #fff4e0;
        color: #b76e00;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-sig {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }
    .tag-good {
        display: inline-block;
        background: #e6f9f0;
        color: #1a7f4b;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }

    /* ---- Dividers ---- */
    .thin-divider {
        border: none;
        border-top: 1px solid #e8e8ed;
        margin: 18px 0;
    }

    /* ---- Interpretation block ---- */
    .interp-block {
        background: #f5f5f7;
        border-left: 3px solid #1d1d1f;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 14px;
        color: #1d1d1f;
        line-height: 1.7;
    }

    /* ---- Fix item ---- */
    .fix-item {
        background: #f5f5f7;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 14px;
        color: #1d1d1f;
        font-weight: 500;
    }

    /* ---- Sub-text ---- */
    .sub-text {
        font-size: 13px;
        color: #6e6e73;
        line-height: 1.6;
    }

    /* ---- Page title ---- */
    .page-title {
        font-size: 34px;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.5px;
    }
    .page-subtitle {
        font-size: 17px;
        color: #6e6e73;
        font-weight: 400;
        margin-top: 4px;
    }

    /* ---- Streamlit tweaks ---- */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #1d1d1f !important;
    }

    .stButton > button {
        background: #1a7f4b;
        color: #ffffff;
        border: none;
        border-radius: 980px;
        padding: 10px 28px;
        font-size: 14px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background: #166d3f;
        color: #ffffff;
    }

    div[data-testid="stTabs"] button {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6e6e73 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #1d1d1f !important;
        border-bottom: 2px solid #1d1d1f !important;
    }

    /* ---- Slider value labels ---- */
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"],
    div[data-testid="stSlider"] p,
    div[data-testid="stSlider"] span {
        color: #1d1d1f !important;
    }

    /* Current selected value thumb label */
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #1d1d1f !important;
        background: #e8e8ed !important;
        border-radius: 4px !important;
        padding: 1px 6px !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }

    /* Slider track */
    div[data-testid="stSlider"] > div > div > div {
        background: #e8e8ed !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: #1d1d1f !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TPA_PARAMS   = ["Hardness", "Resilience", "Cohesiveness", "Springiness", "Chewiness"]
ALL_PARAMS   = TPA_PARAMS + ["Max Shear Force (N)"]
MAJOR_THRESH = 40   # % difference threshold for major
MINOR_THRESH = 10   # % difference below which not shown
SIG_ALPHA    = 0.05

# Apple-inspired discrete color palette for samples
PALETTE = [
    "#0071e3",  # blue (control-ish)
    "#ff6b35",  # orange
    "#a2845e",  # warm brown
    "#ffd60a",  # yellow
    "#30d158",  # green
    "#bf5af2",  # purple
    "#ff375f",  # red
    "#64d2ff",  # cyan
    "#ff9f0a",  # amber
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def z_score_standardize(means_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score standardize column-wise across samples.
    Purpose: remove unit/scale differences between parameters (e.g. Hardness in N vs
    Springiness dimensionless ratio) so that all parameters contribute equally to
    Euclidean distance and PCA. This is purely descriptive scaling — it is NOT used
    for inference. The SD input does not affect this step.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(means_df)
    return pd.DataFrame(scaled, index=means_df.index, columns=means_df.columns)


def compute_euclidean_similarity(z_df: pd.DataFrame, control: str, n_params: int) -> tuple:
    """
    Euclidean distance from control in z-score space, converted to a 0-100 similarity score.

    Anchoring strategy: the theoretical maximum distance in an n-dimensional z-score space
    after standardisation is bounded. We use sqrt(n_params) * 3 as a generous anchor
    (3 standard deviations per dimension) so the scale is ABSOLUTE and does not shift
    when new samples are added. This prevents a sample's score from changing just because
    a different, worse sample was included in the run.
    """
    ctrl_vec = z_df.loc[control].values
    distances = {}
    for sample in z_df.index:
        distances[sample] = euclidean(z_df.loc[sample].values, ctrl_vec)

    # Absolute anchor: a sample 3 SD away on every single parameter simultaneously
    absolute_max = np.sqrt(n_params) * 3.0
    scores = {s: max(0, round(100 * (1 - d / absolute_max))) for s, d in distances.items()}
    return pd.Series(scores), pd.Series(distances)


def welch_t_from_summary(mean1, sd1, n1, mean2, sd2, n2):
    """
    Welch's two-sample t-test computed analytically from summary statistics
    (mean, SD, n) — no synthetic data points are generated.

    This is statistically valid: the Welch t-statistic and Satterthwaite degrees
    of freedom are derived directly from the sufficient statistics of a normal distribution.
    """
    if sd1 == 0 and sd2 == 0:
        return None  # No variance at all — test is undefined
    se1 = (sd1 ** 2) / n1 if n1 > 0 else 0
    se2 = (sd2 ** 2) / n2 if n2 > 0 else 0
    se_diff = np.sqrt(se1 + se2)
    if se_diff == 0:
        return None
    t_stat = (mean1 - mean2) / se_diff
    # Satterthwaite degrees of freedom
    if se1 == 0 and se2 == 0:
        return None
    num   = (se1 + se2) ** 2
    denom = (se1 ** 2 / max(n1 - 1, 1)) + (se2 ** 2 / max(n2 - 1, 1))
    df    = num / denom if denom > 0 else 1.0
    df    = max(df, 1.0)
    p_val = 2 * stats.t.sf(abs(t_stat), df=df)
    return float(p_val)


def run_statistical_tests(raw_data: dict, summary_data: dict, control: str, samples: list, n_replicates: int) -> dict:
    """
    Two-sample Welch t-test for each sample vs control, per parameter.

    Priority:
      1. If actual replicate lists are provided (len >= 2), use scipy ttest_ind directly
         — this is the most accurate path and should always be preferred.
      2. If only summary stats (mean, SD, n) are available, use the analytical
         Welch formula — this is statistically valid but carries a power warning
         when n is small (< 3).
      3. Never fabricate data points from mean ± SD (removed).

    Returns dict: {sample: {param: {"p": float|None, "method": str, "n_ctrl": int, "n_samp": int}}}
    """
    results = {}
    ctrl_raw = raw_data.get(control, {})
    ctrl_sum = summary_data.get(control, {})

    for samp in samples:
        if samp == control:
            continue
        samp_results = {}
        samp_raw = raw_data.get(samp, {})
        samp_sum = summary_data.get(samp, {})

        for param in ALL_PARAMS:
            ctrl_reps = ctrl_raw.get(param, [])
            samp_reps = samp_raw.get(param, [])

            if len(ctrl_reps) >= 2 and len(samp_reps) >= 2:
                # Path 1: real replicates — most accurate
                _, p = stats.ttest_ind(ctrl_reps, samp_reps, equal_var=False)
                method = "Welch t-test (replicates)"
                n_c, n_s = len(ctrl_reps), len(samp_reps)
            else:
                # Path 2: analytical Welch from summary stats
                c_m = ctrl_sum.get(param, {}).get("mean", None)
                c_s = ctrl_sum.get(param, {}).get("sd",   None)
                c_n = ctrl_sum.get(param, {}).get("n",    n_replicates)
                s_m = samp_sum.get(param, {}).get("mean", None)
                s_s = samp_sum.get(param, {}).get("sd",   None)
                s_n = samp_sum.get(param, {}).get("n",    n_replicates)

                if all(v is not None for v in [c_m, c_s, s_m, s_s]) and c_n >= 2 and s_n >= 2:
                    p = welch_t_from_summary(c_m, c_s, c_n, s_m, s_s, s_n)
                    method = "Welch t-test (summary stats)"
                    n_c, n_s = c_n, s_n
                else:
                    p, method, n_c, n_s = None, "insufficient data", 0, 0

            samp_results[param] = {"p": p, "method": method, "n_ctrl": n_c, "n_samp": n_s}
        results[samp] = samp_results
    return results


def pct_diff(val, ctrl_val):
    if ctrl_val == 0:
        return None
    return (val - ctrl_val) / abs(ctrl_val) * 100


# ─────────────────────────────────────────────
# INTERPRETATION ENGINE
# ─────────────────────────────────────────────

TEXTURE_LOGIC = {
    # (param, direction): (short label, reasoning)
    ("Hardness", "low"):       ("Soft structure",          "Low hardness indicates a weak, soft crumb that compresses easily under load."),
    ("Hardness", "high"):      ("Firm / dense structure",  "High hardness means greater resistance to compression — the cake is dense or over-structured."),
    ("Chewiness", "low"):      ("Low chewing resistance",  "Low chewiness often co-occurs with low hardness and cohesiveness, suggesting a fragile, crumbly texture."),
    ("Chewiness", "high"):     ("Chewy / tough texture",   "High chewiness indicates the cake requires significant masticatory work — may be perceived as tough."),
    ("Cohesiveness", "low"):   ("Fragile crumb",           "Low cohesiveness means the internal structure bonds weakly — the crumb falls apart rather than deforming."),
    ("Cohesiveness", "high"):  ("Cohesive crumb",          "High cohesiveness signals a well-bonded crumb structure with good integrity."),
    ("Springiness", "low"):    ("Low elasticity",          "Low springiness means the sample does not recover well after compression — less bouncy, less fresh-like."),
    ("Springiness", "high"):   ("High elasticity",         "High springiness suggests good crumb recovery — typical of a well-leavened, airy structure."),
    ("Resilience", "low"):     ("Poor recovery",           "Low resilience indicates the crumb does not bounce back during the first bite — dense or stale feel."),
    ("Resilience", "high"):    ("Good recovery",           "High resilience means the crumb springs back quickly — airy and fresh texture."),
    ("Max Shear Force (N)", "low"):  ("Crumbly / fragile",       "Low shear force signals weak structural integrity — the cake cuts or breaks with minimal force, suggesting a crumbly or underbaked product."),
    ("Max Shear Force (N)", "high"): ("Tough / resistant",       "High shear force means greater resistance to cutting — the sample may be over-structured, dense, or gummy."),
}

def interpret_sample(diffs_dict: dict, means: pd.Series, ctrl_means: pd.Series) -> tuple[str, str, list]:
    """
    Returns (summary_sentence, detailed_reasoning, list_of_fixes)
    diffs_dict: {param: pct_diff_value}  — only significant ones passed
    """
    descriptors = []
    reasoning_lines = []

    for param, pct in diffs_dict.items():
        direction = "low" if pct < 0 else "high"
        key = (param, direction)
        if key in TEXTURE_LOGIC:
            lbl, reason = TEXTURE_LOGIC[key]
            descriptors.append(lbl)
            reasoning_lines.append(f"{param} is {abs(pct):.0f}% {'lower' if pct < 0 else 'higher'} than control: {reason}")

    if not descriptors:
        return "Very close to control.", "No major texture deviations detected.", []

    # Deduplicate descriptors
    seen = set()
    unique_descriptors = []
    for d in descriptors:
        if d not in seen:
            unique_descriptors.append(d)
            seen.add(d)

    summary = " | ".join(unique_descriptors[:3]) + "."

    # Top 2-3 fixes
    # Sort by magnitude of difference
    sorted_diffs = sorted(diffs_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    fixes = []
    for param, pct in sorted_diffs[:3]:
        direction = "Increase" if pct < 0 else "Decrease"
        magnitude = "significantly" if abs(pct) >= MAJOR_THRESH else "slightly"
        fixes.append(f"{direction} {param.lower()} {magnitude}")

    return summary, reasoning_lines, fixes


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_bar_chart(means_df: pd.DataFrame, sd_df: pd.DataFrame, sample_colors: dict) -> go.Figure:
    """Grouped bar chart — one group per parameter, bars colored by sample."""
    params = means_df.columns.tolist()
    samples = means_df.index.tolist()

    fig = go.Figure()
    for i, samp in enumerate(samples):
        fig.add_trace(go.Bar(
            name=samp,
            x=params,
            y=means_df.loc[samp].values,
            error_y=dict(type="data", array=sd_df.loc[samp].values, visible=True, thickness=1.5, width=4),
            marker_color=sample_colors.get(samp, PALETTE[i % len(PALETTE)]),
            marker_line_width=0,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="TPA & Kramer Shear — All Samples", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", color="#1d1d1f", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#1d1d1f"),
            linecolor="#e0e0e0",
        ),
        yaxis=dict(
            gridcolor="#f0f0f0",
            tickfont=dict(size=12, color="#6e6e73"),
            zeroline=False,
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        bargap=0.18,
        bargroupgap=0.06,
        height=420,
    )
    return fig


def plot_pca(z_df: pd.DataFrame, sample_colors: dict, control: str) -> go.Figure:
    """PCA scatter — samples projected onto PC1 & PC2."""
    if z_df.shape[0] < 2:
        return None

    pca = PCA(n_components=min(2, z_df.shape[1]))
    coords = pca.fit_transform(z_df.values)
    var_exp = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    for i, samp in enumerate(z_df.index):
        is_ctrl = samp == control
        fig.add_trace(go.Scatter(
            x=[coords[i, 0]],
            y=[coords[i, 1] if coords.shape[1] > 1 else 0],
            mode="markers+text",
            name=samp,
            text=[samp],
            textposition="top center",
            textfont=dict(size=12, family="Inter"),
            marker=dict(
                size=14 if is_ctrl else 10,
                color=sample_colors.get(samp, PALETTE[i % len(PALETTE)]),
                symbol="diamond" if is_ctrl else "circle",
                line=dict(width=2 if is_ctrl else 0, color="#1d1d1f"),
            ),
            showlegend=False,
        ))

    xlab = f"PC1 ({var_exp[0]:.1f}% variance)" if len(var_exp) > 0 else "PC1"
    ylab = f"PC2 ({var_exp[1]:.1f}% variance)" if len(var_exp) > 1 else "PC2"

    fig.update_layout(
        title=dict(text="PCA — Texture Space", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
        xaxis=dict(title=xlab, showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#e0e0e0", tickfont=dict(size=11)),
        yaxis=dict(title=ylab, showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#e0e0e0", tickfont=dict(size=11)),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", color="#1d1d1f"),
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a hex color string to an rgba() string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_radar(means_df: pd.DataFrame, z_df: pd.DataFrame, sample_colors: dict, control: str) -> go.Figure:
    """Radar / spider chart in z-score space."""
    params = z_df.columns.tolist()
    fig = go.Figure()
    for i, samp in enumerate(z_df.index):
        vals = z_df.loc[samp].tolist()
        vals_closed = vals + [vals[0]]
        params_closed = params + [params[0]]
        color = sample_colors.get(samp, PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=params_closed,
            fill="toself",
            fillcolor=hex_to_rgba(color, alpha=0.18 if samp == control else 0.12),
            line=dict(color=color, width=2 if samp == control else 1.5),
            name=samp,
            opacity=0.9 if samp == control else 0.75,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, gridcolor="#e8e8ed", tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12, family="Inter")),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        title=dict(text="Texture Profile — Radar (Z-Score)", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", color="#1d1d1f"),
        height=450,
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


# ─────────────────────────────────────────────
# IMAGE ANALYSIS (vision-free, rule-based)
# ─────────────────────────────────────────────

def analyze_crumb_image(img: Image.Image) -> dict:
    """
    Lightweight rule-based crumb image analysis using brightness, contrast, and edge density.
    Returns a dict with observations and inferences.
    """
    import numpy as np

    img_gray = img.convert("L")
    arr = np.array(img_gray, dtype=float)

    brightness  = arr.mean()          # 0-255
    contrast    = arr.std()           # spread
    # Simple edge proxy: variance of Laplacian-like gradient
    from numpy import gradient
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    edge_density = (gx + gy) / 2.0

    # Classify
    if brightness < 90:
        crust_note = "Dark crumb — could indicate caramelisation, cocoa, or over-baking."
        bake_note  = "Potential over-bake or high-sugar formulation."
    elif brightness > 190:
        crust_note = "Very light crumb — may suggest under-baking or a pale, low-sugar batter."
        bake_note  = "Check bake time and temperature."
    else:
        crust_note = "Crumb color appears normal — mid-range browning."
        bake_note  = "Bake level appears standard."

    if contrast > 55:
        pore_note  = "High contrast indicates visible, non-uniform pores — open, irregular crumb structure."
        airy_note  = "Structure appears porous / airy."
    elif contrast < 25:
        pore_note  = "Low contrast indicates a dense, tight crumb with minimal visible pores."
        airy_note  = "Structure appears dense / compact."
    else:
        pore_note  = "Moderate contrast — medium crumb porosity."
        airy_note  = "Structure appears intermediate."

    if edge_density > 15:
        texture_note = "High edge density — many crumb cell walls visible, suggesting a fine, tight cell structure."
    elif edge_density < 6:
        texture_note = "Low edge density — large or ill-defined crumb cells, consistent with a coarse or open structure."
    else:
        texture_note = "Moderate edge density — balanced crumb cell definition."

    return {
        "Brightness":   round(brightness, 1),
        "Contrast":     round(contrast, 1),
        "Edge Density": round(edge_density, 2),
        "Crumb Color":  crust_note,
        "Bake Level":   bake_note,
        "Pore Structure": pore_note,
        "Aeration":     airy_note,
        "Cell Structure": texture_note,
    }


# ─────────────────────────────────────────────
# SIDEBAR — SAMPLE SETUP
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div style='padding:24px 0 8px 0;'><span style='font-size:20px;font-weight:600;color:#1d1d1f;'>Cakamoka</span><br><span style='font-size:12px;color:#6e6e73;'>TPA & Texture Analyzer</span></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e8e8ed;margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>Sample Configuration</p>", unsafe_allow_html=True)
    n_samples = st.number_input("Number of samples (including control)", min_value=2, max_value=9, value=3, step=1)

    sample_names = []
    for i in range(int(n_samples)):
        default = "Control" if i == 0 else f"Sample {chr(65+i-1)}"
        name = st.text_input(f"Sample {i+1} name", value=default, key=f"sname_{i}")
        sample_names.append(name.strip() if name.strip() else default)

    control_sample = st.selectbox("Which is the control?", options=sample_names)

    st.markdown("<hr style='border:none;border-top:1px solid #e8e8ed;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Analysis Settings</p>", unsafe_allow_html=True)
    major_thresh = st.slider("Major difference threshold (%)", min_value=20, max_value=60, value=40, step=5,
                             help="Percentage difference above which a parameter is flagged as 'major'.")
    show_minor_thresh = st.slider("Minimum % diff to display", min_value=5, max_value=25, value=10, step=5,
                                  help="Differences smaller than this are omitted from the report.")

    st.markdown("<hr style='border:none;border-top:1px solid #e8e8ed;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Replicates</p>", unsafe_allow_html=True)
    n_replicates = st.number_input(
        "Default replicates (fallback)",
        min_value=2, max_value=20, value=9, step=1,
        help="Used only for the analytical t-test fallback when summary stats are entered without an explicit n. When uploading a CSV, n is counted automatically per sample."
    )
    st.markdown(
        "<p class='sub-text' style='margin-top:6px;'>Replicate counts can differ between samples — each sample's n is determined from its data. Max Shear Force can have a different n from TPA parameters within the same sample.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:none;border-top:1px solid #e8e8ed;margin:16px 0;'>", unsafe_allow_html=True)
    if st.button("🔄 Reboot App", use_container_width=True, help="Clear all results and start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style='padding: 36px 0 24px 0;'>
    <div class='page-title'>TPA Texture Analyzer</div>
    <div class='page-subtitle'>Compare cake samples against a control using texture profile analysis and Kramer shear data.</div>
</div>
""", unsafe_allow_html=True)

tab_input, tab_results, tab_viz, tab_image = st.tabs(["Data Input", "Results & Insights", "Visualizations", "Crumb Image Analysis"])

# ─────────────────────────────────────────────
# TAB 1 — DATA INPUT
# ─────────────────────────────────────────────
with tab_input:

    # ─── CSV Upload ───
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Import Data from CSV</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-text'>"
        "Upload a CSV where each row is one replicate measurement. Each sample may have a different number "
        "of replicates — the count is determined automatically per sample. Max Shear Force may have fewer "
        "replicates than the TPA parameters within the same sample; simply leave those rows blank in that "
        "column and they will be excluded. "
        "<br><br>"
        "Required columns: <b>Sample</b>, <b>Hardness</b>, <b>Resilience</b>, <b>Cohesiveness</b>, "
        "<b>Springiness</b>, <b>Chewiness</b>, <b>MaxShear</b>. "
        "The control sample must be named exactly as configured in the sidebar."
        "</p>",
        unsafe_allow_html=True,
    )

    # Download template button
    template_cols = ["Sample", "Hardness", "Resilience", "Cohesiveness", "Springiness", "Chewiness", "MaxShear"]
    template_rows = []
    for sname in sample_names:
        template_rows.append({c: (sname if c == "Sample" else "") for c in template_cols})
    template_df  = pd.DataFrame(template_rows)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="Download blank template",
        data=template_csv,
        file_name="tpa_template.csv",
        mime="text/csv",
    )

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    input_data = {}
    csv_loaded = False

    if uploaded_csv:
        try:
            csv_df = pd.read_csv(uploaded_csv)

            # Validate required columns
            required = {"Sample", "Hardness", "Resilience", "Cohesiveness", "Springiness", "Chewiness"}
            missing  = required - set(csv_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            else:
                # Show preview
                st.markdown("<p class='section-title' style='margin-top:16px;'>Preview</p>", unsafe_allow_html=True)
                st.dataframe(csv_df, use_container_width=True)

                col_map = {
                    "Hardness":            "Hardness",
                    "Resilience":          "Resilience",
                    "Cohesiveness":        "Cohesiveness",
                    "Springiness":         "Springiness",
                    "Chewiness":           "Chewiness",
                    "Max Shear Force (N)": "MaxShear",
                }

                csv_input      = {}
                sample_n_table = {}  # for display

                for sname, grp in csv_df.groupby("Sample", sort=False):
                    sname = str(sname).strip()
                    pdata = {}
                    n_info = {}
                    for param, col in col_map.items():
                        if col in grp.columns:
                            vals = grp[col].dropna().tolist()
                            pdata[param] = {"reps": vals}
                            n_info[param] = len(vals)
                        else:
                            # MaxShear column absent — mark as missing, not error
                            pdata[param] = {"reps": []}
                            n_info[param] = 0
                    csv_input[sname]      = pdata
                    sample_n_table[sname] = n_info

                input_data = csv_input
                csv_loaded = True

                # Show per-sample n table
                st.markdown("<p class='section-title' style='margin-top:16px;'>Replicate counts detected</p>", unsafe_allow_html=True)
                n_display = pd.DataFrame(sample_n_table).T
                n_display.index.name = "Sample"
                st.dataframe(n_display, use_container_width=True)

                detected_samples = list(csv_input.keys())
                st.markdown(
                    f"<p style='font-size:12px;color:#1a7f4b;margin-top:8px;font-weight:500;'>"
                    f"{len(csv_input)} samples loaded: {', '.join(detected_samples)}. "
                    f"Click Run Analysis to proceed.</p>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    else:
        st.markdown(
            "<div style='text-align:center;padding:40px 0;color:#aeaeb2;font-size:13px;'>"
            "No file uploaded yet. Drop your CSV above or download the blank template to get started."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_analysis = st.button("Run Analysis", disabled=not csv_loaded)


# ─────────────────────────────────────────────
# ANALYSIS ENGINE (runs on button click or if session state present)
# ─────────────────────────────────────────────

def build_means_sd_df(input_data, sample_names):
    """
    Build means and SD dataframes.
    - For replicate mode: mean and SD are computed from the actual data (ddof=1).
    - For summary mode: values are taken as entered; n is also stored but not used here.
    Mean and SD are NEVER manually typed when replicates are available.
    """
    means_rows, sd_rows = {}, {}
    for samp in sample_names:
        if samp not in input_data:
            continue
        m_row, s_row = {}, {}
        for param in ALL_PARAMS:
            pd_ = input_data[samp].get(param, {})
            if "reps" in pd_:
                reps = pd_["reps"]
                m_row[param] = float(np.mean(reps))
                s_row[param] = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
            else:
                m_row[param] = float(pd_.get("mean", 1.0))
                s_row[param] = float(pd_.get("sd",   0.0))
        means_rows[samp] = m_row
        sd_rows[samp]    = s_row
    return pd.DataFrame(means_rows).T, pd.DataFrame(sd_rows).T


def build_raw_and_summary(input_data, sample_names, n_replicates_default):
    """
    Separate raw replicate lists (for direct t-tests) from summary dicts (for analytical t-tests).
    Never fabricates data points from mean ± SD.
    Returns:
        raw_data:     {sample: {param: [rep1, rep2, ...]}}   — only populated when replicates are available
        summary_data: {sample: {param: {mean, sd, n}}}       — always populated (computed from reps if available)
    """
    raw_data     = {}
    summary_data = {}

    for samp in sample_names:
        if samp not in input_data:
            continue
        raw_params  = {}
        sum_params  = {}
        for param in ALL_PARAMS:
            pd_ = input_data[samp].get(param, {})
            if "reps" in pd_:
                reps = pd_["reps"]
                raw_params[param] = reps
                m = float(np.mean(reps))
                s = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
                sum_params[param] = {"mean": m, "sd": s, "n": len(reps)}
            else:
                # Summary mode — no raw data available
                m = float(pd_.get("mean", 1.0))
                s = float(pd_.get("sd",   0.0))
                n = int(pd_.get("n",      n_replicates_default))
                raw_params[param] = []  # empty — signals "no raw replicates"
                sum_params[param] = {"mean": m, "sd": s, "n": n}
        raw_data[samp]     = raw_params
        summary_data[samp] = sum_params

    return raw_data, summary_data


if run_analysis or "analysis_done" in st.session_state:
    if run_analysis:
        st.session_state["analysis_done"] = True
        st.session_state["input_data"]    = input_data
        st.session_state["sample_names"]  = sample_names
        st.session_state["control"]       = control_sample
        st.session_state["major_thresh"]  = major_thresh
        st.session_state["minor_thresh"]  = show_minor_thresh

    # Retrieve from session
    _input   = st.session_state.get("input_data", input_data)
    _samples = st.session_state.get("sample_names", sample_names)
    _control = st.session_state.get("control", control_sample)
    _major   = st.session_state.get("major_thresh", major_thresh)
    _minor   = st.session_state.get("minor_thresh", show_minor_thresh)

    means_df, sd_df = build_means_sd_df(_input, _samples)
    raw_data, summary_data = build_raw_and_summary(_input, _samples, int(n_replicates))

    if _control not in means_df.index:
        st.error(f"Control sample '{_control}' not found in data. Please check sample names.")
        st.stop()

    z_df = z_score_standardize(means_df)
    scores_series, distances_series = compute_euclidean_similarity(z_df, _control, len(ALL_PARAMS))
    stat_results = run_statistical_tests(raw_data, summary_data, _control, _samples, int(n_replicates))

    # Assign colors — control always gets first color
    sample_colors = {}
    ci = 0
    for s in _samples:
        sample_colors[s] = PALETTE[ci % len(PALETTE)]
        ci += 1

    # Sort non-control samples by score descending
    non_ctrl = [s for s in _samples if s != _control]
    ranked   = sorted(non_ctrl, key=lambda s: scores_series.get(s, 0), reverse=True)

    # ─────────────────────────────────────────────
    # TAB 2 — RESULTS & INSIGHTS
    # ─────────────────────────────────────────────
    with tab_results:

        # ── SUMMARY TABLE ──
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Similarity Overview</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Euclidean distance in standardized texture space. A score of 100 is identical to control.</p>", unsafe_allow_html=True)

        summary_rows = []
        for rank_i, samp in enumerate(ranked, 1):
            sc = scores_series.get(samp, 0)
            dist = round(distances_series.get(samp, 0), 3)
            if sc >= 75:
                label = "Very close"
            elif sc >= 50:
                label = "Moderate difference"
            elif sc >= 25:
                label = "Large difference"
            else:
                label = "Very different"
            summary_rows.append({"Rank": f"#{rank_i}", "Sample": samp, "Similarity Score": f"{sc} / 100", "Distance": dist, "Assessment": label})

        ctrl_row = {"Rank": "—", "Sample": f"{_control} (Control)", "Similarity Score": "100 / 100", "Distance": 0.0, "Assessment": "Reference"}
        summary_df = pd.DataFrame([ctrl_row] + summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── PER-SAMPLE CARDS ──
        for rank_i, samp in enumerate(ranked, 1):
            sc   = scores_series.get(samp, 0)
            dist = distances_series.get(samp, 0)

            # % diffs
            all_pct = {}
            for param in ALL_PARAMS:
                ctrl_val = means_df.loc[_control, param]
                samp_val = means_df.loc[samp, param]
                p = pct_diff(samp_val, ctrl_val)
                if p is not None:
                    all_pct[param] = p

            # Filtered diffs (above minor threshold)
            shown_pct = {k: v for k, v in all_pct.items() if abs(v) >= _minor}

            # Significant params
            sig_params = []
            low_power_warning = False
            if samp in stat_results:
                for param, test in stat_results[samp].items():
                    pval   = test.get("p")
                    method = test.get("method", "")
                    n_c    = test.get("n_ctrl", 0)
                    n_s    = test.get("n_samp", 0)
                    if pval is not None and pval < SIG_ALPHA:
                        sig_params.append((param, pval, method, n_c, n_s))
                    if n_c < 3 or n_s < 3:
                        low_power_warning = True

            # Interpretation
            interp_pct = {k: v for k, v in all_pct.items() if abs(v) >= _minor}
            summary_text, reasoning_lines, fixes = interpret_sample(interp_pct, means_df.loc[samp], means_df.loc[_control])

            # Score color
            if sc >= 75:
                score_color = "#30d158"
            elif sc >= 50:
                score_color = "#ff9f0a"
            else:
                score_color = "#ff375f"

            st.markdown(f"<div class='card'>", unsafe_allow_html=True)

            # Header row
            h_col1, h_col2 = st.columns([3, 1])
            with h_col1:
                st.markdown(f"<div class='rank-badge'>#{rank_i} of {len(ranked)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:22px;font-weight:600;color:#1d1d1f;'>{samp}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub-text' style='margin-top:4px;'>{summary_text}</div>", unsafe_allow_html=True)
            with h_col2:
                st.markdown(f"<div style='text-align:right;'><div class='score-big' style='color:{score_color};'>{sc}</div><div class='score-label'>Similarity / 100</div></div>", unsafe_allow_html=True)

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

            r_col1, r_col2, r_col3 = st.columns(3)

            # Column 1: Key differences
            with r_col1:
                st.markdown("<p class='section-title'>Key Differences vs Control</p>", unsafe_allow_html=True)
                if shown_pct:
                    for param, pct in sorted(shown_pct.items(), key=lambda x: abs(x[1]), reverse=True):
                        arrow  = "down" if pct < 0 else "up"
                        symbol = "-" if pct < 0 else "+"
                        tag_class = "tag-major" if abs(pct) >= _major else "tag-minor"
                        tag_label = "major" if abs(pct) >= _major else "minor"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin-bottom:8px;'>"
                            f"<span style='font-size:14px;color:#1d1d1f;font-weight:500;flex:1;'>{param}</span>"
                            f"<span style='font-size:13px;color:{'#c0392b' if pct < 0 else '#1a7f4b'};font-weight:600;margin-right:8px;'>{symbol}{abs(pct):.0f}%</span>"
                            f"<span class='{tag_class}'>{tag_label}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("<p class='sub-text'>No differences above the display threshold.</p>", unsafe_allow_html=True)

            # Column 2: Statistical significance
            with r_col2:
                st.markdown("<p class='section-title'>Statistical Tests</p>", unsafe_allow_html=True)
                if sig_params:
                    for param, pval, method, n_c, n_s in sorted(sig_params, key=lambda x: x[1]):
                        # Format p-value: use scientific notation below 0.001 so it never displays as 0.0000
                        if pval < 0.001:
                            p_display = f"{pval:.2e}"
                        else:
                            p_display = f"{pval:.4f}"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin-bottom:8px;'>"
                            f"<span style='font-size:14px;color:#1d1d1f;font-weight:500;flex:1;'>{param}</span>"
                            f"<span class='tag-sig'>p = {p_display}</span>"
                            f"</div>"
                            f"<p style='font-size:11px;color:#aeaeb2;margin-top:-6px;margin-bottom:8px;'>"
                            f"{method} &nbsp;|&nbsp; ctrl n={n_c}, sample n={n_s}</p>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<p class='sub-text'>No parameters significant at p &lt; 0.05.</p>",
                        unsafe_allow_html=True,
                    )
                if low_power_warning:
                    st.markdown(
                        "<div style='background:#fff8e7;border-left:3px solid #ff9f0a;border-radius:0 6px 6px 0;"
                        "padding:8px 12px;margin-top:8px;font-size:12px;color:#b76e00;'>"
                        "Low power warning: n &lt; 3 in at least one group. "
                        "With few replicates, the test may miss real differences. "
                        "Interpret p-values with caution and increase replicates where possible."
                        "</div>",
                        unsafe_allow_html=True,
                    )

            # Column 3: Top fixes
            with r_col3:
                st.markdown("<p class='section-title'>Recommended Fixes</p>", unsafe_allow_html=True)
                if fixes:
                    for fix in fixes[:3]:
                        st.markdown(f"<div class='fix-item'>{fix}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='sub-text'>No significant changes recommended.</p>", unsafe_allow_html=True)

            # Interpretation block
            if reasoning_lines:
                st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
                st.markdown("<p class='section-title'>Interpretation</p>", unsafe_allow_html=True)
                content = "<br>".join([f"&bull; {line}" for line in reasoning_lines])
                st.markdown(f"<div class='interp-block'>{content}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── RAW DATA TABLE ──
        with st.expander("Raw Data — Means & Standard Deviations"):
            combined = pd.DataFrame()
            for samp in _samples:
                row_m = means_df.loc[samp].rename(lambda c: f"{c} (mean)")
                row_s = sd_df.loc[samp].rename(lambda c: f"{c} (SD)")
                row = pd.concat([row_m, row_s]).to_frame(name=samp).T
                combined = pd.concat([combined, row])
            st.dataframe(combined.round(4), use_container_width=True)

    # ─────────────────────────────────────────────
    # TAB 3 — VISUALIZATIONS
    # ─────────────────────────────────────────────
    with tab_viz:

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Grouped Bar Chart</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Mean values with standard deviation error bars. All parameters on a common axis for direct visual comparison.</p>", unsafe_allow_html=True)
        fig_bar = plot_bar_chart(means_df, sd_df, sample_colors)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-title'>PCA — Texture Space</p>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text'>Each point is one sample projected onto the two principal components of texture variation. Samples close together have similar profiles. The control is shown as a diamond.</p>", unsafe_allow_html=True)
            if len(_samples) >= 2:
                fig_pca = plot_pca(z_df, sample_colors, _control)
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("PCA requires at least 2 samples.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_v2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-title'>Radar Chart</p>", unsafe_allow_html=True)
            st.markdown("<p class='sub-text'>Z-score standardized values plotted as a spider chart. A sample with a profile similar to control will overlap closely with it.</p>", unsafe_allow_html=True)
            if len(_samples) >= 2:
                fig_radar = plot_radar(means_df, z_df, sample_colors, _control)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Radar chart requires at least 2 samples.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Percentage difference heatmap vs control
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Percentage Difference vs Control</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Each cell shows the percentage difference of a test sample from the control. Red = lower than control, blue = higher.</p>", unsafe_allow_html=True)

        pct_rows = {}
        for samp in non_ctrl:
            row = {}
            for param in ALL_PARAMS:
                ctrl_v = means_df.loc[_control, param]
                samp_v = means_df.loc[samp, param]
                row[param] = round(pct_diff(samp_v, ctrl_v) or 0, 1)
            pct_rows[samp] = row

        pct_hm_df = pd.DataFrame(pct_rows).T

        fig_heat = go.Figure(data=go.Heatmap(
            z=pct_hm_df.values,
            x=pct_hm_df.columns.tolist(),
            y=pct_hm_df.index.tolist(),
            colorscale=[[0, "#ff375f"], [0.5, "#ffffff"], [1, "#0071e3"]],
            zmid=0,
            text=[[f"{v:+.1f}%" for v in row] for row in pct_hm_df.values],
            texttemplate="%{text}",
            textfont=dict(size=12, family="Inter"),
            colorbar=dict(title="% Diff", tickfont=dict(size=11)),
        ))
        fig_heat.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", color="#1d1d1f"),
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(l=40, r=40, t=20, b=40),
            height=max(200, 60 * len(non_ctrl) + 80),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 4 — CRUMB IMAGE ANALYSIS
# ─────────────────────────────────────────────
with tab_image:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Crumb Image Analysis</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>Upload a photograph of a cake cross-section. The tool performs an automated visual analysis of crumb color, pore structure, and cell wall definition using image processing. For best results, use a well-lit, close-up shot of the cut face.</p>", unsafe_allow_html=True)

    uploaded_images = st.file_uploader(
        "Upload one or more cake cross-section images",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_images:
        for img_file in uploaded_images:
            img = Image.open(img_file)
            result = analyze_crumb_image(img)

            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='section-title'>{img_file.name}</p>", unsafe_allow_html=True)

            img_col, info_col = st.columns([1, 2])
            with img_col:
                st.image(img, use_container_width=True)
            with info_col:
                metric_items = [
                    ("Brightness", f"{result['Brightness']} / 255", result["Bake Level"]),
                    ("Contrast",   f"{result['Contrast']}",         result["Pore Structure"]),
                    ("Edge Density", f"{result['Edge Density']}",   result["Cell Structure"]),
                ]
                for label, value, note in metric_items:
                    st.markdown(
                        f"<div style='margin-bottom:14px;'>"
                        f"<span style='font-size:13px;font-weight:600;color:#1d1d1f;'>{label}:</span> "
                        f"<span style='font-size:13px;color:#3a3a3c;'>{value}</span><br>"
                        f"<span class='sub-text'>{note}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='interp-block'>"
                    f"<b>Summary:</b><br>{result['Aeration']} {result['Crumb Color']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;color:#aeaeb2;font-size:14px;'>
            No image uploaded yet. Drop a cake cross-section photo above.
        </div>
        """, unsafe_allow_html=True)
