"""
crumb_analysis_pipeline.py
==========================
Advanced crumb analysis pipeline for Cakamoka TPA Analyzer.

Provides a fully self-contained, importable pipeline that wraps the core
texture-profile-analysis logic into a clean API.  The Streamlit front-end
(app.py) can delegate heavy lifting to these helpers, and the pipeline can
also be driven from a script or notebook without any UI dependency.

Pipeline stages
---------------
1. load_csv          – parse a CSV of replicate measurements
2. build_summaries   – compute per-sample means, SDs, and replicate counts
3. standardize       – z-score standardise the means matrix
4. similarity        – Euclidean similarity scores vs. the control
5. statistics        – Welch two-sample t-tests (replicates or summary stats)
6. interpret         – rule-based texture interpretation + fix recommendations
7. image_features    – lightweight image-based crumb feature extraction
8. run_pipeline      – convenience wrapper that executes all stages in order
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

TPA_PARAMS: List[str] = [
    "Hardness",
    "Resilience",
    "Cohesiveness",
    "Springiness",
    "Chewiness",
]
ALL_PARAMS: List[str] = TPA_PARAMS + ["Max Shear Force (N)"]

#: CSV column name → canonical parameter name
CSV_COL_MAP: Dict[str, str] = {
    "Hardness":            "Hardness",
    "Resilience":          "Resilience",
    "Cohesiveness":        "Cohesiveness",
    "Springiness":         "Springiness",
    "Chewiness":           "Chewiness",
    "Max Shear Force (N)": "MaxShear",
    "MaxShear":            "MaxShear",
}

#: Reverse map: canonical → friendly
CANONICAL_MAP: Dict[str, str] = {
    "MaxShear": "Max Shear Force (N)",
}

MAJOR_THRESH: float = 40.0   # % diff → "major"
MINOR_THRESH: float = 10.0   # % diff below which deviations are suppressed
SIG_ALPHA: float = 0.05

TEXTURE_LOGIC: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Hardness",           "low"):  ("Soft structure",         "Low hardness indicates a weak, soft crumb that compresses easily under load."),
    ("Hardness",           "high"): ("Firm / dense structure", "High hardness means greater resistance to compression — the cake is dense or over-structured."),
    ("Chewiness",          "low"):  ("Low chewing resistance", "Low chewiness often co-occurs with low hardness and cohesiveness, suggesting a fragile, crumbly texture."),
    ("Chewiness",          "high"): ("Chewy / tough texture",  "High chewiness indicates the cake requires significant masticatory work — may be perceived as tough."),
    ("Cohesiveness",       "low"):  ("Fragile crumb",          "Low cohesiveness means the internal structure bonds weakly — the crumb falls apart rather than deforming."),
    ("Cohesiveness",       "high"): ("Cohesive crumb",         "High cohesiveness signals a well-bonded crumb structure with good integrity."),
    ("Springiness",        "low"):  ("Low elasticity",         "Low springiness means the sample does not recover well after compression — less bouncy, less fresh-like."),
    ("Springiness",        "high"): ("High elasticity",        "High springiness suggests good crumb recovery — typical of a well-leavened, airy structure."),
    ("Resilience",         "low"):  ("Poor recovery",          "Low resilience indicates the crumb does not bounce back during the first bite — dense or stale feel."),
    ("Resilience",         "high"): ("Good recovery",          "High resilience means the crumb springs back quickly — airy and fresh texture."),
    ("Max Shear Force (N)","low"):  ("Crumbly / fragile",      "Low shear force signals weak structural integrity — the cake cuts or breaks with minimal force."),
    ("Max Shear Force (N)","high"): ("Tough / resistant",      "High shear force means greater resistance to cutting — the sample may be over-structured or dense."),
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleSummary:
    """Per-sample descriptive statistics."""
    name:    str
    means:   Dict[str, float] = field(default_factory=dict)
    sds:     Dict[str, float] = field(default_factory=dict)
    ns:      Dict[str, int]   = field(default_factory=dict)
    reps:    Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Euclidean similarity result for one sample vs. the control."""
    sample:   str
    score:    int           # 0–100
    distance: float
    label:    str


@dataclass
class StatTestResult:
    """Welch t-test result for one parameter of one sample."""
    param:   str
    p_value: Optional[float]
    method:  str
    n_ctrl:  int
    n_samp:  int

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < SIG_ALPHA


@dataclass
class InterpretationResult:
    """Texture interpretation for one sample."""
    sample:          str
    summary:         str
    reasoning_lines: List[str]
    fixes:           List[str]


@dataclass
class ImageFeatures:
    """Features extracted from a crumb photograph."""
    brightness:    float
    contrast:      float
    edge_density:  float
    crumb_color:   str
    bake_level:    str
    pore_structure: str
    aeration:      str
    cell_structure: str


@dataclass
class PipelineResult:
    """Aggregated output of the full analysis pipeline."""
    samples:         List[str]
    control:         str
    means_df:        pd.DataFrame
    sd_df:           pd.DataFrame
    z_df:            pd.DataFrame
    pca_coords:      Optional[np.ndarray]
    pca_variance:    Optional[np.ndarray]
    similarities:    List[SimilarityResult]
    stat_tests:      Dict[str, List[StatTestResult]]   # sample → [StatTestResult]
    interpretations: List[InterpretationResult]


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path_or_buffer) -> Dict[str, SampleSummary]:
    """
    Parse a CSV file of replicate measurements into a dict of
    :class:`SampleSummary` objects keyed by sample name.

    Required CSV columns: ``Sample``, ``Hardness``, ``Resilience``,
    ``Cohesiveness``, ``Springiness``, ``Chewiness``.
    Optional column: ``MaxShear`` or ``Max Shear Force (N)``.

    Parameters
    ----------
    path_or_buffer:
        File path (str / Path) or any file-like object accepted by
        :func:`pandas.read_csv`.

    Returns
    -------
    dict
        ``{sample_name: SampleSummary}``

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    df = pd.read_csv(path_or_buffer)

    required = {"Sample", "Hardness", "Resilience", "Cohesiveness", "Springiness", "Chewiness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    summaries: Dict[str, SampleSummary] = {}
    for sname, grp in df.groupby("Sample", sort=False):
        sname = str(sname).strip()
        ss = SampleSummary(name=sname)
        for col, param in CSV_COL_MAP.items():
            friendly = CANONICAL_MAP.get(param, param)
            if col in grp.columns:
                vals = grp[col].dropna().tolist()
                ss.reps[friendly]  = [float(v) for v in vals]
                ss.ns[friendly]    = len(vals)
                ss.means[friendly] = float(np.mean(vals)) if vals else float("nan")
                ss.sds[friendly]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summaries[sname] = ss

    return summaries


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — BUILD SUMMARY DATAFRAMES
# ──────────────────────────────────────────────────────────────────────────────

def build_summaries(
    summaries: Dict[str, SampleSummary],
    sample_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a dict of :class:`SampleSummary` objects into two aligned
    DataFrames — one for means and one for standard deviations.

    Parameters
    ----------
    summaries:
        Output of :func:`load_csv` or manually constructed summaries.
    sample_order:
        Optional ordering of samples (rows).  Defaults to ``dict`` insertion
        order.

    Returns
    -------
    means_df, sd_df : pd.DataFrame
        Shape ``(n_samples, n_params)``.
    """
    order = sample_order if sample_order is not None else list(summaries.keys())
    means_rows: Dict[str, Dict[str, float]] = {}
    sd_rows:    Dict[str, Dict[str, float]] = {}

    for samp in order:
        if samp not in summaries:
            continue
        ss = summaries[samp]
        m_row: Dict[str, float] = {}
        s_row: Dict[str, float] = {}
        for param in ALL_PARAMS:
            if param in ss.means:
                m_row[param] = ss.means[param]
                s_row[param] = ss.sds.get(param, 0.0)
            else:
                m_row[param] = float("nan")
                s_row[param] = 0.0
        means_rows[samp] = m_row
        sd_rows[samp]    = s_row

    means_df = pd.DataFrame(means_rows).T
    sd_df    = pd.DataFrame(sd_rows).T
    return means_df, sd_df


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Z-SCORE STANDARDISATION
# ──────────────────────────────────────────────────────────────────────────────

def standardize(means_df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise z-score standardisation of the means matrix.

    This removes unit/scale differences between parameters (e.g. Hardness in N
    vs. dimensionless Springiness ratio) so that all parameters contribute
    equally to Euclidean distance and PCA.  The transformation is purely
    descriptive — it is applied to means only and the SD matrix is unaffected.

    Parameters
    ----------
    means_df : pd.DataFrame
        Shape ``(n_samples, n_params)``.

    Returns
    -------
    pd.DataFrame
        Standardised matrix with the same shape and index/columns as the input.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(means_df.fillna(means_df.mean()))
    return pd.DataFrame(scaled, index=means_df.index, columns=means_df.columns)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SIMILARITY SCORES
# ──────────────────────────────────────────────────────────────────────────────

def compute_similarity(
    z_df: pd.DataFrame,
    control: str,
) -> List[SimilarityResult]:
    """
    Euclidean similarity scores (0–100) for every sample vs. the control in
    z-score space.

    The theoretical maximum distance is anchored at ``sqrt(n_params) * 3``
    (i.e. 3 standard deviations per parameter simultaneously).  Using an
    **absolute** anchor means a sample's score does not change when new samples
    are added to the run.

    Parameters
    ----------
    z_df : pd.DataFrame
        Standardised means matrix (output of :func:`standardize`).
    control : str
        Name of the control sample (must be a row in *z_df*).

    Returns
    -------
    list of SimilarityResult
        One entry per sample (including the control itself).
    """
    n_params = z_df.shape[1]
    absolute_max = np.sqrt(n_params) * 3.0
    ctrl_vec = z_df.loc[control].values

    results: List[SimilarityResult] = []
    for samp in z_df.index:
        dist  = euclidean(z_df.loc[samp].values, ctrl_vec)
        score = max(0, round(100 * (1 - dist / absolute_max)))
        if score >= 75:
            label = "Very close"
        elif score >= 50:
            label = "Moderate difference"
        elif score >= 25:
            label = "Large difference"
        else:
            label = "Very different"
        results.append(SimilarityResult(sample=samp, score=score, distance=round(dist, 4), label=label))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — STATISTICAL TESTS
# ──────────────────────────────────────────────────────────────────────────────

def _welch_from_summary(
    mean1: float, sd1: float, n1: int,
    mean2: float, sd2: float, n2: int,
) -> Optional[float]:
    """Analytical Welch two-sample t-test from summary statistics."""
    if sd1 == 0 and sd2 == 0:
        return None
    se1 = (sd1 ** 2) / n1 if n1 > 0 else 0.0
    se2 = (sd2 ** 2) / n2 if n2 > 0 else 0.0
    se_diff = np.sqrt(se1 + se2)
    if se_diff == 0:
        return None
    t_stat = (mean1 - mean2) / se_diff
    num   = (se1 + se2) ** 2
    denom = (se1 ** 2 / max(n1 - 1, 1)) + (se2 ** 2 / max(n2 - 1, 1))
    df    = max(num / denom if denom > 0 else 1.0, 1.0)
    return float(2 * stats.t.sf(abs(t_stat), df=df))


def run_statistics(
    summaries: Dict[str, SampleSummary],
    control: str,
    default_n: int = 9,
) -> Dict[str, List[StatTestResult]]:
    """
    Welch two-sample t-tests for each non-control sample vs. the control,
    across all TPA and shear parameters.

    Priority
    --------
    1. If both groups have ≥ 2 actual replicate values, :func:`scipy.stats.ttest_ind`
       is used directly — this is the most accurate path.
    2. If only summary statistics are available, the analytical Welch formula
       is used from means, SDs, and n counts.
    3. Returns ``p_value=None`` when data are insufficient.

    Parameters
    ----------
    summaries : dict
        Output of :func:`load_csv` or manually constructed summaries.
    control : str
        Name of the control sample.
    default_n : int
        Fallback replicate count used when ``n`` is not recorded on a summary.

    Returns
    -------
    dict
        ``{sample_name: [StatTestResult, ...]}`` — one result per parameter.
        The control sample is excluded.
    """
    ctrl = summaries.get(control)
    if ctrl is None:
        raise ValueError(f"Control sample '{control}' not found in summaries.")

    output: Dict[str, List[StatTestResult]] = {}

    for samp_name, ss in summaries.items():
        if samp_name == control:
            continue
        results: List[StatTestResult] = []
        for param in ALL_PARAMS:
            ctrl_reps = ctrl.reps.get(param, [])
            samp_reps = ss.reps.get(param, [])

            if len(ctrl_reps) >= 2 and len(samp_reps) >= 2:
                _, p = stats.ttest_ind(ctrl_reps, samp_reps, equal_var=False)
                method = "Welch t-test (replicates)"
                n_c, n_s = len(ctrl_reps), len(samp_reps)
            else:
                c_m = ctrl.means.get(param)
                c_s = ctrl.sds.get(param, 0.0)
                c_n = ctrl.ns.get(param, default_n)
                s_m = ss.means.get(param)
                s_s = ss.sds.get(param, 0.0)
                s_n = ss.ns.get(param, default_n)

                if (
                    c_m is not None and s_m is not None
                    and not np.isnan(c_m) and not np.isnan(s_m)
                    and c_n >= 2 and s_n >= 2
                ):
                    p = _welch_from_summary(c_m, c_s, c_n, s_m, s_s, s_n)
                    method = "Welch t-test (summary stats)"
                    n_c, n_s = c_n, s_n
                else:
                    p, method, n_c, n_s = None, "insufficient data", 0, 0

            results.append(StatTestResult(
                param=param,
                p_value=float(p) if p is not None else None,
                method=method,
                n_ctrl=n_c,
                n_samp=n_s,
            ))
        output[samp_name] = results

    return output


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6 — INTERPRETATION
# ──────────────────────────────────────────────────────────────────────────────

def interpret(
    sample_name: str,
    means:       pd.Series,
    ctrl_means:  pd.Series,
    minor_thresh: float = MINOR_THRESH,
    major_thresh: float = MAJOR_THRESH,
) -> InterpretationResult:
    """
    Rule-based texture interpretation for a single sample vs. control.

    Parameters
    ----------
    sample_name : str
        Label used in the returned :class:`InterpretationResult`.
    means : pd.Series
        Mean values for the sample (index = parameter names).
    ctrl_means : pd.Series
        Mean values for the control.
    minor_thresh : float
        Minimum absolute % difference to include in interpretation.
    major_thresh : float
        Absolute % difference above which a deviation is flagged as "major".

    Returns
    -------
    InterpretationResult
    """
    diffs: Dict[str, float] = {}
    for param in ALL_PARAMS:
        ctrl_val = ctrl_means.get(param)
        samp_val = means.get(param)
        if ctrl_val and ctrl_val != 0 and samp_val is not None and not np.isnan(samp_val):
            pct = (samp_val - ctrl_val) / abs(ctrl_val) * 100.0
            if abs(pct) >= minor_thresh:
                diffs[param] = pct

    descriptors: List[str] = []
    reasoning:   List[str] = []

    for param, pct in diffs.items():
        direction = "low" if pct < 0 else "high"
        key = (param, direction)
        if key in TEXTURE_LOGIC:
            lbl, reason = TEXTURE_LOGIC[key]
            descriptors.append(lbl)
            reasoning.append(
                f"{param} is {abs(pct):.0f}% {'lower' if pct < 0 else 'higher'} than control: {reason}"
            )

    if not descriptors:
        return InterpretationResult(
            sample=sample_name,
            summary="Very close to control.",
            reasoning_lines=["No major texture deviations detected."],
            fixes=[],
        )

    seen: set = set()
    unique_desc = [d for d in descriptors if not (d in seen or seen.add(d))]  # type: ignore[func-returns-value]
    summary = " | ".join(unique_desc[:3]) + "."

    sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    fixes: List[str] = []
    for param, pct in sorted_diffs[:3]:
        direction = "Increase" if pct < 0 else "Decrease"
        magnitude = "significantly" if abs(pct) >= major_thresh else "slightly"
        fixes.append(f"{direction} {param.lower()} {magnitude}")

    return InterpretationResult(
        sample=sample_name,
        summary=summary,
        reasoning_lines=reasoning,
        fixes=fixes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 7 — CRUMB IMAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_crumb_image(img: Image.Image) -> ImageFeatures:
    """
    Lightweight rule-based feature extraction from a crumb photograph.

    Uses pixel-level brightness, contrast (standard deviation), and a gradient-
    based edge density proxy to characterise:

    * Crumb colour / bake level
    * Pore structure uniformity
    * Cell-wall definition

    No external vision model is required.

    Parameters
    ----------
    img : PIL.Image.Image
        The crumb image (any colour mode; converted to greyscale internally).

    Returns
    -------
    ImageFeatures
    """
    gray = img.convert("L")
    arr  = np.array(gray, dtype=float)

    brightness   = arr.mean()
    contrast     = arr.std()
    gx           = np.abs(np.diff(arr, axis=1)).mean()
    gy           = np.abs(np.diff(arr, axis=0)).mean()
    edge_density = (gx + gy) / 2.0

    if brightness < 90:
        crumb_color = "Dark crumb — could indicate caramelisation, cocoa, or over-baking."
        bake_level  = "Potential over-bake or high-sugar formulation."
    elif brightness > 190:
        crumb_color = "Very light crumb — may suggest under-baking or a pale, low-sugar batter."
        bake_level  = "Check bake time and temperature."
    else:
        crumb_color = "Crumb color appears normal — mid-range browning."
        bake_level  = "Bake level appears standard."

    if contrast > 55:
        pore_structure = "High contrast indicates visible, non-uniform pores — open, irregular crumb structure."
        aeration       = "Structure appears porous / airy."
    elif contrast < 25:
        pore_structure = "Low contrast indicates a dense, tight crumb with minimal visible pores."
        aeration       = "Structure appears dense / compact."
    else:
        pore_structure = "Moderate contrast — medium crumb porosity."
        aeration       = "Structure appears intermediate."

    if edge_density > 15:
        cell_structure = "High edge density — many crumb cell walls visible, suggesting a fine, tight cell structure."
    elif edge_density < 6:
        cell_structure = "Low edge density — large or ill-defined crumb cells, consistent with a coarse or open structure."
    else:
        cell_structure = "Moderate edge density — balanced crumb cell definition."

    return ImageFeatures(
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        edge_density=round(edge_density, 2),
        crumb_color=crumb_color,
        bake_level=bake_level,
        pore_structure=pore_structure,
        aeration=aeration,
        cell_structure=cell_structure,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 8 — PCA PROJECTION
# ──────────────────────────────────────────────────────────────────────────────

def compute_pca(
    z_df: pd.DataFrame,
    n_components: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project the standardised means matrix onto its first principal components.

    Parameters
    ----------
    z_df : pd.DataFrame
        Standardised means matrix (output of :func:`standardize`).
    n_components : int
        Number of PCs to retain (capped at ``min(n_samples, n_params)``).

    Returns
    -------
    coords : np.ndarray or None
        Shape ``(n_samples, n_components)``.  ``None`` if fewer than 2 samples.
    explained_variance : np.ndarray or None
        Fraction of variance explained by each retained PC.
    """
    if z_df.shape[0] < 2:
        return None, None

    n = min(n_components, z_df.shape[0], z_df.shape[1])
    pca    = PCA(n_components=n)
    coords = pca.fit_transform(z_df.values)
    return coords, pca.explained_variance_ratio_


# ──────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    path_or_buffer,
    control: str,
    sample_order: Optional[List[str]] = None,
    minor_thresh: float = MINOR_THRESH,
    major_thresh: float = MAJOR_THRESH,
    default_n: int = 9,
) -> PipelineResult:
    """
    Execute the complete crumb analysis pipeline from a CSV file.

    Parameters
    ----------
    path_or_buffer :
        Path or file-like accepted by :func:`load_csv`.
    control : str
        Name of the control sample as it appears in the CSV ``Sample`` column.
    sample_order : list of str, optional
        Desired row order in output DataFrames.  Defaults to CSV order.
    minor_thresh : float
        Minimum % difference shown in interpretations (default 10 %).
    major_thresh : float
        % difference threshold for "major" flag (default 40 %).
    default_n : int
        Fallback replicate count for the analytical t-test (default 9).

    Returns
    -------
    PipelineResult
        All intermediate and final outputs bundled in one object.

    Example
    -------
    >>> result = run_pipeline("sample_data.csv", control="Control")
    >>> for sim in result.similarities:
    ...     print(sim.sample, sim.score, sim.label)
    """
    summaries = load_csv(path_or_buffer)

    if control not in summaries:
        raise ValueError(
            f"Control sample '{control}' not found in CSV.  "
            f"Available samples: {list(summaries.keys())}"
        )

    order = sample_order or list(summaries.keys())

    means_df, sd_df = build_summaries(summaries, sample_order=order)
    z_df            = standardize(means_df)
    similarities    = compute_similarity(z_df, control)
    stat_tests      = run_statistics(summaries, control, default_n=default_n)
    pca_coords, pca_var = compute_pca(z_df)

    interpretations: List[InterpretationResult] = []
    for samp in order:
        if samp == control or samp not in means_df.index:
            continue
        interp = interpret(
            sample_name=samp,
            means=means_df.loc[samp],
            ctrl_means=means_df.loc[control],
            minor_thresh=minor_thresh,
            major_thresh=major_thresh,
        )
        interpretations.append(interp)

    return PipelineResult(
        samples=order,
        control=control,
        means_df=means_df,
        sd_df=sd_df,
        z_df=z_df,
        pca_coords=pca_coords,
        pca_variance=pca_var,
        similarities=similarities,
        stat_tests=stat_tests,
        interpretations=interpretations,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="crumb_analysis_pipeline",
        description="Run the Cakamoka crumb analysis pipeline from the command line.",
    )
    parser.add_argument("csv",     help="Path to the replicate measurements CSV file.")
    parser.add_argument("control", help="Name of the control sample (must match CSV exactly).")
    parser.add_argument("--minor-thresh", type=float, default=MINOR_THRESH,
                        help=f"Min %% difference to display (default {MINOR_THRESH}).")
    parser.add_argument("--major-thresh", type=float, default=MAJOR_THRESH,
                        help=f"%%  threshold for 'major' flag (default {MAJOR_THRESH}).")
    parser.add_argument("--default-n", type=int, default=9,
                        help="Fallback replicate count for analytical t-test (default 9).")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.csv,
            control=args.control,
            minor_thresh=args.minor_thresh,
            major_thresh=args.major_thresh,
            default_n=args.default_n,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Cakamoka TPA Pipeline  |  control: {result.control}")
    print(f"{'='*60}\n")

    print("Similarity Scores")
    print("-" * 40)
    for sim in sorted(result.similarities, key=lambda s: s.score, reverse=True):
        marker = " (control)" if sim.sample == result.control else ""
        print(f"  {sim.sample:<20} {sim.score:>3}/100  {sim.label}{marker}")

    print()
    for interp in result.interpretations:
        print(f"[{interp.sample}]  {interp.summary}")
        for line in interp.reasoning_lines:
            print(f"    • {line}")
        if interp.fixes:
            print("  Recommended fixes:")
            for fix in interp.fixes:
                print(f"    → {fix}")
        print()


if __name__ == "__main__":
    _cli()
