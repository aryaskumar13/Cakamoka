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
from pathlib import Path
from PIL import Image
import tempfile
import warnings

from crumb_analysis_pipeline import AnalyzerConfig, CrumbAnalyzer

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

# ─────────────────────────────────────────────
# TEXTURE PARAMETER CONSTANTS
# ─────────────────────────────────────────────
ALL_PARAMS = ["Hardness", "Resilience", "Cohesiveness", "Springiness", "Chewiness", "MaxShear"]

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ─────────────────────────────────────────────
# STATISTICAL CONSTANTS
# ─────────────────────────────────────────────
SIG_ALPHA = 0.05  # Significance level for statistical tests

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def pct_diff(val1, val2):
    """Calculate percentage difference: (val1 - val2) / val2 * 100"""
    if val2 == 0 or val1 is None or val2 is None:
        return None
    return round((val1 - val2) / val2 * 100, 1)

def interpret_sample(pct_diffs, sample_means, control_means, stat_tests=None):
    """Generate interpretation text and recommendations based on % differences.
    
    stat_tests: dict with test results per parameter, e.g. {"MaxShear": {"p": 0.02, "significant": True, ...}}
    """
    # First, check if there are ANY statistically significant differences
    num_sig_diffs = 0
    if stat_tests:
        for param, test in stat_tests.items():
            if isinstance(test, dict) and test.get('significant', False):
                num_sig_diffs += 1
    
    # If NO statistically significant differences, sample is essentially very similar to control
    if num_sig_diffs == 0:
        summary = "Very similar to control (no statistically significant differences)"
        reasoning = ["All texture parameters are statistically indistinguishable from the control formulation."]
        fixes = ["Current formulation matches control performance — no adjustments needed."]
        return summary, reasoning, fixes
    
    # Otherwise, analyze based on percentage differences
    major_issues = []
    minor_issues = []
    good_aspects = []
    
    # Hardness
    h_pct = pct_diffs.get("Hardness")
    if h_pct is not None:
        if abs(h_pct) >= 40:
            if h_pct > 0:
                major_issues.append("Significantly harder than control")
            else:
                major_issues.append("Significantly softer than control")
        elif abs(h_pct) >= 20:
            if h_pct > 0:
                minor_issues.append("Harder than control")
            else:
                minor_issues.append("Softer than control")
        else:
            good_aspects.append("Hardness similar to control")
    
    # Resilience
    r_pct = pct_diffs.get("Resilience")
    if r_pct is not None:
        if abs(r_pct) >= 40:
            if r_pct > 0:
                major_issues.append("Much more elastic than control")
            else:
                major_issues.append("Much less elastic than control")
        elif abs(r_pct) >= 20:
            if r_pct > 0:
                minor_issues.append("More elastic than control")
            else:
                minor_issues.append("Less elastic than control")
        else:
            good_aspects.append("Elasticity similar to control")
    
    # Cohesiveness
    c_pct = pct_diffs.get("Cohesiveness")
    if c_pct is not None:
        if abs(c_pct) >= 40:
            if c_pct > 0:
                major_issues.append("Much more cohesive than control")
            else:
                major_issues.append("Much less cohesive than control")
        elif abs(c_pct) >= 20:
            if c_pct > 0:
                minor_issues.append("More cohesive than control")
            else:
                minor_issues.append("Less cohesive than control")
        else:
            good_aspects.append("Cohesiveness similar to control")
    
    # Springiness
    s_pct = pct_diffs.get("Springiness")
    if s_pct is not None:
        if abs(s_pct) >= 40:
            if s_pct > 0:
                major_issues.append("Much springier than control")
            else:
                major_issues.append("Much less springy than control")
        elif abs(s_pct) >= 20:
            if s_pct > 0:
                minor_issues.append("Springier than control")
            else:
                minor_issues.append("Less springy than control")
        else:
            good_aspects.append("Springiness similar to control")
    
    # Chewiness
    ch_pct = pct_diffs.get("Chewiness")
    if ch_pct is not None:
        if abs(ch_pct) >= 40:
            if ch_pct > 0:
                major_issues.append("Much chewier than control")
            else:
                major_issues.append("Much less chewy than control")
        elif abs(ch_pct) >= 20:
            if ch_pct > 0:
                minor_issues.append("Chewier than control")
            else:
                minor_issues.append("Less chewy than control")
        else:
            good_aspects.append("Chewiness similar to control")
    
    # Max Shear - Critical for cake structure
    ms_pct = pct_diffs.get("MaxShear")
    if ms_pct is not None:
        # Check if MaxShear difference is statistically significant
        ms_is_sig = False
        if stat_tests and "MaxShear" in stat_tests:
            ms_is_sig = stat_tests["MaxShear"].get("significant", False)
        
        if ms_pct < 0:  # Lower MaxShear
            # Only flag as CRUMBLY if statistically significant
            if ms_is_sig:
                if ms_pct <= -30:
                    major_issues.append("Much lower Max Shear than control - CRUMBLY, weak structure")
                elif ms_pct <= -15:
                    major_issues.append("Lower Max Shear than control - CRUMBLY structure")
                else:  # -15 to 0
                    minor_issues.append("Lower Max Shear than control - CRUMBLY tendencies")
            else:
                # Not statistically significant, so don't call it crumbly
                if ms_pct <= -15:
                    minor_issues.append("Lower Max Shear than control (not significant)")
                else:
                    good_aspects.append("Max Shear slightly lower but similar to control")
        elif abs(ms_pct) >= 30:
            major_issues.append("Much higher Max Shear than control")
        elif abs(ms_pct) >= 15:
            minor_issues.append("Higher Max Shear than control")
        elif abs(ms_pct) >= 5:
            minor_issues.append("Slightly higher Max Shear than control")
        else:
            good_aspects.append("Max Shear similar to control")
    
    # Generate summary text shown under sample name.
    # If statistically different, avoid qualitative "minor/major" labels.
    if num_sig_diffs > 0:
        summary = f"Statistically different from control ({num_sig_diffs} significant parameter{'s' if num_sig_diffs != 1 else ''})"
    elif major_issues:
        summary = f"Major differences: {', '.join(major_issues[:2])}"
        if len(major_issues) > 2:
            summary += f" (+{len(major_issues)-2} more)"
    elif minor_issues:
        summary = f"Minor differences: {', '.join(minor_issues[:2])}"
        if len(minor_issues) > 2:
            summary += f" (+{len(minor_issues)-2} more)"
    else:
        summary = "Very similar to control"
    
    # Reasoning
    reasoning = []
    if major_issues:
        reasoning.append("Major differences suggest significant formulation or processing changes.")
        # Check if Max Shear is a major issue
        if any("max shear" in issue.lower() for issue in major_issues):
            reasoning.append("Max Shear differences indicate potential issues with cake structure and crumb integrity.")
    if minor_issues:
        reasoning.append("Minor differences may be due to natural variation or slight adjustments.")
        # Check if Max Shear is a minor issue
        if any("max shear" in issue.lower() for issue in minor_issues):
            reasoning.append("Max Shear variations may affect cake texture and structural stability.")
    if good_aspects:
        reasoning.append("Similar aspects indicate consistency with control formulation.")
    
    # Fixes
    fixes = []
    if major_issues:
        fixes.append("Review formulation ingredients and ratios.")
        fixes.append("Check mixing and baking processes.")
        # Specific fixes for Max Shear/structure issues
        if any("max shear" in issue.lower() or "structure" in issue.lower() for issue in major_issues):
            fixes.append("Address structural issues: check flour quality, fat content, and baking temperature.")
            fixes.append("Consider adding stabilizers or adjusting moisture content for better crumb structure.")
    if minor_issues:
        fixes.append("Monitor process consistency.")
        fixes.append("Consider slight ingredient adjustments.")
        # Specific fixes for Max Shear issues
        if any("max shear" in issue.lower() or "structure" in issue.lower() for issue in minor_issues):
            fixes.append("Fine-tune formulation for improved structural integrity.")
    if not major_issues and not minor_issues:
        fixes.append("Current formulation is well-matched to control.")
    
    return summary, reasoning, fixes

def compute_parameter_correlations(means_df: pd.DataFrame, min_r_squared: float = 0.25, sig_alpha: float = 0.05) -> dict:
    """
    Compute correlations between texture parameters across samples.
    Returns only statistically significant correlations with high effect size.
    
    Args:
        means_df: DataFrame with samples as rows, parameters as columns
        min_r_squared: Minimum R² threshold to report (default 0.25 = 25% variance explained)
        sig_alpha: Significance level (default 0.05)
    
    Returns:
        dict with 'correlations' (list of significant pairs) and 'matrix' (full correlation matrix)
    """
    params = means_df.columns.tolist()
    correlations = []
    corr_matrix = pd.DataFrame(index=params, columns=params, dtype=float)
    
    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            if i >= j:  # Skip diagonal and duplicates
                if i == j:
                    corr_matrix.loc[param1, param2] = 1.0
                continue
            
            # Calculate Pearson correlation
            try:
                x = means_df[param1].values
                y = means_df[param2].values
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) < 3:  # Need at least 3 points
                    continue
                
                r, p_val = stats.pearsonr(x_clean, y_clean)
                r_squared = r ** 2
                
                # Store in matrix
                corr_matrix.loc[param1, param2] = r
                corr_matrix.loc[param2, param1] = r
                
                # Only report if significant AND meaningful effect size
                if p_val < sig_alpha and r_squared >= min_r_squared:
                    correlations.append({
                        'param1': param1,
                        'param2': param2,
                        'r': round(r, 3),
                        'r_squared': round(r_squared, 3),
                        'p_value': round(p_val, 4),
                        'direction': 'positive' if r > 0 else 'negative'
                    })
            except:
                pass
    
    # Fill diagonal
    for param in params:
        corr_matrix.loc[param, param] = 1.0
    
    # Convert to float
    corr_matrix = corr_matrix.astype(float)
    
    return {
        'correlations': sorted(correlations, key=lambda x: abs(x['r']), reverse=True),
        'matrix': corr_matrix
    }

def compute_structure_score(metrics: dict, config=None) -> tuple:
    """
    Compute Structure Score (0-100) and classify as Strong or Weak based on structural metrics.
    
    Args:
        metrics: dict with keys like 'pore_cv', 'porosity_uniformity', 'homogeneity', etc.
        config: optional AnalyzerConfig for thresholds
    
    Returns:
        tuple of (score: float, classification: str, reasons: list, interpretation: str)
    """
    from crumb_analysis_pipeline import AnalyzerConfig
    cfg = config or AnalyzerConfig()
    
    # ── GROUP 1: Uniformity (30%) ──
    pore_cv = float(metrics.get('pore_cv', 0.5))
    porosity_uniformity = float(metrics.get('porosity_uniformity', 0.1))
    homogeneity = float(metrics.get('homogeneity', 0.4))
    
    # Normalize: lower is better for CV and uniformity, higher is better for homogeneity
    uniformity_score = (1.0 - np.clip(pore_cv / 1.5, 0, 1)) * 0.33  # High CV = weak
    uniformity_score += np.clip(1.0 - porosity_uniformity / 0.15, 0, 1) * 0.33  # High std = weak
    uniformity_score += np.clip(homogeneity / 0.6, 0, 1) * 0.34  # Higher = good
    uniformity_score = np.clip(uniformity_score * 100, 0, 100)
    uniformity_weight = 0.30
    
    # ── GROUP 2: Wall Integrity (25%) ──
    mean_wall_thickness = float(metrics.get('mean_wall_thickness', 3.0))
    wall_thickness_var = float(metrics.get('wall_thickness_var', 5.0))
    
    # Normalize: higher thickness is better, lower variance is better
    wall_integrity_score = np.clip(mean_wall_thickness / 6.0, 0, 1) * 0.6 * 100  # Good thickness
    wall_integrity_score += np.clip(1.0 - wall_thickness_var / 10.0, 0, 1) * 0.4 * 100  # Consistent thickness
    wall_integrity_score = np.clip(wall_integrity_score, 0, 100)
    wall_weight = 0.25
    
    # ── GROUP 3: Network Stability (20%) ──
    connectivity_ratio = float(metrics.get('connectivity_ratio', 0.25))
    clustering_index = float(metrics.get('clustering_index', 1.0))
    
    # Normalize: higher connectivity is good, higher clustering (>1=clustered) is bad
    network_stability = np.clip(connectivity_ratio / 0.5, 0, 1) * 0.5 * 100  # Higher connectivity = good
    network_stability += np.clip(1.0 - clustering_index / 2.0, 0, 1) * 0.5 * 100  # Lower clustering = good  
    network_stability = np.clip(network_stability, 0, 100)
    network_weight = 0.20
    
    # ── GROUP 4: Cell Quality (15%) ──
    mean_pore_size = float(metrics.get('mean_pore_size', 100.0))
    circularity = float(metrics.get('circularity', 0.6))
    
    # Normalize: moderate pore size is good (not too large), higher circularity is good
    cell_quality = np.clip(1.0 - abs(mean_pore_size - 100) / 150, 0, 1) * 0.4 * 100  # ~100px is ideal
    cell_quality += np.clip(circularity / 0.8, 0, 1) * 0.6 * 100  # Higher circularity = good
    cell_quality = np.clip(cell_quality, 0, 100)
    cell_weight = 0.15
    
    # ── GROUP 5: Fracture Behavior (5%) ──
    fracture_index = float(metrics.get('fracture_index', 0.02))
    
    # Normalize: lower fracture index is better (fewer internal fractures)
    fracture_score = np.clip(1.0 - fracture_index / 0.05, 0, 1) * 100  # Low fractures = good
    fracture_weight = 0.05
    
    # ── GROUP 6: Porosity (5%) ──
    porosity = float(metrics.get('porosity', 0.2))
    
    # Normalize: moderate porosity (0.15-0.35) is ideal
    porosity_score = np.clip(1.0 - abs(porosity - 0.25) / 0.25, 0, 1) * 100  # Mid-range is good
    porosity_weight = 0.05
    
    # ── COMPUTE BASE SCORE ──
    base_score = (uniformity_score * uniformity_weight + 
                   wall_integrity_score * wall_weight +
                   network_stability * network_weight +
                   cell_quality * cell_weight +
                   fracture_score * fracture_weight +
                   porosity_score * porosity_weight)
    
    # ── CRITICAL INTERACTION RULES ──
    adjusted_score = base_score
    interaction_notes = []
    
    # Rule 1: High Pore CV AND (high clustering OR high connectivity) → WEAK
    if pore_cv > 0.8 and (clustering_index > cfg.clustering_weak or connectivity_ratio < 0.15):
        adjusted_score = min(adjusted_score, 50)  # Force weak territory
        interaction_notes.append(f"Irregular pore sizes + poor network stability = fracture risk")
    
    # Rule 2: High wall thickness variance → bias toward WEAK
    if wall_thickness_var > cfg.wall_variance_strong:
        adjusted_score -= 15
        interaction_notes.append(f"Inconsistent wall thickness reduces structural reliability")
    
    # Rule 3: High circularity alone does NOT indicate strong structure
    # (but combined with thick walls and low CV = strong)
    if circularity > 0.75 and homogeneity < 0.35 and mean_wall_thickness < 2.5:
        adjusted_score -= 10
        interaction_notes.append(f"Regular pores alone insufficient without strong walls")
    
    # Ensure score stays in 0-100 range
    adjusted_score = np.clip(adjusted_score, 0, 100)
    
    # ── CLASSIFY ──
    if adjusted_score >= 65:
        classification = "Strong"
        interpretation = "Structure is cohesive with stable pore network and strong walls. Cake will slice cleanly without excessive crumbling."
    else:
        classification = "Weak"
        interpretation = "Structure is prone to crumbling due to irregular pores or poor wall support. Cake may fracture or shed crumbs under stress."
    
    # ── MECHANISTIC REASONS ──
    reasons = []
    
    # Check dominant factors
    if uniformity_score < 50:
        reasons.append(f"Uneven pore distribution (CV={pore_cv:.2f}) creates weak points for crack propagation")
    if wall_integrity_score < 50:
        reasons.append(f"Thin or inconsistent walls (thickness variance={wall_thickness_var:.2f}) cannot support pore pressure")
    if network_stability < 50:
        reasons.append(f"Poor connectivity ratio ({connectivity_ratio:.2f}) means pores are isolated—fractures don't transfer stress")
    if cell_quality < 50 and circularity < 0.5:
        reasons.append(f"Irregular pore shapes (circularity={circularity:.2f}) create stress concentration points")
    if fracture_index > 0.03:
        reasons.append(f"High fracture index ({fracture_index:.4f}) indicates pre-existing micro-tears")
    
    if not reasons:
        if adjusted_score >= 75:
            reasons.append(f"Excellent uniformity and wall integrity support stable gas cell network")
        elif adjusted_score >= 65:
            reasons.append(f"Balanced pore-wall geometry maintains structural coherence despite minor irregularities")
        else:
            reasons.append(f"Cumulative effect of moderate defects reduces structural reliability")
    
    reasons.extend(interaction_notes)
    
    return adjusted_score, classification, reasons[:3], interpretation

def compute_maxshear_relationships(means_df: pd.DataFrame, min_points: int = 3) -> list:
    """Compute robust Spearman and Pearson relationships of MaxShear vs TPA params except Hardness."""
    results = []
    if "MaxShear" not in means_df.columns:
        return results

    comparison_params = [p for p in ALL_PARAMS if p not in ("Hardness", "MaxShear")]

    for param in comparison_params:
        if param not in means_df.columns:
            continue

        pair = means_df[["MaxShear", param]].apply(pd.to_numeric, errors="coerce").dropna()
        n_used = int(len(pair))
        entry = {
            "parameter": param,
            "n": n_used,
            "spearman_rho": None,
            "r_squared": None,
            "p_value": None,
            "direction": "not computed",
            "strength": "Not computed",
            "significant": None,
            "robust": False,
            "note": "",
        }

        if n_used < min_points:
            entry["note"] = f"Need at least {min_points} paired samples (have {n_used})."
            results.append(entry)
            continue

        if pair["MaxShear"].nunique() < 2 or pair[param].nunique() < 2:
            entry["note"] = "One variable is constant across samples; correlation undefined."
            results.append(entry)
            continue

        spearman_rho, spearman_p = stats.spearmanr(pair["MaxShear"], pair[param])
        pearson_r, _ = stats.pearsonr(pair["MaxShear"], pair[param])

        if np.isnan(spearman_rho) or np.isnan(spearman_p) or np.isnan(pearson_r):
            entry["note"] = "Correlation returned NaN (insufficient variability after filtering)."
            results.append(entry)
            continue

        spearman_rho = float(spearman_rho)
        spearman_p = float(spearman_p)
        r_squared = float(pearson_r ** 2)

        entry["spearman_rho"] = round(spearman_rho, 3)
        entry["r_squared"] = round(r_squared, 3)
        entry["p_value"] = round(spearman_p, 4)
        entry["direction"] = "positive" if spearman_rho >= 0 else "negative"
        abs_rho = abs(spearman_rho)
        if abs_rho >= 0.8:
            strength = "Very Strong"
        elif abs_rho >= 0.6:
            strength = "Strong"
        elif abs_rho >= 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
        entry["strength"] = strength
        entry["significant"] = bool(spearman_p < SIG_ALPHA)
        entry["robust"] = bool(n_used >= 5 and spearman_p < SIG_ALPHA and r_squared >= 0.25)
        entry["note"] = ""
        results.append(entry)

    return sorted(
        results,
        key=lambda x: abs(x["spearman_rho"]) if isinstance(x["spearman_rho"], float) else -1,
        reverse=True,
    )

def z_score_standardize(means_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the DataFrame using z-score (mean 0, std 1)."""
    scaler = StandardScaler()
    z_df = pd.DataFrame(
        scaler.fit_transform(means_df),
        index=means_df.index,
        columns=means_df.columns
    )
    return z_df

def compute_euclidean_similarity(z_df: pd.DataFrame, control: str, n_params: int) -> tuple:
    """Compute similarity scores and distances."""
    control_vec = z_df.loc[control]
    scores = {}
    distances = {}
    for sample in z_df.index:
        if sample == control:
            scores[sample] = 100.0
            distances[sample] = 0.0
        else:
            dist = euclidean(control_vec, z_df.loc[sample])
            # Convert distance to similarity score (0-100)
            # Max possible distance for n_params dimensions is roughly sqrt(n_params * 4) for z-scores
            max_dist = np.sqrt(n_params * 4)  # Conservative estimate
            score = max(0, 100 * (1 - dist / max_dist))
            scores[sample] = round(score, 1)
            distances[sample] = round(dist, 3)
    return pd.Series(scores), pd.Series(distances)

def run_statistical_tests(raw_data, summary_data, control, samples, n_replicates_default):
    """Run Welch two-sample t-tests for each parameter between control and each sample."""
    results = {}
    for samp in samples:
        if samp == control:
            continue
        samp_results = {}
        all_params_sig = True  # Track if all parameters are significantly different

        for param in ALL_PARAMS:
            # Prefer raw replicates when available
            if samp in raw_data and control in raw_data:
                ctrl_reps = raw_data[control].get(param, [])
                samp_reps = raw_data[samp].get(param, [])
                if len(ctrl_reps) >= 2 and len(samp_reps) >= 2:
                    t_stat, p_val = stats.ttest_ind(ctrl_reps, samp_reps, equal_var=False, nan_policy='omit')
                    is_valid = not (pd.isna(t_stat) or pd.isna(p_val))
                    is_sig = bool(is_valid and p_val < SIG_ALPHA)
                    samp_results[param] = {
                        'p': round(float(p_val), 4) if is_valid else None,
                        'f_stat': round(float(t_stat), 4) if is_valid else None,
                        'method': 'Welch t-test (replicates)',
                        'n_ctrl': len(ctrl_reps),
                        'n_samp': len(samp_reps),
                        'significant': is_sig
                    }
                    if not is_sig:
                        all_params_sig = False
                    continue

            # Fallback to summary stats when raw replicates are unavailable
            if samp in summary_data and control in summary_data:
                ctrl_stats = summary_data[control].get(param, {})
                samp_stats = summary_data[samp].get(param, {})
                if ctrl_stats and samp_stats:
                    m1, sd1, n1 = ctrl_stats['mean'], ctrl_stats['sd'], ctrl_stats['n']
                    m2, sd2, n2 = samp_stats['mean'], samp_stats['sd'], samp_stats['n']
                    if n1 >= 2 and n2 >= 2:
                        t_stat, p_val = stats.ttest_ind_from_stats(m1, sd1, n1, m2, sd2, n2, equal_var=False)
                        is_valid = not (pd.isna(t_stat) or pd.isna(p_val))
                        is_sig = bool(is_valid and p_val < SIG_ALPHA)
                        samp_results[param] = {
                            'p': round(float(p_val), 4) if is_valid else None,
                            'f_stat': round(float(t_stat), 4) if is_valid else None,
                            'method': 'Welch t-test (summary)',
                            'n_ctrl': n1,
                            'n_samp': n2,
                            'significant': is_sig
                        }
                        if not is_sig:
                            all_params_sig = False
                        continue

            # No data available
            samp_results[param] = {
                'p': None,
                'f_stat': None,
                'method': 'insufficient data',
                'n_ctrl': 0,
                'n_samp': 0,
                'significant': False
            }
            all_params_sig = False

        # Add overall assessment for this sample
        samp_results['_overall'] = {
            'all_params_different': all_params_sig,
            'significant_params': sum(1 for p in samp_results.values() if isinstance(p, dict) and p.get('significant', False))
        }

        results[samp] = samp_results
    return results

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
        xaxis=dict(title="Texture Parameters", tickfont=dict(size=12)),
        yaxis=dict(title="Mean Value", tickfont=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        font=dict(family="Inter", color="#1d1d1f"),
        height=500,
        margin=dict(l=60, r=60, t=80, b=120),
    )
    return fig

def plot_pca(z_df: pd.DataFrame, sample_colors: dict, control: str) -> go.Figure:
    """PCA scatter — samples projected onto PC1 & PC2."""
    pca = PCA(n_components=min(2, z_df.shape[1]))
    coords = pca.fit_transform(z_df.values)
    var_exp = pca.explained_variance_ratio_ * 100
    
    fig = go.Figure()
    for i, sample in enumerate(z_df.index):
        color = sample_colors.get(sample, PALETTE[i % len(PALETTE)])
        symbol = "diamond" if sample == control else "circle"
        fig.add_trace(go.Scatter(
            x=[coords[i, 0]],
            y=[coords[i, 1]],
            mode="markers+text",
            marker=dict(size=12, color=color, symbol=symbol, line=dict(width=2, color="#ffffff")),
            text=[sample],
            textposition="top center",
            textfont=dict(size=10, color="#1d1d1f"),
            name=sample,
            showlegend=False,
        ))
    
    fig.update_layout(
        title=dict(text=f"PCA — Texture Space (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1] if len(var_exp) > 1 else 0:.1f}%)", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(title="PC1", zeroline=True, zerolinewidth=1, zerolinecolor="#e0e0e0"),
        yaxis=dict(title="PC2", zeroline=True, zerolinewidth=1, zerolinecolor="#e0e0e0"),
        font=dict(family="Inter", color="#1d1d1f"),
        height=500,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig

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

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to rgba tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

# ─────────────────────────────────────────────
# IMAGE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def analyze_cake_image_basic(image):
    """Fallback image analysis when robust CV pipeline is unavailable."""
    gray = image.convert("L")
    img_array = np.array(gray)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)

    from scipy import ndimage

    sobel_x = ndimage.sobel(img_array, axis=0)
    sobel_y = ndimage.sobel(img_array, axis=1)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_density = np.mean(edge_magnitude) / 255 * 100

    return {
        "Brightness": round(float(brightness), 1),
        "Contrast": round(float(contrast), 1),
        "Edge Density": round(float(edge_density), 2),
    }

def run_robust_crumb_analysis(uploaded_items: list) -> tuple:
    """Run crumb-analysis pipeline metrics on uploaded files and return scored DataFrame."""
    cfg = AnalyzerConfig(debug_visualization=False, try_imagej=False)
    analyzer = CrumbAnalyzer(cfg)

    rows = []
    image_map = {}
    errors = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for item in uploaded_items:
            file_obj = item.get("file")
            if file_obj is None:
                continue

            sample_label = f"{item['name']} (#{item['index'] + 1})"
            try:
                image = Image.open(file_obj).convert("RGB")
                image_map[sample_label] = image

                tmp_path = tmp_root / f"sample_{item['index']}.png"
                image.save(tmp_path)

                rgb = analyzer._load_image(tmp_path)
                gray, norm, blur = analyzer._preprocess(rgb)
                crumb, pores, roi = analyzer._segment_crumb(blur)

                pore = analyzer._pore_features(pores, roi)
                wall, _, _ = analyzer._wall_features(crumb)
                conn, _, _, _ = analyzer._connectivity_features(crumb)
                frac, _ = analyzer._fracture_features(blur, crumb, pores)
                tex = analyzer._texture_features(norm, roi, crumb)
                spatial = analyzer._spatial_features(pores, roi)

                rows.append(
                    {
                        "Sample name": sample_label,
                        "Porosity": pore["porosity"],
                        "Mean pore size": pore["mean_pore_size"],
                        "Pore CV": pore["pore_cv"],
                        "Circularity": pore["circularity"],
                        "Mean wall thickness": wall["mean_wall_thickness"],
                        "Wall thickness variance": wall["wall_thickness_var"],
                        "Thin region fraction": wall["thin_region_fraction"],
                        "Porosity uniformity": spatial["porosity_uniformity"],
                        "Clustering index": spatial["clustering_index"],
                        "Connectivity ratio": conn["connectivity_ratio"],
                        "Fracture index": frac["fracture_index"],
                        "Homogeneity": tex["glcm_homogeneity"],
                        "GLCM contrast": tex["glcm_contrast"],
                        "GLCM entropy": tex["glcm_entropy"],
                        "Fractal dimension": tex["fractal_dimension"],
                    }
                )
            except Exception as exc:
                errors[sample_label] = str(exc)

    if not rows:
        return None, image_map, errors

    metrics_df = pd.DataFrame(rows)
    scored_df = analyzer._compute_scores(metrics_df)
    scored_df["Interpretation"] = scored_df.apply(analyzer._interpret_row, axis=1)

    numeric_cols = [
        "Porosity",
        "Mean pore size",
        "Pore CV",
        "Circularity",
        "Mean wall thickness",
        "Wall thickness variance",
        "Thin region fraction",
        "Porosity uniformity",
        "Clustering index",
        "Connectivity ratio",
        "Fracture index",
        "Homogeneity",
        "GLCM contrast",
        "GLCM entropy",
        "Fractal dimension",
        "Crumb Strength Score",
    ]
    scored_df[numeric_cols] = scored_df[numeric_cols].round(4)
    return scored_df, image_map, errors

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
    template_df = pd.DataFrame(template_rows)
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
            missing = required - set(csv_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            else:
                # Show preview
                st.markdown("<p class='section-title' style='margin-top:16px;'>Preview</p>", unsafe_allow_html=True)
                st.dataframe(csv_df, use_container_width=True)

                # Support common MaxShear header variants to prevent mis-mapping.
                col_candidates = {
                    "Hardness": ["Hardness"],
                    "Resilience": ["Resilience"],
                    "Cohesiveness": ["Cohesiveness"],
                    "Springiness": ["Springiness"],
                    "Chewiness": ["Chewiness"],
                    "MaxShear": ["MaxShear", "Max Shear Force (N)", "Max Shear", "MaxShearForce"],
                }
                resolved_cols = {}
                for param, candidates in col_candidates.items():
                    resolved_cols[param] = next((c for c in candidates if c in csv_df.columns), None)

                if resolved_cols["MaxShear"] is None:
                    st.error("Missing MaxShear column. Accepted headers: MaxShear, Max Shear Force (N), Max Shear, MaxShearForce")
                    st.stop()

                csv_input = {}
                sample_n_table = {}  # for display

                for sname, grp in csv_df.groupby("Sample", sort=False):
                    sname = str(sname).strip()
                    pdata = {}
                    n_info = {}
                    for param in ALL_PARAMS:
                        col = resolved_cols.get(param)
                        if col is not None:
                            vals = pd.to_numeric(grp[col], errors="coerce").dropna().tolist()
                            pdata[param] = {"reps": vals}
                            n_info[param] = len(vals)
                        else:
                            pdata[param] = {"reps": []}
                            n_info[param] = 0
                    csv_input[sname] = pdata
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
    
    # Run Analysis Button
    analysis_done = "analysis_done" in st.session_state
    if analysis_done:
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <button style="background-color: #30d158; color: white; border: none; border-radius: 8px; padding: 12px 24px; font-size: 16px; font-weight: 500; cursor: default;" disabled>
                ✅ Analysis Complete
            </button>
        </div>
        """, unsafe_allow_html=True)
        run_analysis = False
    else:
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
    means_rows = {}
    sd_rows = {}

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
                s_row[param] = float(pd_.get("sd", 0.0))
        means_rows[samp] = m_row
        sd_rows[samp] = s_row
    return pd.DataFrame(means_rows).T, pd.DataFrame(sd_rows).T

def build_raw_and_summary(input_data, sample_names, n_replicates_default):
    """
    Separate raw replicate lists (for direct t-tests) from summary dicts (for analytical t-tests).
    Never fabricates data points from mean ± SD.
    Returns:
        raw_data: {sample: {param: [rep1, rep2, ...]}} — only populated when replicates are available
        summary_data: {sample: {param: {mean, sd, n}}} — always populated (computed from reps if available)
    """
    raw_data = {}
    summary_data = {}

    for samp in sample_names:
        if samp not in input_data:
            continue
        raw_params = {}
        sum_params = {}
        for param in ALL_PARAMS:
            pd_ = input_data[samp].get(param, {})
            if "reps" in pd_:
                reps = pd_["reps"]
                raw_params[param] = reps
                m = float(np.mean(reps))
                s = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
                sum_params[param] = {"mean": m, "sd": s, "n": len(reps)}
            else:
                # Summary mode — no data available
                m = float(pd_.get("mean", 1.0))
                s = float(pd_.get("sd", 0.0))
                n = int(pd_.get("n", n_replicates_default))
                raw_params[param] = []  # empty — signals "no raw replicates"
                sum_params[param] = {"mean": m, "sd": s, "n": n}
        raw_data[samp] = raw_params
        summary_data[samp] = sum_params

    return raw_data, summary_data

if run_analysis or "analysis_done" in st.session_state:
    if run_analysis:
        st.session_state["analysis_done"] = True
        st.session_state["input_data"] = input_data
        st.session_state["sample_names"] = sample_names
        st.session_state["control"] = control_sample
        st.session_state["major_thresh"] = major_thresh
        st.session_state["minor_thresh"] = show_minor_thresh

    # Retrieve from session
    _input = st.session_state.get("input_data", input_data)
    _samples = st.session_state.get("sample_names", sample_names)
    _control = st.session_state.get("control", control_sample)
    _major = st.session_state.get("major_thresh", major_thresh)
    _minor = st.session_state.get("minor_thresh", show_minor_thresh)

    means_df, sd_df = build_means_sd_df(_input, _samples)
    raw_data, summary_data = build_raw_and_summary(_input, _samples, int(n_replicates))

    missing_samples = [s for s in _samples if s not in means_df.index]
    if missing_samples:
        st.warning(
            f"These sample names were not found in uploaded data and will be ignored: {', '.join(missing_samples)}. "
            "Please ensure the Sample column matches the configured sample names."
        )

    if _control not in means_df.index:
        st.error(f"Control sample '{_control}' not found in data. Please check sample names.")
        st.stop()

    _samples = [s for s in _samples if s in means_df.index]

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
    ranked = sorted(non_ctrl, key=lambda s: scores_series.get(s, 0), reverse=True)

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
            
            # Check if any parameters are statistically significantly different
            has_sig_diff = False
            if samp in stat_results and '_overall' in stat_results[samp]:
                has_sig_diff = stat_results[samp]['_overall'].get('significant_params', 0) > 0
            
            # Assessment: prioritize statistical significance over Euclidean distance
            if not has_sig_diff:
                # No significant differences = very close, regardless of score
                label = "Very close (not significantly different)"
            elif sc >= 75:
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
            if samp not in means_df.index:
                continue  # Skip samples not in data
            sc = scores_series.get(samp, 0)
            dist = distances_series.get(samp, 0)

            # % diffs
            all_pct = {}
            for param in ALL_PARAMS:
                ctrl_val = means_df.loc[_control, param]
                samp_val = means_df.loc[samp, param]
                p = pct_diff(samp_val, ctrl_val)
                if p is not None:
                    all_pct[param] = p

            # Filtered diffs (above minor threshold), but always include MaxShear even if small change
            # MaxShear is critical for cake structure, so show all changes (including 0%)
            shown_pct = {
                k: v for k, v in all_pct.items()
                if abs(v) >= _minor or k == "MaxShear"
            }

            # Significant params
            sig_params = []
            if samp in stat_results:
                for param, test in stat_results[samp].items():
                    pval = test.get("p")
                    method = test.get("method", "")
                    n_c = test.get("n_ctrl", 0)
                    n_s = test.get("n_samp", 0)
                    if pval is not None and pval < SIG_ALPHA:
                        sig_params.append((param, pval, method, n_c, n_s))

            # Interpretation
            interp_pct = {
                k: v for k, v in all_pct.items()
                if abs(v) >= _minor or k == "MaxShear"
            }
            # Pass statistical test results for MaxShear significance check
            sample_stat_tests = stat_results.get(samp, {}) if samp in stat_results else {}
            summary_text, reasoning_lines, fixes = interpret_sample(interp_pct, means_df.loc[samp], means_df.loc[_control], sample_stat_tests)

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

            # MaxShear specific analysis - critical for cake structure
            if "MaxShear" in means_df.columns and samp in means_df.index and _control in means_df.index:
                ms_samp = means_df.loc[samp, "MaxShear"]
                ms_ctrl = means_df.loc[_control, "MaxShear"]
                ms_pct = pct_diff(ms_samp, ms_ctrl)

                ms_sig = False
                ms_p_val = None
                if samp in stat_results and "MaxShear" in stat_results[samp]:
                    ms_stat = stat_results[samp]["MaxShear"]
                    ms_sig = ms_stat.get('significant', False)
                    ms_p_val = ms_stat.get('p')

                st.markdown("**Cake Structure Analysis (Max Shear):**")
                structure_status = ""
                if ms_pct is not None:
                    if ms_pct < 0:
                        # Only flag as CRUMBLY if statistically significant
                        if ms_sig:
                            structure_status = f"⚠️ **Lower Max Shear ({ms_pct:.1f}%) - CRUMBLY structure**"
                        else:
                            structure_status = f"✅ **Lower Max Shear ({ms_pct:.1f}%) - Not significantly different**"
                    elif ms_pct >= 15:
                        structure_status = f"✅ **Higher Max Shear ({ms_pct:.1f}%) - Stronger structure than control**"
                    else:
                        structure_status = f"✅ **Max Shear similar to control ({ms_pct:.1f}%)**"

                if ms_p_val is not None:
                    structure_status += f" | p-value: {ms_p_val:.3f}"
                    if ms_sig:
                        structure_status += " (significant)"
                    else:
                        structure_status += " (not significant)"

                st.markdown(structure_status)

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

            r_col1, r_col2, r_col3 = st.columns(3)

            # Column 1: Key differences — only medium/major, skip near-zero non-sig
            with r_col1:
                st.markdown("<p class='section-title'>Key Differences</p>", unsafe_allow_html=True)
                # Build filtered list: only medium (>=_minor) or major (>=_major), skip ~0 non-sig
                meaningful_pct = {}
                for param, pct in all_pct.items():
                    is_sig = sample_stat_tests.get(param, {}).get('significant', False)
                    if abs(pct) < 2 and not is_sig:
                        continue  # near-zero and non-significant — skip entirely
                    if abs(pct) >= _minor or is_sig:
                        meaningful_pct[param] = pct

                if meaningful_pct:
                    higher_items = sorted(
                        [(p, v) for p, v in meaningful_pct.items() if v > 0],
                        key=lambda x: abs(x[1]), reverse=True
                    )
                    lower_items = sorted(
                        [(p, v) for p, v in meaningful_pct.items() if v < 0],
                        key=lambda x: abs(x[1]), reverse=True
                    )
                    if higher_items:
                        st.markdown("**Higher than control (↑):**")
                        for param, pct in higher_items:
                            tag_class = "tag-major" if abs(pct) >= _major else "tag-minor"
                            st.markdown(f"<div class='{tag_class}'>{param}: +{pct:.1f}%</div>", unsafe_allow_html=True)
                    if lower_items:
                        st.markdown("**Lower than control (↓):**")
                        for param, pct in lower_items:
                            tag_class = "tag-major" if abs(pct) >= _major else "tag-minor"
                            st.markdown(f"<div class='{tag_class}'>{param}: {pct:.1f}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='sub-text'>No meaningful difference from control</div>", unsafe_allow_html=True)

            # Column 2: Statistical significance
            with r_col2:
                st.markdown("<p class='section-title'>Statistical Significance</p>", unsafe_allow_html=True)
                if sig_params:
                    for param, pval, method, n_c, n_s in sig_params[:3]:  # Limit to 3
                        st.markdown(f"<div class='tag-sig'>{param}: p={pval:.3f} ({method})</div>", unsafe_allow_html=True)
                    if len(sig_params) > 3:
                        st.markdown(f"<div class='sub-text'>+{len(sig_params)-3} more significant differences</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='sub-text'>No statistically significant differences detected</div>", unsafe_allow_html=True)

            # Column 3: Interpretation & fixes
            with r_col3:
                st.markdown("<p class='section-title'>Interpretation</p>", unsafe_allow_html=True)
                if reasoning_lines:
                    for line in reasoning_lines[:2]:
                        st.markdown(f"<div class='interp-block'>{line}</div>", unsafe_allow_html=True)
                if fixes:
                    st.markdown("<p class='section-title' style='margin-top:16px;'>Recommendations</p>", unsafe_allow_html=True)
                    for fix in fixes[:2]:
                        st.markdown(f"<div class='fix-item'>{fix}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── RAW DATA TABLE ──
        with st.expander("Raw Data — Means & Standard Deviations"):
            combined = pd.DataFrame()
            for samp in _samples:
                if samp not in means_df.index:
                    continue
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
        st.markdown("**Key Insights:**")
        # Generate sample-specific insights
        insights = []
        for samp in _samples:
            if samp == _control:
                insights.append(f"- **{_control}** (control): Reference sample with baseline texture properties")
            else:
                # Compare to control — MaxShear captured separately, not mixed into TPA param list
                higher_params = []
                lower_params = []
                maxshear_diff = None
                for param in ALL_PARAMS:
                    if samp in means_df.index and _control in means_df.index:
                        samp_val = means_df.loc[samp, param]
                        ctrl_val = means_df.loc[_control, param]
                        pct_diff_val = pct_diff(samp_val, ctrl_val)
                        if pct_diff_val is not None:
                            if param == "MaxShear":
                                maxshear_diff = pct_diff_val
                                continue  # report separately below, not in TPA params list
                            if abs(pct_diff_val) >= 10:
                                if pct_diff_val > 0:
                                    higher_params.append(f"{param.lower()} (+{pct_diff_val:.0f}%)")
                                else:
                                    lower_params.append(f"{param.lower()} ({pct_diff_val:.0f}%)")

                # Determine MaxShear significance for insight
                ms_sig_bar = False
                if samp in stat_results and "MaxShear" in stat_results[samp]:
                    ms_sig_bar = stat_results[samp]["MaxShear"].get("significant", False)

                insight = f"- **{samp}**: "
                if not higher_params and not lower_params:
                    insight += "Similar texture profile to control"
                else:
                    parts = []
                    if higher_params:
                        parts.append(f"Higher in {', '.join(higher_params[:3])}")
                    if lower_params:
                        parts.append(f"Lower in {', '.join(lower_params[:3])}")
                    insight += "; ".join(parts)

                # MaxShear structural note — use actual direction correctly
                if maxshear_diff is not None and abs(maxshear_diff) >= 5:
                    if maxshear_diff > 0:
                        if ms_sig_bar:
                            insight += f" — **MaxShear higher by {maxshear_diff:.0f}% (sig.) → stronger, firmer structure**"
                        else:
                            insight += f" — MaxShear higher by {maxshear_diff:.0f}% → stronger structure"
                    else:
                        if ms_sig_bar:
                            insight += f" — **MaxShear lower by {abs(maxshear_diff):.0f}% (sig.) → CRUMBLY, weaker structure**"
                        elif abs(maxshear_diff) >= 10:
                            insight += f" — MaxShear lower by {abs(maxshear_diff):.0f}% → may affect structure"

                # Add statistical significance if available
                if samp in stat_results and '_overall' in stat_results[samp]:
                    overall = stat_results[samp]['_overall']
                    if overall['all_params_different']:
                        insight += " - all parameters significantly different"
                    elif overall['significant_params'] > 0:
                        insight += f" - {overall['significant_params']} parameters significantly different"

                    # Specifically check MaxShear significance
                    if "MaxShear" in stat_results[samp] and stat_results[samp]["MaxShear"].get('significant', False):
                        ms_p = stat_results[samp]["MaxShear"].get('p')
                        if ms_p is not None:
                            insight += f" - MaxShear p={ms_p:.3f}"

                insights.append(insight)

        for insight in insights:
            st.markdown(insight)
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
                    st.markdown("**Key Insights:**")
                    pca_insights = []

                    # Rebuild the PCA model to compute loadings
                    pca_model = PCA(n_components=min(2, z_df.shape[1]))
                    pca_coords = pca_model.fit_transform(z_df.values)
                    var_exp = pca_model.explained_variance_ratio_ * 100
                    pca_df = pd.DataFrame(
                        pca_coords,
                        index=z_df.index,
                        columns=["PC1", "PC2"][:pca_coords.shape[1]]
                    )
                    if "PC2" not in pca_df.columns:
                        pca_df["PC2"] = 0.0

                    # Loadings: which parameters drive each PC
                    loadings = pd.DataFrame(
                        pca_model.components_.T,
                        index=z_df.columns,
                        columns=["PC1", "PC2"][:pca_model.n_components_]
                    )
                    if "PC2" not in loadings.columns:
                        loadings["PC2"] = 0.0
                    pc1_top = loadings["PC1"].abs().nlargest(2).index.tolist()
                    pc2_top = loadings["PC2"].abs().nlargest(2).index.tolist()

                    # Per-sample Euclidean distances from control in PCA space
                    if _control in pca_df.index:
                        ctrl_pca = pca_df.loc[_control, ["PC1", "PC2"]].values
                        dist_map = {}
                        for s in _samples:
                            if s != _control and s in pca_df.index:
                                d = float(np.linalg.norm(ctrl_pca - pca_df.loc[s, ["PC1", "PC2"]].values))
                                dist_map[s] = round(d, 2)

                        if dist_map:
                            sorted_dist = sorted(dist_map.items(), key=lambda x: x[1])
                            closest_sample = sorted_dist[0][0]
                            furthest_sample = sorted_dist[-1][0]
                            closest_dist = sorted_dist[0][1]
                            furthest_dist = sorted_dist[-1][1]
                            
                            # Decision-driving conclusions only
                            pca_insights.append(
                                f"**{closest_sample}** is texture-closest to control ({closest_dist:.2f} units). "
                                f"**{furthest_sample}** shows the most divergent profile ({furthest_dist:.2f} units)."
                            )
                            
                            # Mechanistic insight on separation drivers
                            if closest_dist > 1.5:
                                pca_insights.append(
                                    f"Even the closest formulation differs noticeably. "
                                    f"Key drivers: **{' + '.join(pc1_top)}**. Consider formulation adjustments."
                                )
                            elif furthest_dist > 2.5:
                                pca_insights.append(
                                    f"**{furthest_sample}** represents a significant texture shift driven by **{' and '.join(pc1_top)}**. "
                                    f"Verify this is intentional."
                                )
                            else:
                                pca_insights.append(
                                    f"Formulations cluster tightly, indicating consistent process control."
                                )

                    # Cluster groupings when ≥ 4 samples
                    if len(_samples) > 3:
                        from sklearn.cluster import KMeans
                        try:
                            pca_subset = pca_df.loc[_samples, ["PC1", "PC2"]].values
                            kmeans = KMeans(n_clusters=min(3, len(_samples)), random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(pca_subset)
                            cluster_groups: dict = {}
                            for ci, s in enumerate(_samples):
                                cluster_groups.setdefault(clusters[ci], []).append(s)
                            for cid, members in cluster_groups.items():
                                if len(members) > 1:
                                    pca_insights.append(
                                        f"- **{', '.join(members)}** cluster together — similar overall texture behaviour."
                                    )
                        except Exception:
                            pass

                    for insight in pca_insights:
                        st.markdown(insight)
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
                st.markdown("**Key Insights:**")
                radar_insights = []

                # Z-score threshold for "meaningfully different" (0.5 SD)
                ZSCORE_THRESH = 0.5

                for samp in _samples:
                    if samp != _control and samp in z_df.index:
                        ctrl_z = z_df.loc[_control, ALL_PARAMS]
                        samp_z = z_df.loc[samp, ALL_PARAMS]
                        diff_z = samp_z - ctrl_z

                        # STRUCTURE-FOCUSED Z-SCORE INTERPRETATION (Hardness ignored completely)
                        # Step 1: Categorize deviations
                        structure_params = ["Cohesiveness", "Springiness", "Resilience", "Chewiness", "MaxShear"]
                        z_deviations = {}
                        for param in structure_params:
                            if param in ALL_PARAMS:
                                z_val = diff_z[param]
                                abs_z = abs(z_val)
                                if abs_z > 0.5:  # Meaningful deviation threshold
                                    if abs_z > 1.5:
                                        magnitude = "large"
                                    else:
                                        magnitude = "moderate"
                                    direction = "reduced" if z_val < 0 else "enhanced"
                                    z_deviations[param] = (z_val, magnitude, direction)
                        
                        # Step 2: Apply failure rule
                        # If ≥2 of {Cohesiveness, Springiness, Resilience} have |z| > 1.5 → WEAK
                        critical_params = ["Cohesiveness", "Springiness", "Resilience"]
                        large_reductions = sum(
                            1 for p in critical_params 
                            if p in z_deviations and z_deviations[p][1] == "large" and z_deviations[p][2] == "reduced"
                        )
                        
                        # Step 3: Classify
                        if large_reductions >= 2:
                            structure_class = "WEAK"
                            decision = "Reject"
                        elif any(z_deviations.get(p, (None, None, "enhanced"))[2] == "reduced" for p in critical_params if z_deviations.get(p)):
                            structure_class = "SLIGHTLY WEAKENED"
                            decision = "Adjust"
                        elif z_deviations:
                            structure_class = "STRONG"
                            decision = "Keep"
                        else:
                            structure_class = "STRONG"
                            decision = "Keep"
                        
                        # Step 4: Build insight
                        if not z_deviations:
                            radar_insights.append(
                                f"- **{samp}**: Structure identical to control. "
                                f"**Decision: {decision}** — No adjustments needed. Perfect match."
                            )
                        else:
                            # Key structural deviations (max 2)
                            key_issues = []
                            for param in ["Cohesiveness", "Springiness", "Resilience"]:  # Priority order
                                if param in z_deviations:
                                    z_val, mag, direc = z_deviations[param]
                                    if param == "Cohesiveness" and direc == "reduced":
                                        key_issues.append(f"internal bonding weaker ({mag})")
                                    elif param == "Springiness" and direc == "reduced":
                                        key_issues.append(f"elastic recovery impaired ({mag})")
                                    elif param == "Resilience" and direc == "reduced":
                                        key_issues.append(f"recovery delayed ({mag})")
                            
                            # Support with secondary metrics if structural issues absent
                            if not key_issues:
                                for param in ["MaxShear", "Chewiness"]:
                                    if param in z_deviations:
                                        z_val, mag, direc = z_deviations[param]
                                        if param == "MaxShear" and direc == "reduced":
                                            key_issues.append(f"shear strength lower — crumble risk")
                                        elif param == "Chewiness" and direc == "enhanced":
                                            key_issues.append(f"texture more labored")
                            
                            # Format output
                            insight = f"- **{samp}**: "
                            if key_issues:
                                insight += f"{', '.join(key_issues[:2])}. "
                            insight += f"**Structure: {structure_class}**. **Decision: {decision}**"
                            if structure_class == "WEAK":
                                insight += " — Internal structure compromised; reject batch."
                            elif structure_class == "SLIGHTLY WEAKENED":
                                insight += " — Elasticity may need improvement; acceptable if minor."
                            
                            radar_insights.append(insight)

                if not radar_insights:
                    radar_insights.append("- Samples show varied texture profiles compared to control.")

                for insight in radar_insights:
                    st.markdown(insight)
            else:
                st.info("Radar chart requires at least 2 samples.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Percentage difference heatmap vs control
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Percentage Difference vs Control</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Each cell shows the percentage difference of a test sample from the control. Red = lower than control, blue = higher.</p>", unsafe_allow_html=True)

        pct_rows = {}
        for samp in non_ctrl:
            if samp not in means_df.index:
                continue
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
            hoverongaps=False,
        ))

        fig_heat.update_layout(
            title=dict(text="Percentage Difference Heatmap", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(title="Texture Parameters", tickfont=dict(size=12)),
            yaxis=dict(title="Samples", tickfont=dict(size=12), autorange="reversed"),
            font=dict(family="Inter", color="#1d1d1f"),
            height=400,
            margin=dict(l=60, r=60, t=80, b=60),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("**Key Insights:**")
        # Generate heatmap insights
        heatmap_insights = []
        
        # Find which parameters show the most and least variation
        param_variation = {}
        for param in ALL_PARAMS:
            param_vals = [pct_hm_df.loc[samp, param] for samp in pct_hm_df.index]
            param_variation[param] = np.std(param_vals) if param_vals else 0
        
        most_variable_param = max(param_variation, key=param_variation.get)
        least_variable_param = min(param_variation, key=param_variation.get)
        
        if param_variation[most_variable_param] > 0:
            heatmap_insights.append(f"- **{most_variable_param}** shows the most variation across samples, indicating it's the key differentiator between formulations.")
        if param_variation[least_variable_param] > 0 and least_variable_param != most_variable_param:
            heatmap_insights.append(f"- **{least_variable_param}** remains most consistent across samples, showing stable performance regardless of formulation changes.")
        
        # Find which samples differ most from control
        sample_variation = {}
        for samp in pct_hm_df.index:
            sample_vals = [abs(pct_hm_df.loc[samp, param]) for param in pct_hm_df.columns]
            sample_variation[samp] = np.mean(sample_vals)
        
        if sample_variation:
            most_different_samp = max(sample_variation, key=sample_variation.get)
            least_different_samp = min(sample_variation, key=sample_variation.get)
            heatmap_insights.append(f"- **{most_different_samp}** differs most from control overall (most colored row).")
            if least_different_samp != most_different_samp:
                heatmap_insights.append(f"- **{least_different_samp}** is closest to control overall (lightest row).")
        
        # Check for MaxShear issues
        if "MaxShear" in pct_hm_df.columns:
            maxshear_vals = pct_hm_df["MaxShear"].values
            samples_lower_maxshear = [pct_hm_df.index[i] for i, v in enumerate(maxshear_vals) if v < 0]
            if samples_lower_maxshear:
                heatmap_insights.append(f"- **Red in MaxShear column** for {', '.join(samples_lower_maxshear)}: These samples have lower shear force (potential structure issues).")
        
        for insight in heatmap_insights:
            st.markdown(insight)
        st.markdown("</div>", unsafe_allow_html=True)

        # Parameter Correlation Analysis — MaxShear vs All Other Variables
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>MaxShear Relationships</p>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-text'>Spearman ρ (rank-based, robust to outliers) and Pearson R² (linear variance explained) "
            "for MaxShear vs TPA parameters (Hardness excluded). "
            "<b>Very Strong</b>: |ρ| ≥ 0.8 &nbsp;|&nbsp; <b>Strong</b>: 0.6–0.8 &nbsp;|&nbsp; "
            "<b>Moderate</b>: 0.4–0.6 &nbsp;|&nbsp; <b>Weak</b>: &lt; 0.4. "
            "R² ≥ 0.50 means MaxShear explains ≥ 50% of that parameter's variance across samples.</p>",
            unsafe_allow_html=True,
        )

        # Calculate robust Spearman ρ and Pearson R² for MaxShear vs TPA variables (excluding Hardness)
        maxshear_corr_results = compute_maxshear_relationships(means_df, min_points=3)

        # Display table
        if maxshear_corr_results:
            corr_table_data = []

            def _sort_key(x):
                v = x.get('spearman_rho', 'Not computed')
                return abs(v) if isinstance(v, float) else -1

            for result in sorted(maxshear_corr_results, key=_sort_key, reverse=True):
                rho = result['spearman_rho']
                r2  = result['r_squared']
                pv  = result['p_value']
                sig = result['significant']
                corr_table_data.append({
                    'Parameter':             result['parameter'],
                    'n pairs':               result['n'],
                    'Spearman ρ':            rho,
                    'R² (Pearson)':          r2,
                    'p-value':               pv,
                    'Strength':              result['strength'],
                    'Direction':             result['direction'],
                    'Robust':                'Yes' if result.get('robust', False) else 'No',
                    'Significant (p<0.05)':  ('✓' if sig else '✗') if sig is not None else 'Not computed',
                    'Note':                  result.get('note', ''),
                })

            corr_df = pd.DataFrame(corr_table_data)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

            # Interpretation — 4-tier: Very Strong / Strong / Moderate / Weak
            st.markdown("**Interpretation:**")
            computed = [r for r in maxshear_corr_results if isinstance(r.get('spearman_rho'), float)]
            not_computed = [r for r in maxshear_corr_results if not isinstance(r.get('spearman_rho'), float)]

            tier_labels = ["Very Strong", "Strong", "Moderate", "Weak"]
            tier_found = {t: False for t in tier_labels}
            for tier in tier_labels:
                tier_rows = [r for r in computed if r['strength'] == tier]
                if not tier_rows:
                    continue
                tier_found[tier] = True
                for corr in sorted(tier_rows, key=lambda x: abs(x['spearman_rho']), reverse=True):
                    arrow = "increases with" if corr['direction'] == 'positive' else "decreases with"
                    sig_note = "p<0.05 ✓" if corr['significant'] else "p≥0.05"
                    r2_note = f"R²={corr['r_squared']:.3f} ({corr['r_squared']*100:.0f}% variance explained)"
                    st.markdown(
                        f"- **{tier}** — MaxShear {arrow} **{corr['parameter']}** "
                        f"(n={corr['n']}, ρ={corr['spearman_rho']:+.3f}, {r2_note}, {sig_note})"
                    )

            if not any(tier_found[t] for t in ["Very Strong", "Strong"]):
                st.markdown("- No strong correlations (|ρ| ≥ 0.6) detected — MaxShear may vary independently of TPA parameters in this dataset.")

            if not_computed:
                for row in not_computed:
                    st.markdown(f"- **{row['parameter']}**: {row.get('note', 'Not computed')}")
        else:
            st.markdown("_Insufficient data to calculate correlations. Need at least 3 samples._")

        st.markdown("</div>", unsafe_allow_html=True)

    # Crumb image analysis tab renders outside the analysis block so users can upload images immediately.

# ─────────────────────────────────────────────
# TAB 4 — CRUMB IMAGE ANALYSIS
# ─────────────────────────────────────────────
with tab_image:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Cake Crumb Image Analysis</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-text'>Upload crumb images per sample to quantify brightness, contrast, and edge density, then compare qualitative bake and structure notes side by side.</p>",
        unsafe_allow_html=True,
    )

    uploaded_images = []
    cols = st.columns(2)
    for i, sname in enumerate(sample_names):
        with cols[i % 2]:
            upload = st.file_uploader(
                f"Upload image — {sname}",
                type=["jpg", "jpeg", "png"],
                key=f"crumb_upload_{i}",
            )
            uploaded_images.append({"index": i, "name": sname, "file": upload})

    uploaded_count = sum(1 for item in uploaded_images if item["file"] is not None)
    st.markdown(
        f"<p class='sub-text' style='margin-top:8px;'>Uploaded {uploaded_count} / {len(sample_names)} sample images.</p>",
        unsafe_allow_html=True,
    )

    if uploaded_count > 0:
        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

        try:
            robust_df, image_map, robust_errors = run_robust_crumb_analysis(uploaded_images)
        except Exception as exc:
            robust_df, image_map, robust_errors = None, {}, {"Pipeline": str(exc)}

        if robust_df is not None and not robust_df.empty:
            st.markdown("<p class='section-title'>Robust Crumb Metrics (all uploaded samples)</p>", unsafe_allow_html=True)
            display_cols = [
                "Sample name",
                "Porosity",
                "Mean pore size",
                "Pore CV",
                "Circularity",
                "Porosity uniformity",
                "Clustering index",
                "Mean wall thickness",
                "Wall thickness variance",
                "Thin region fraction",
                "Connectivity ratio",
                "Fracture index",
                "Homogeneity",
                "GLCM contrast",
                "GLCM entropy",
                "Fractal dimension",
                "Crumb Strength Score",
                "Classification",
            ]
            st.dataframe(robust_df[display_cols].sort_values("Crumb Strength Score", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("<p class='sub-text'>Easy read: higher wall thickness and connectivity with lower fracture index generally indicates stronger crumb structure.</p>", unsafe_allow_html=True)

            for _, row in robust_df.sort_values("Crumb Strength Score", ascending=False).iterrows():
                sample_label = row["Sample name"]
                st.markdown(f"<p class='section-title' style='margin-top:14px;'>{sample_label}</p>", unsafe_allow_html=True)
                col_img1, col_img2 = st.columns([1, 2])
                with col_img1:
                    if sample_label in image_map:
                        st.image(image_map[sample_label], use_container_width=True, caption=f"{sample_label} crumb")
                with col_img2:
                    st.markdown(
                        f"**Classification:** {row['Classification']} | "
                        f"**Score:** {row['Crumb Strength Score']:.1f}/100"
                    )
                    st.markdown(f"**Interpretation:** {row['Interpretation']}")

                    quick_metrics = pd.DataFrame(
                        {
                            "Metric": [
                                "Porosity",
                                "Mean pore size",
                                "Cell size CV (Pore CV)",
                                "Circularity",
                                "Porosity uniformity (std)",
                                "Clustering index",
                                "Mean wall thickness",
                                "Wall thickness variance",
                                "Connectivity ratio",
                                "Fracture index",
                                "Homogeneity (GLCM)",
                            ],
                            "Value": [
                                row["Porosity"],
                                row["Mean pore size"],
                                row["Pore CV"],
                                row["Circularity"],
                                row["Porosity uniformity"],
                                row["Clustering index"],
                                row["Mean wall thickness"],
                                row["Wall thickness variance"],
                                row["Connectivity ratio"],
                                row["Fracture index"],
                                row["Homogeneity"],
                            ],
                        }
                    )
                    st.dataframe(quick_metrics, use_container_width=True, hide_index=True)
        else:
            st.warning("Robust crumb pipeline could not extract advanced metrics from the uploaded images.")

        if robust_errors:
            for sample_label, err in robust_errors.items():
                st.error(f"Could not analyze {sample_label}: {err}")

        # Fallback basic metrics if robust output is unavailable.
        if robust_df is None or robust_df.empty:
            st.markdown("<p class='section-title' style='margin-top:14px;'>Fallback Basic Metrics</p>", unsafe_allow_html=True)
            for item in uploaded_images:
                sname = item["name"]
                file_obj = item["file"]
                if file_obj is None:
                    continue
                try:
                    image = Image.open(file_obj)
                    basic = analyze_cake_image_basic(image)
                    st.markdown(f"**{sname}** — Brightness: {basic['Brightness']}, Contrast: {basic['Contrast']}, Edge Density: {basic['Edge Density']}")
                except Exception:
                    continue
    else:
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#aeaeb2;font-size:14px;'>"
            "Upload one or more sample crumb images above to start analysis."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0;color:#6e6e73;font-size:12px;border-top:1px solid #e8e8ed;margin-top:60px;'>
    Cakamoka — Texture Profile Analysis Tool | Built with Streamlit
</div>
""", unsafe_allow_html=True)