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
                    major_issues.append("Much lower Kramer shear than control - CRUMBLY, weak structure")
                elif ms_pct <= -15:
                    major_issues.append("Lower Kramer shear than control - CRUMBLY structure")
                else:  # -15 to 0
                    minor_issues.append("Lower Kramer shear than control - CRUMBLY tendencies")
            else:
                # Not statistically significant, so don't call it crumbly
                if ms_pct <= -15:
                    minor_issues.append("Lower Kramer shear than control (not significant)")
                else:
                    good_aspects.append("Kramer shear slightly lower but similar to control")
        elif abs(ms_pct) >= 30:
            major_issues.append("Much higher Kramer shear than control")
        elif abs(ms_pct) >= 15:
            minor_issues.append("Higher Kramer shear than control")
        elif abs(ms_pct) >= 5:
            minor_issues.append("Slightly higher Kramer shear than control")
        else:
            good_aspects.append("Kramer shear similar to control")
    
    # Generate summary
    if major_issues:
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
        # Check if MaxShear is a major issue
        if any("MaxShear" in issue.lower() or "kramer shear" in issue.lower() for issue in major_issues):
            reasoning.append("MaxShear differences indicate potential issues with cake structure and crumb integrity.")
    if minor_issues:
        reasoning.append("Minor differences may be due to natural variation or slight adjustments.")
        # Check if MaxShear is a minor issue
        if any("MaxShear" in issue.lower() or "kramer shear" in issue.lower() for issue in minor_issues):
            reasoning.append("MaxShear variations may affect cake texture and structural stability.")
    if good_aspects:
        reasoning.append("Similar aspects indicate consistency with control formulation.")
    
    # Fixes
    fixes = []
    if major_issues:
        fixes.append("Review formulation ingredients and ratios.")
        fixes.append("Check mixing and baking processes.")
        # Specific fixes for MaxShear/structure issues
        if any("MaxShear" in issue.lower() or "kramer shear" in issue.lower() or "structure" in issue.lower() for issue in major_issues):
            fixes.append("Address structural issues: check flour quality, fat content, and baking temperature.")
            fixes.append("Consider adding stabilizers or adjusting moisture content for better crumb structure.")
    if minor_issues:
        fixes.append("Monitor process consistency.")
        fixes.append("Consider slight ingredient adjustments.")
        # Specific fixes for MaxShear issues
        if any("MaxShear" in issue.lower() or "kramer shear" in issue.lower() or "structure" in issue.lower() for issue in minor_issues):
            fixes.append("Fine-tune formulation for improved structural integrity.")
    if not major_issues and not minor_issues:
        fixes.append("Current formulation is well-matched to control.")
    
    return summary, reasoning, fixes

def z_score_standardize(means_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize each parameter (column) across samples."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(means_df)
    return pd.DataFrame(scaled, index=means_df.index, columns=means_df.columns)

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

def z_score_standardize(means_df: pd.DataFrame) -> pd.DataFrame:

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
    """Run one-way ANOVA for each parameter between control and each sample."""
    results = {}
    for samp in samples:
        if samp == control:
            continue
        samp_results = {}
        all_params_sig = True  # Track if all parameters are significantly different

        for param in ALL_PARAMS:
            # Try analytical ANOVA first (if summary stats available)
            if samp in summary_data and control in summary_data:
                ctrl_stats = summary_data[control].get(param, {})
                samp_stats = summary_data[samp].get(param, {})
                if ctrl_stats and samp_stats:
                    m1, sd1, n1 = ctrl_stats['mean'], ctrl_stats['sd'], ctrl_stats['n']
                    m2, sd2, n2 = samp_stats['mean'], samp_stats['sd'], samp_stats['n']
                    if n1 >= 2 and n2 >= 2:
                        # Use F-test approximation for ANOVA with 2 groups
                        # F = t² for 2 groups
                        t_stat, p_val = stats.ttest_ind_from_stats(m1, sd1, n1, m2, sd2, n2, equal_var=False)
                        f_stat = t_stat ** 2
                        # For ANOVA with 2 groups, df = n1 + n2 - 2
                        df = n1 + n2 - 2
                        p_val_anova = 1 - stats.f.cdf(f_stat, 1, df)
                        samp_results[param] = {
                            'p': round(p_val_anova, 4),
                            'f_stat': round(f_stat, 4),
                            'method': 'ANOVA (analytical)',
                            'n_ctrl': n1,
                            'n_samp': n2,
                            'significant': p_val_anova < SIG_ALPHA
                        }
                        if p_val_anova >= SIG_ALPHA:
                            all_params_sig = False
                        continue

            # Fallback to replicate ANOVA
            if samp in raw_data and control in raw_data:
                ctrl_reps = raw_data[control].get(param, [])
                samp_reps = raw_data[samp].get(param, [])
                if len(ctrl_reps) >= 2 and len(samp_reps) >= 2:
                    # One-way ANOVA with 2 groups
                    all_data = ctrl_reps + samp_reps
                    groups = ['control'] * len(ctrl_reps) + [samp] * len(samp_reps)
                    f_stat, p_val = stats.f_oneway(ctrl_reps, samp_reps)
                    samp_results[param] = {
                        'p': round(p_val, 4),
                        'f_stat': round(f_stat, 4),
                        'method': 'ANOVA (replicates)',
                        'n_ctrl': len(ctrl_reps),
                        'n_samp': len(samp_reps),
                        'significant': p_val < SIG_ALPHA
                    }
                    if p_val >= SIG_ALPHA:
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

def analyze_cake_image(image):
    """Analyze cake crumb image for texture parameters."""
    # Convert to grayscale
    gray = image.convert('L')
    img_array = np.array(gray)
    
    # Brightness (mean intensity)
    brightness = np.mean(img_array)
    
    # Contrast (std of intensity)
    contrast = np.std(img_array)
    
    # Edge density (using Sobel)
    from scipy import ndimage
    sobel_x = ndimage.sobel(img_array, axis=0)
    sobel_y = ndimage.sobel(img_array, axis=1)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_density = np.mean(edge_magnitude) / 255 * 100  # Normalize to 0-100
    
    # Pore structure (contrast in local regions)
    # Simple approximation: high contrast indicates visible pores
    pore_contrast = contrast
    
    # Aeration (inverse of density)
    # Simple: higher brightness = more air pockets
    aeration = brightness / 255 * 100
    
    # Cell structure (edge density as proxy)
    cell_structure = edge_density
    
    # Interpretations
    if brightness > 200:
        crust_note = "Very light crumb — may suggest under-baking or a pale, low-sugar batter."
        bake_note = "Check bake time and temperature."
    elif brightness > 170:
        crust_note = "Light crumb color — potentially over-baked or high-sugar formulation."
        bake_note = "Potential over-bake or high-sugar formulation."
    elif brightness < 120:
        crust_note = "Dark crumb — may indicate over-baking or high Maillard reaction."
        bake_note = "Reduce bake time or temperature."
    else:
        crust_note = "Crumb color appears normal — mid-range browning."
        bake_note = "Bake level appears standard."
    
    if contrast > 55:
        pore_note = "High contrast indicates visible, non-uniform pores — open, irregular crumb structure."
        airy_note = "Structure appears porous / airy."
    elif contrast < 25:
        pore_note = "Low contrast indicates a dense, tight crumb with minimal visible pores."
        airy_note = "Structure appears dense / compact."
    else:
        pore_note = "Moderate contrast — medium crumb porosity."
        airy_note = "Structure appears intermediate."
    
    if edge_density > 15:
        texture_note = "High edge density — many crumb cell walls visible, suggesting a fine, tight cell structure."
    elif edge_density < 6:
        texture_note = "Low edge density — large or ill-defined crumb cells, consistent with a coarse or open structure."
    else:
        texture_note = "Moderate edge density — balanced crumb cell definition."
    
    return {
        "Brightness": round(brightness, 1),
        "Contrast": round(contrast, 1),
        "Edge Density": round(edge_density, 2),
        "Crumb Color": crust_note,
        "Bake Level": bake_note,
        "Pore Structure": pore_note,
        "Aeration": airy_note,
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
                st.dataframe(csv_df, width='stretch')

                col_map = {
                    "Hardness": "Hardness",
                    "Resilience": "Resilience",
                    "Cohesiveness": "Cohesiveness",
                    "Springiness": "Springiness",
                    "Chewiness": "Chewiness",
                    "Max Shear Force (N)": "MaxShear",
                }

                csv_input = {}
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
                    csv_input[sname] = pdata
                    sample_n_table[sname] = n_info

                input_data = csv_input
                csv_loaded = True

                # Show per-sample n table
                st.markdown("<p class='section-title' style='margin-top:16px;'>Replicate counts detected</p>", unsafe_allow_html=True)
                n_display = pd.DataFrame(sample_n_table).T
                n_display.index.name = "Sample"
                st.dataframe(n_display, width='stretch')

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
        st.dataframe(summary_df, width='stretch', hide_index=True)
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
            low_power_warning = False
            if samp in stat_results:
                for param, test in stat_results[samp].items():
                    pval = test.get("p")
                    method = test.get("method", "")
                    n_c = test.get("n_ctrl", 0)
                    n_s = test.get("n_samp", 0)
                    if pval is not None and pval < SIG_ALPHA:
                        sig_params.append((param, pval, method, n_c, n_s))
                    if n_c < 3 or n_s < 3:
                        low_power_warning = True

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

                st.markdown("**Cake Structure Analysis (MaxShear):**")
                structure_status = ""
                if ms_pct is not None:
                    if ms_pct < 0:
                        # Only flag as CRUMBLY if statistically significant
                        if ms_sig:
                            structure_status = f"⚠️ **Lower MaxShear ({ms_pct:.1f}%) - CRUMBLY structure**"
                        else:
                            structure_status = f"✅ **Lower MaxShear ({ms_pct:.1f}%) - Not significantly different**"
                    elif ms_pct >= 15:
                        structure_status = f"✅ **Higher MaxShear ({ms_pct:.1f}%) - Stronger structure than control**"
                    else:
                        structure_status = f"✅ **MaxShear similar to control ({ms_pct:.1f}%)**"

                if ms_p_val is not None:
                    structure_status += f" | p-value: {ms_p_val:.3f}"
                    if ms_sig:
                        structure_status += " (significant)"
                    else:
                        structure_status += " (not significant)"

                st.markdown(structure_status)

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

            r_col1, r_col2, r_col3 = st.columns(3)

            # Column 1: Key differences
            with r_col1:
                st.markdown("<p class='section-title'>Key Differences</p>", unsafe_allow_html=True)
                if shown_pct:
                    for param, pct in shown_pct.items():
                        tag_class = "tag-major" if abs(pct) >= _major else "tag-minor"
                        st.markdown(f"<div class='{tag_class}'>{param}: {pct:+.1f}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='sub-text'>All parameters within ±10% of control</div>", unsafe_allow_html=True)

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
                if low_power_warning:
                    st.markdown("<div class='sub-text' style='color:#ff9f0a;margin-top:8px;'>⚠️ Low statistical power (n<3)</div>", unsafe_allow_html=True)

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
            st.dataframe(combined.round(4), width='stretch')

    # ─────────────────────────────────────────────
    # TAB 3 — VISUALIZATIONS
    # ─────────────────────────────────────────────
    with tab_viz:

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Grouped Bar Chart</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Mean values with standard deviation error bars. All parameters on a common axis for direct visual comparison.</p>", unsafe_allow_html=True)
        fig_bar = plot_bar_chart(means_df, sd_df, sample_colors)
        st.plotly_chart(fig_bar, width='stretch')
        st.markdown("**Key Insights:**")
        # Generate sample-specific insights
        insights = []
        for samp in _samples:
            if samp == _control:
                insights.append(f"- **{_control}** (control): Reference sample with baseline texture properties")
            else:
                # Compare to control
                higher_params = []
                lower_params = []
                maxshear_diff = None
                for param in ALL_PARAMS:
                    if samp in means_df.index and _control in means_df.index:
                        samp_val = means_df.loc[samp, param]
                        ctrl_val = means_df.loc[_control, param]
                        pct_diff_val = pct_diff(samp_val, ctrl_val)
                        if pct_diff_val is not None:
                            # Always include MaxShear since it's critical for structure; use 10% threshold for other params
                            threshold = 0 if param == "MaxShear" else 10
                            if abs(pct_diff_val) >= threshold or (param == "MaxShear" and pct_diff_val != 0):
                                if pct_diff_val > 0:
                                    higher_params.append(f"{param.lower()} (+{pct_diff_val:.0f}%)")
                                else:
                                    lower_params.append(f"{param.lower()} ({pct_diff_val:.0f}%)")
                            if param == "MaxShear":
                                maxshear_diff = pct_diff_val

                insight = f"- **{samp}**: "
                if higher_params and lower_params:
                    insight += f"Higher in {', '.join(higher_params[:2])}; lower in {', '.join(lower_params[:2])}"
                elif higher_params:
                    insight += f"Higher in {', '.join(higher_params[:3])}"
                elif lower_params:
                    insight += f"Lower in {', '.join(lower_params[:3])}"
                else:
                    insight += "Similar texture profile to control"

                # Special emphasis on MaxShear for cake structure - report if significantly lower OR meaningfully lower
                if maxshear_diff is not None and maxshear_diff < 0:  # Only if lower
                    if samp in stat_results and "MaxShear" in stat_results[samp]:
                        ms_test = stat_results[samp]["MaxShear"]
                        ms_sig_insight = ms_test.get("significant", False)
                        if ms_sig_insight:
                            insight += f" - **CRUMBLY: Significantly lower MaxShear ({maxshear_diff:.0f}%) deteriorates structural integrity**"
                        elif abs(maxshear_diff) >= 10:  # Only report lower if meaningful
                            insight += f" - **Lower MaxShear ({maxshear_diff:.0f}%) - may affect structure**"
                    elif abs(maxshear_diff) >= 10:
                        insight += f" - **Lower MaxShear ({maxshear_diff:.0f}%) - may affect structure**"
                elif maxshear_diff is not None and maxshear_diff > 0 and abs(maxshear_diff) >= 5:
                    insight += f" - **Higher MaxShear ({maxshear_diff:.0f}%) suggests stronger structure**"

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
                    st.plotly_chart(fig_pca, width='stretch')
                    st.markdown("**Key Insights:**")
                    # PCA sample insights
                    pca_insights = []

                    # Find most similar and different samples to control
                    if _control in z_df.index:
                        ctrl_pca = z_df.loc[_control, ['PC1', 'PC2']].values
                        distances = {}
                        for samp in _samples:
                            if samp != _control and samp in z_df.index:
                                samp_pca = z_df.loc[samp, ['PC1', 'PC2']].values
                                distances[samp] = np.linalg.norm(ctrl_pca - samp_pca)

                        if distances:
                            most_similar = min(distances, key=distances.get)
                            most_different = max(distances, key=distances.get)
                            pca_insights.append(f"- **{most_similar}** is closest to {_control}, so its overall texture profile is most comparable to the control.")
                            if most_different != most_similar:
                                pca_insights.append(f"- **{most_different}** is furthest from {_control}, indicating the most distinct texture profile among the samples.")

                    # Identify sample groupings that share texture behavior
                    if len(_samples) > 3:
                        from sklearn.cluster import KMeans
                        try:
                            pca_coords = z_df.loc[_samples, ['PC1', 'PC2']].values
                            kmeans = KMeans(n_clusters=min(3, len(_samples)), random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(pca_coords)

                            cluster_groups = {}
                            for i, samp in enumerate(_samples):
                                cluster_id = clusters[i]
                                cluster_groups.setdefault(cluster_id, []).append(samp)

                            for cluster_id, samples_in_cluster in cluster_groups.items():
                                if len(samples_in_cluster) > 1:
                                    pca_insights.append(f"- **{', '.join(samples_in_cluster)}** form a similar texture cluster.")
                        except Exception:
                            pass

                    if not pca_insights:
                        pca_insights.append("- The samples occupy distinct positions in texture space, showing how each one compares to the control.")

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
                st.plotly_chart(fig_radar, width='stretch')
                st.markdown("**Key Insights:**")
                # Radar chart sample insights
                radar_insights = []

                # Find samples with the largest deviations from the average profile
                for samp in _samples:
                    if samp != _control and samp in z_df.index:
                        profile = z_df.loc[samp, ALL_PARAMS].values
                        avg_profile = z_df.loc[_samples, ALL_PARAMS].mean().values
                        deviation = np.abs(profile - avg_profile)
                        max_dev_param = ALL_PARAMS[np.argmax(deviation)]
                        max_deviation = deviation.max()

                        if max_deviation > 0.5:
                            direction = "higher" if profile[np.argmax(deviation)] > avg_profile[np.argmax(deviation)] else "lower"
                            radar_insights.append(f"- **{samp}** differs most on {max_dev_param.lower()} ({direction}), showing its strongest texture contrast with the sample set.")

                # Find most balanced vs most extreme profiles
                if len(_samples) > 2:
                    profile_variances = {}
                    for samp in _samples:
                        if samp in z_df.index:
                            profile_variances[samp] = np.var(z_df.loc[samp, ALL_PARAMS].values)

                    if profile_variances:
                        most_balanced = min(profile_variances, key=profile_variances.get)
                        most_extreme = max(profile_variances, key=profile_variances.get)

                        if most_balanced != _control:
                            radar_insights.append(f"- **{most_balanced}** has the most balanced texture profile, indicating consistency across parameters.")
                        if most_extreme != _control and most_extreme != most_balanced:
                            radar_insights.append(f"- **{most_extreme}** has the most variable profile, suggesting a more extreme texture behavior.")

                if not radar_insights:
                    radar_insights.append("- Samples show subtle differences in standardized texture parameters, with no sample dominating any single attribute.")

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
        st.plotly_chart(fig_heat, width='stretch')
        
        st.markdown("**Key Insights:**")
        # Generate heatmap insights
        heatmap_insights = []
        
        # Find which parameters show the most variation
        param_variation = {}
        for param in ALL_PARAMS:
            param_vals = [pct_hm_df.loc[samp, param] for samp in pct_hm_df.index if samp in pct_hm_df.columns]
            param_variation[param] = np.std(param_vals) if param_vals else 0
        
        most_variable_param = max(param_variation, key=param_variation.get)
        least_variable_param = min(param_variation, key=param_variation.get)
        
        heatmap_insights.append(f"- **{most_variable_param}** shows the most variation across samples (most colored cells), indicating it's the key differentiator.")
        heatmap_insights.append(f"- **{least_variable_param}** is most consistent across samples (lighter cells).")
        
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

        # Parameter Correlation Analysis
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Parameter Correlations</p>", unsafe_allow_html=True)
        st.markdown("<p class='sub-text'>Pearson correlations between texture parameters. Only reports correlations with R² ≥ 0.25 and p < 0.05 (confident relationships).</p>", unsafe_allow_html=True)
        
        corr_results = compute_parameter_correlations(means_df, min_r_squared=0.25, sig_alpha=0.05)
        corr_matrix = corr_results['matrix']
        significant_corrs = corr_results['correlations']
        
        # Plot correlation heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[[0, "#ff375f"], [0.5, "#ffffff"], [1, "#0071e3"]],
            zmid=0,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        fig_corr.update_layout(
            title=dict(text="Texture Parameter Correlation Matrix (Pearson)", font=dict(size=16, family="Inter", color="#1d1d1f"), x=0.0),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(title="Parameters", tickfont=dict(size=11)),
            yaxis=dict(title="Parameters", tickfont=dict(size=11)),
            font=dict(family="Inter", color="#1d1d1f"),
            height=500,
            margin=dict(l=100, r=60, t=80, b=100),
        )
        st.plotly_chart(fig_corr, width='stretch')
        
        # Report significant correlations
        st.markdown("**Significant Correlations (p < 0.05, R² ≥ 0.25):**")
        
        if significant_corrs:
            for corr in significant_corrs:
                direction_text = "increases with" if corr['direction'] == 'positive' else "decreases with"
                r_pct = corr['r_squared'] * 100
                st.markdown(
                    f"- **{corr['param1']} {direction_text} {corr['param2']}**: "
                    f"r = {corr['r']}, R² = {corr['r_squared']} ({r_pct:.0f}% variance explained), p = {corr['p_value']}"
                )
            
            # Add interpretation
            st.markdown("**Interpretation:**")
            
            # Check for hardness-related correlations
            hardness_corrs = [c for c in significant_corrs if 'Hardness' in c['param1'] or 'Hardness' in c['param2']]
            if hardness_corrs:
                st.markdown("- **Hardness correlations**: Changes in cake hardness strongly affect..." + 
                           ", ".join([c['param2'] if c['param1'] == 'Hardness' else c['param1'] for c in hardness_corrs[:2]]))
            
            # Check for MaxShear correlations (structure-related)
            maxshear_corrs = [c for c in significant_corrs if 'MaxShear' in c['param1'] or 'MaxShear' in c['param2']]
            if maxshear_corrs:
                st.markdown("- **MaxShear (structure) correlations**: Cake structural strength (MaxShear) is connected to " +
                           ", ".join([c['param2'] if c['param1'] == 'MaxShear' else c['param1'] for c in maxshear_corrs[:2]]) +
                           ". This suggests formulation changes affect multiple structural properties simultaneously.")
            
            # Check for other clusters
            cohesion_corrs = [c for c in significant_corrs if 'Cohesiveness' in c['param1'] or 'Cohesiveness' in c['param2']]
            if cohesion_corrs:
                st.markdown("- **Cohesiveness correlations**: Cake cohesion is strongly linked to " +
                           ", ".join([c['param2'] if c['param1'] == 'Cohesiveness' else c['param1'] for c in cohesion_corrs[:2]]) +
                           ". These parameters may be modified by similar formulation factors.")
        else:
            st.markdown("_No significant correlations detected with confidence threshold R² ≥ 0.25 (p < 0.05). Parameters vary independently, suggesting diverse formulation effects._")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TAB 4 — CRUMB IMAGE ANALYSIS
    # ─────────────────────────────────────────────
    with tab_image:

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>Cake Crumb Image Analysis</p>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sub-text'>Upload a photo of your cake crumb to analyze texture characteristics. The analysis provides quantitative metrics and qualitative interpretations of crumb structure, aeration, and bake level.</p>",
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader("Upload cake crumb image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_image:
            try:
                image = Image.open(uploaded_image)
                
                # Display image
                col_img1, col_img2 = st.columns([1, 2])
                with col_img1:
                    st.image(image, width='stretch', caption="Uploaded Image")
                
                with col_img2:
                    # Analyze
                    results = analyze_cake_image(image)
                    
                    st.markdown("<p class='section-title' style='margin-top:0;'>Quantitative Metrics</p>", unsafe_allow_html=True)
                    metrics_df = pd.DataFrame({
                        "Parameter": list(results.keys())[:3],
                        "Value": [results[k] for k in list(results.keys())[:3]]
                    })
                    st.dataframe(metrics_df, width='stretch', hide_index=True)
                    
                    st.markdown("<p class='section-title'>Qualitative Analysis</p>", unsafe_allow_html=True)
                    for key, value in list(results.items())[3:]:
                        st.markdown(f"**{key}:** {value}")
                        
            except Exception as e:
                st.error(f"Could not analyze image: {e}")
        else:
            st.markdown(
                "<div style='text-align:center;padding:60px 0;color:#aeaeb2;font-size:14px;'>"
                "Upload a cake crumb image above to analyze texture characteristics."
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