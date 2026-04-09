import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import feature, measure, morphology
from skimage.feature import graycomatrix, graycoprops


@dataclass
class AnalyzerConfig:
    # Preprocessing
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    debug_visualization: bool = False

    # Pore filtering
    min_pore_area_px: int = 12

    # Thin region threshold (wall thickness in px)
    thin_thickness_px: float = 2.5

    # Fracture detection
    canny_low: int = 50
    canny_high: int = 150
    pore_boundary_dilate_px: int = 1

    # Texture features
    glcm_levels: int = 32
    glcm_distances: Tuple[int, ...] = (1, 2)
    glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    # Composite scoring weights (must sum to 1.0) — includes new structural metrics
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "wall_thickness": 0.12,
            "connectivity": 0.12,
            "homogeneity": 0.08,
            "pore_cv": 0.10,
            "thin_fraction": 0.05,
            "fracture_index": 0.10,
            "excess_porosity": 0.03,
            "circularity": 0.12,
            "porosity_uniformity": 0.08,
            "clustering": 0.12,
            "wall_variance": 0.08,
        }
    )

    # Excessive porosity threshold (fraction, not percent)
    porosity_high_threshold: float = 0.35

    # Classification thresholds
    strong_threshold: float = 70.0
    moderate_threshold: float = 40.0

    # Classification thresholds aligned with specification rules
    cell_size_cv_weak: float = 1.50       # CV above this → weak (spec: CV > 1.5)
    circularity_weak: float = 0.75        # mean circularity below this → weak (spec: < 0.75)
    circularity_strong: float = 0.85      # mean circularity above this → strong (spec: > 0.85)
    clustering_weak: float = 1.50         # NNI-inverted: >1 = clustered = bad
    porosity_uniformity_strong: float = 0.05  # std below this = uniform = good
    wall_variance_strong: float = 4.0     # variance below this = consistent = good
    connectivity_strong: float = 4.0      # connectivity ratio above this → strong (spec: > 4)
    homogeneity_strong: float = 0.90      # homogeneity above this → strong (spec: > 0.9)
    homogeneity_weak: float = 0.80        # homogeneity below this → weak (spec: < 0.8)
    porosity_weak: float = 0.05           # porosity above this → weak (spec: > 0.05)
    porosity_strong: float = 0.01         # porosity below this → strong/dense (spec: < 0.01)
    mean_pore_size_large: float = 150.0   # mean pore size above this → weak tendency
    fracture_index_weak: float = 0.03     # fracture index above this → weak
    wall_thickness_strong: float = 4.0    # mean wall thickness above this → strong
    wall_thickness_weak: float = 2.0      # mean wall thickness below this → weak

    # Minimum reliable-variable fraction required to produce a classification
    classification_min_reliable_fraction: float = 0.70

    # Illumination normalization
    illum_blur_ksize: int = 61            # large Gaussian kernel for background estimation
    median_ksize: int = 3                 # median denoising kernel after CLAHE

    # Optional ImageJ integration
    try_imagej: bool = False


class CrumbAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._validate_weights()
        self._ij = None
        if self.config.try_imagej:
            self._try_init_imagej()

    def _validate_weights(self) -> None:
        total = sum(self.config.score_weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"score_weights must sum to 1.0, got {total}")

    def _try_init_imagej(self) -> None:
        try:
            import imagej  # pyimagej

            self._ij = imagej.init(mode="headless")
            print("ImageJ initialized (optional).")
        except Exception as exc:
            print(f"ImageJ not available, continuing without it: {exc}")
            self._ij = None

    def _collect_images(self, input_path: Path) -> List[Path]:
        if input_path.is_file():
            return [input_path]
        if input_path.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
            return files
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _preprocess(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess with mandatory illumination correction before any feature extraction.

        Pipeline:
          1. Grayscale conversion
          2. Background estimation via large Gaussian blur
          3. Illumination normalization: corrected = gray / background * 128
          4. CLAHE contrast enhancement
          5. Median denoising
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if self._ij is not None:
            # Optional ImageJ hook point.
            pass

        # ── Step 2: Estimate slow-varying illumination background ──
        k = self.config.illum_blur_ksize
        # Kernel must be odd
        if k % 2 == 0:
            k += 1
        background = cv2.GaussianBlur(gray.astype(np.float32), (k, k), 0)
        # Avoid divide-by-zero; clip background floor to 1
        background = np.clip(background, 1.0, None)

        # ── Step 3: Normalize illumination ──
        corrected = (gray.astype(np.float32) / background * 128.0)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # ── Step 4: CLAHE on the illumination-corrected image ──
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )
        norm = clahe.apply(corrected)

        # ── Step 5: Median denoising ──
        mk = self.config.median_ksize
        if mk % 2 == 0:
            mk += 1
        blur = cv2.medianBlur(norm, mk)

        return gray, norm, blur

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        labeled = measure.label(mask, connectivity=2)
        props = measure.regionprops(labeled)
        if not props:
            return np.zeros_like(mask, dtype=bool)
        largest = max(props, key=lambda p: p.area)
        return labeled == largest.label

    def _segment_crumb(self, blur_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment crumb using adaptive thresholding (not global) to handle residual
        local intensity variation after illumination correction."""
        h, w = blur_gray.shape
        # Block size must be odd and reasonably large relative to image
        block = max(11, (min(h, w) // 20) | 1)  # bitwise OR 1 forces odd
        if block % 2 == 0:
            block += 1

        # Adaptive Gaussian threshold — pores tend to be darker than surrounding crumb
        th_adapt = cv2.adaptiveThreshold(
            blur_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 2
        )
        b1 = th_adapt > 0
        b2 = ~b1

        l1 = self._largest_component(morphology.remove_small_objects(b1, 128))
        l2 = self._largest_component(morphology.remove_small_objects(b2, 128))

        crumb = l1 if l1.sum() >= l2.sum() else l2
        crumb = morphology.binary_closing(crumb, morphology.disk(2))
        crumb = morphology.remove_small_objects(crumb, 256)
        crumb = self._largest_component(crumb)

        # ROI is filled crumb body (includes pores as potential voids).
        roi = ndi.binary_fill_holes(crumb)

        # Pores = air regions inside ROI but not in crumb.
        pores = roi & (~crumb)
        pores = morphology.remove_small_objects(pores, self.config.min_pore_area_px)

        return crumb.astype(bool), pores.astype(bool), roi.astype(bool)

    def _validate_segmentation(
        self, gray_original: np.ndarray, norm: np.ndarray, pores: np.ndarray, roi: np.ndarray
    ) -> Tuple[str, List[str]]:
        """Validate preprocessing and segmentation quality.

        Returns:
            confidence: 'HIGH' | 'MEDIUM' | 'LOW'
            warnings: list of human-readable warning strings
        """
        warnings_list: List[str] = []
        penalty = 0

        # ── Check 1: Residual lighting gradient in illumination-corrected image ──
        # Tile the corrected image and measure std of tile means; high spread = gradient remains
        h, w = norm.shape
        n_tiles = 4
        th = max(1, h // n_tiles)
        tw = max(1, w // n_tiles)
        tile_means = []
        for ti in range(n_tiles):
            for tj in range(n_tiles):
                tile = norm[ti * th:(ti + 1) * th, tj * tw:(tj + 1) * tw]
                if tile.size > 0:
                    tile_means.append(float(tile.mean()))
        if len(tile_means) > 1:
            gradient_spread = float(np.std(tile_means))
            if gradient_spread > 30:
                warnings_list.append(
                    f"Lighting gradient likely present after correction (tile mean std={gradient_spread:.1f}). "
                    "Illumination-sensitive metrics may be unreliable."
                )
                penalty += 2
            elif gradient_spread > 15:
                warnings_list.append(
                    f"Mild residual lighting variation detected (tile mean std={gradient_spread:.1f})."
                )
                penalty += 1

        # ── Check 2: Pore coverage sanity ──
        roi_area = int(roi.sum())
        pore_area = int(pores.sum())
        porosity = pore_area / roi_area if roi_area > 0 else 0.0
        if porosity < 0.005:
            warnings_list.append(
                "Very few pores detected (<0.5% porosity). Segmentation may have failed or image has no visible air cells."
            )
            penalty += 2
        elif porosity > 0.80:
            warnings_list.append(
                f"Extremely high porosity ({porosity:.1%}). Segmentation polarity may be inverted."
            )
            penalty += 2

        # ── Check 3: Pore count — too few to compute reliable statistics ──
        labeled = measure.label(pores, connectivity=2)
        n_pores = labeled.max()
        if n_pores < 5:
            warnings_list.append(
                f"Only {n_pores} pore(s) detected. Statistical metrics (CV, clustering, uniformity) are not reliable."
            )
            penalty += 2
        elif n_pores < 15:
            warnings_list.append(
                f"Low pore count ({n_pores}). Statistical metrics have reduced reliability."
            )
            penalty += 1

        # ── Check 4: ROI coverage — if ROI is very small, likely segmentation failure ──
        total_pixels = gray_original.size
        roi_fraction = roi_area / total_pixels if total_pixels > 0 else 0.0
        if roi_fraction < 0.05:
            warnings_list.append(
                f"ROI covers only {roi_fraction:.1%} of image. Segmentation may have failed."
            )
            penalty += 2

        # ── Assign confidence ──
        if penalty == 0:
            confidence = "HIGH"
        elif penalty <= 1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return confidence, warnings_list

    def _pore_features(self, pores: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
        roi_area = float(roi.sum())
        air_area = float(pores.sum())
        porosity = air_area / roi_area if roi_area > 0 else np.nan

        labeled = measure.label(pores, connectivity=2)
        labeled_props = measure.regionprops(labeled)
        areas = np.array([r.area for r in labeled_props], dtype=float)

        if areas.size == 0:
            return {
                "porosity": porosity,
                "mean_pore_size": 0.0,
                "std_pore_size": 0.0,
                "pore_cv": 0.0,
                "circularity": 0.0,
            }

        mean_a = float(np.mean(areas))
        std_a = float(np.std(areas))
        cv_a = float(std_a / mean_a) if mean_a > 1e-9 else 0.0
        # Mean circularity: 4π·area/perimeter² (1.0 = perfect circle)
        circularity_vals = []
        for r in labeled_props:
            if r.perimeter > 0:
                c = (4.0 * np.pi * r.area) / (r.perimeter ** 2)
                circularity_vals.append(float(min(c, 1.0)))
        mean_circ = float(np.mean(circularity_vals)) if circularity_vals else 0.0

        return {
            "porosity": porosity,
            "mean_pore_size": mean_a,
            "std_pore_size": std_a,
            "pore_cv": cv_a,
            "circularity": mean_circ,
        }

    def _wall_features(self, crumb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        dist = ndi.distance_transform_edt(crumb)
        local_thickness = 2.0 * dist
        crumb_pixels = crumb.sum()
        mean_thick = float(local_thickness[crumb].mean()) if crumb_pixels > 0 else np.nan

        thin_mask = crumb & (local_thickness < self.config.thin_thickness_px)
        thin_frac = float(thin_mask.sum() / crumb_pixels) if crumb_pixels > 0 else np.nan

        wall_thick_var = float(np.var(local_thickness[crumb])) if crumb_pixels > 0 else np.nan

        return {
            "mean_wall_thickness": mean_thick,
            "thin_region_fraction": thin_frac,
            "wall_thickness_var": wall_thick_var,
        }, local_thickness, thin_mask
    def _spatial_features(self, pores: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
        """Porosity uniformity (std of tile-level porosities) and Clark-Evans clustering index."""
        ys, xs = np.where(roi)
        if ys.size == 0:
            return {"porosity_uniformity": 0.0, "clustering_index": 1.0}

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        n_tiles = 4  # 4×4 grid → 16 candidate tiles

        tile_h = max(1, (y1 - y0) // n_tiles)
        tile_w = max(1, (x1 - x0) // n_tiles)

        local_pors = []
        for ti in range(n_tiles):
            for tj in range(n_tiles):
                sy = y0 + ti * tile_h
                ey = y0 + (ti + 1) * tile_h
                sx = x0 + tj * tile_w
                ex = x0 + (tj + 1) * tile_w
                tile_roi = roi[sy:ey, sx:ex]
                tile_pores = pores[sy:ey, sx:ex]
                roi_px = int(tile_roi.sum())
                if roi_px > 10:
                    local_pors.append(float(tile_pores.sum()) / roi_px)

        porosity_std = float(np.std(local_pors)) if len(local_pors) > 1 else 0.0

        # Clark-Evans NNI: mean observed NN-dist / expected NN-dist under CSR
        # Invert so that NNI < 1 (clustered) → high clustering_index value (bad)
        labeled_p = measure.label(pores, connectivity=2)
        props_p = measure.regionprops(labeled_p)
        n_pores = len(props_p)

        if n_pores < 2:
            clustering_index = 1.0
        else:
            centroids = np.array([p.centroid for p in props_p], dtype=float)
            roi_area = float(roi.sum())
            density = n_pores / roi_area
            tree = cKDTree(centroids)
            dists, _ = tree.query(centroids, k=2)  # k=2: first hit is self
            mean_nn = float(np.mean(dists[:, 1]))
            expected_nn = 0.5 / np.sqrt(density) if density > 0 else np.inf
            nni = mean_nn / expected_nn if expected_nn > 0 else 1.0
            # Invert: clustered (NNI<1) → clustering_index > 1
            clustering_index = float(1.0 / max(float(nni), 0.1))

        return {
            "porosity_uniformity": porosity_std,
            "clustering_index": clustering_index,
        }

    def _connectivity_features(self, crumb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        skel = morphology.skeletonize(crumb)

        # Neighbor count on skeleton (8-neighborhood)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        conv = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
        # 11 => one neighbor, >=13 => at least three neighbors
        endpoints = skel & (conv == 11)
        branchpoints = skel & (conv >= 13)

        n_end = int(endpoints.sum())
        n_branch = int(branchpoints.sum())
        conn_ratio = float(n_branch / max(n_end, 1))

        n_regions = int(measure.label(crumb, connectivity=2).max())

        return {
            "endpoints": n_end,
            "branchpoints": n_branch,
            "connectivity_ratio": conn_ratio,
            "disconnected_regions": n_regions,
        }, skel, endpoints, branchpoints

    def _fracture_features(self, blur_gray: np.ndarray, crumb: np.ndarray, pores: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        edges = cv2.Canny(blur_gray, self.config.canny_low, self.config.canny_high) > 0

        pore_boundary = morphology.binary_dilation(
            morphology.binary_dilation(pores) ^ pores,
            morphology.disk(self.config.pore_boundary_dilate_px),
        )

        internal_edges = edges & crumb & (~pore_boundary)
        crumb_pixels = max(int(crumb.sum()), 1)
        fracture_index = float(internal_edges.sum() / crumb_pixels)

        return {"fracture_index": fracture_index}, internal_edges

    def _texture_features(self, gray: np.ndarray, roi: np.ndarray, crumb: np.ndarray) -> Dict[str, float]:
        ys, xs = np.where(roi)
        if ys.size == 0:
            return {
                "glcm_contrast": np.nan,
                "glcm_homogeneity": np.nan,
                "glcm_entropy": np.nan,
                "fractal_dimension": np.nan,
            }

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = gray[y0:y1, x0:x1]
        crop_roi = roi[y0:y1, x0:x1]

        # Quantize for GLCM.
        levels = self.config.glcm_levels
        q = np.floor((crop.astype(np.float32) / 256.0) * levels).astype(np.uint8)
        q[q == levels] = levels - 1

        # Mask out non-ROI by assigning zero; still interpretable for homogeneous backgrounds.
        q_masked = q.copy()
        q_masked[~crop_roi] = 0

        glcm = graycomatrix(
            q_masked,
            distances=list(self.config.glcm_distances),
            angles=list(self.config.glcm_angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )

        contrast = float(graycoprops(glcm, "contrast").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())

        vals = crop[crop_roi]
        hist, _ = np.histogram(vals, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else np.nan

        fractal_dim = self._fractal_dimension(crumb)

        return {
            "glcm_contrast": contrast,
            "glcm_homogeneity": homogeneity,
            "glcm_entropy": entropy,
            "fractal_dimension": fractal_dim,
        }

    def _fractal_dimension(self, binary: np.ndarray) -> float:
        # Box-counting dimension for binary structure complexity.
        z = binary.astype(bool)
        if z.sum() == 0:
            return np.nan

        p = min(z.shape)
        n = 2 ** int(np.floor(np.log2(max(2, p))))
        z = z[:n, :n]

        sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
        counts = []

        for size in sizes:
            h = n // size
            reshaped = z.reshape(h, size, h, size).transpose(0, 2, 1, 3)
            blocks = reshaped.any(axis=(2, 3))
            counts.append(blocks.sum())

        counts = np.array(counts, dtype=float)
        valid = counts > 0
        if valid.sum() < 2:
            return np.nan

        coeffs = np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)
        return float(coeffs[0])

    def _save_visuals(
        self,
        sample_name: str,
        out_dir: Path,
        original: np.ndarray,
        crumb: np.ndarray,
        skeleton: np.ndarray,
        thin_mask: np.ndarray,
        fractures: np.ndarray,
    ) -> None:
        sample_dir = out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        binary_vis = (crumb.astype(np.uint8) * 255)

        skel_overlay = original.copy()
        skel_overlay[skeleton] = [255, 255, 0]

        thin_overlay = original.copy()
        thin_overlay[thin_mask] = [255, 0, 0]

        fracture_overlay = original.copy()
        fracture_overlay[fractures] = [0, 0, 255]

        cv2.imwrite(str(sample_dir / "original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "binary_segmentation.png"), binary_vis)
        cv2.imwrite(str(sample_dir / "skeleton_overlay.png"), cv2.cvtColor(skel_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "thin_regions_red.png"), cv2.cvtColor(thin_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sample_dir / "fractures_blue.png"), cv2.cvtColor(fracture_overlay, cv2.COLOR_RGB2BGR))

        # A clean panel for quick visual review.
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        ax[0].imshow(original)
        ax[0].set_title("Original")
        ax[1].imshow(binary_vis, cmap="gray")
        ax[1].set_title("Binary Segmentation")
        ax[2].imshow(skel_overlay)
        ax[2].set_title("Skeleton Overlay")
        ax[3].imshow(thin_overlay)
        ax[3].set_title("Thin Regions (Red)")
        ax[4].imshow(fracture_overlay)
        ax[4].set_title("Fractures (Blue)")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(sample_dir / "visual_summary.png", dpi=150)
        plt.close(fig)

        if self.config.debug_visualization:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(binary_vis, cmap="gray")
            ax.set_title(f"Debug Binary: {sample_name}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(sample_dir / "debug_binary.png", dpi=150)
            plt.close(fig)

    def _compute_scores(self, df: pd.DataFrame, seg_confidence: Optional[str] = None) -> pd.DataFrame:
        """Score each sample using absolute-threshold rule voting per specification.

        Each rule produces +1 (strong indicator) or -1 (weak indicator).
        Confidence is tracked per metric; LOW-confidence metrics are excluded from voting
        and marked 'Not Reliable' in the confidence column.

        Classification only produced when ≥70% of rules are HIGH/MEDIUM confidence.
        """
        cfg = self.config
        out = df.copy()

        # LOW segmentation confidence degrades all metrics that depend on pore/wall detection
        seg_low = (seg_confidence == "LOW") if seg_confidence else False

        def _score_row(row: pd.Series) -> pd.Series:  # noqa: C901
            votes = 0
            max_votes = 0
            reliable_count = 0
            total_rules = 0
            conf_map: Dict[str, str] = {}

            def _vote(metric_name: str, value, strong_fn, weak_fn,
                      base_conf: str = "HIGH", nan_unreliable: bool = True):
                nonlocal votes, max_votes, reliable_count, total_rules
                total_rules += 1
                if nan_unreliable and (value is None or (isinstance(value, float) and np.isnan(value))):
                    conf_map[metric_name] = "NOT RELIABLE"
                    return
                if seg_low and metric_name not in ("Homogeneity", "Fracture index"):
                    conf_map[metric_name] = "LOW"
                    return
                conf = base_conf
                conf_map[metric_name] = conf
                if conf in ("HIGH", "MEDIUM"):
                    reliable_count += 1
                    max_votes += 1
                    if strong_fn(value):
                        votes += 1
                    elif weak_fn(value):
                        votes -= 1

            # 1. Porosity (AERATION)
            _vote("Porosity", row.get("Porosity"),
                  strong_fn=lambda v: v < cfg.porosity_strong,
                  weak_fn=lambda v: v > cfg.porosity_weak)

            # 2. Mean pore size (AERATION)
            _vote("Mean pore size", row.get("Mean pore size"),
                  strong_fn=lambda v: v < 80,
                  weak_fn=lambda v: v > cfg.mean_pore_size_large)

            # 3. Cell size CV (UNIFORMITY)
            pore_cv = row.get("Pore CV")
            cv_conf = "MEDIUM" if (pore_cv is not None and not np.isnan(pore_cv) and pore_cv < 0.3) else "HIGH"
            _vote("Cell size CV", pore_cv,
                  strong_fn=lambda v: v < 0.5,
                  weak_fn=lambda v: v > cfg.cell_size_cv_weak,
                  base_conf=cv_conf)

            # 4. Porosity uniformity (UNIFORMITY)
            _vote("Porosity uniformity", row.get("Porosity uniformity"),
                  strong_fn=lambda v: v < cfg.porosity_uniformity_strong,
                  weak_fn=lambda v: v > 0.10)

            # 5. Circularity (GEOMETRY)
            _vote("Circularity", row.get("Circularity"),
                  strong_fn=lambda v: v > cfg.circularity_strong,
                  weak_fn=lambda v: v < cfg.circularity_weak)

            # 6. Mean wall thickness (STRUCTURE)
            _vote("Mean wall thickness", row.get("Mean wall thickness"),
                  strong_fn=lambda v: v > cfg.wall_thickness_strong,
                  weak_fn=lambda v: v < cfg.wall_thickness_weak)

            # 7. Wall thickness variance (STRUCTURE)
            wtv = row.get("Wall thickness variance")
            # Thick + uniform walls → strong; thick + highly variable → moderate (no vote)
            mwt = row.get("Mean wall thickness", 0.0) or 0.0
            def _wall_var_strong(v):
                return v < cfg.wall_variance_strong and mwt > cfg.wall_thickness_strong
            def _wall_var_weak(v):
                return v >= cfg.wall_variance_strong
            _vote("Wall thickness variance", wtv,
                  strong_fn=_wall_var_strong,
                  weak_fn=_wall_var_weak)

            # 8. Connectivity ratio (NETWORK)
            _vote("Connectivity ratio", row.get("Connectivity ratio"),
                  strong_fn=lambda v: v > cfg.connectivity_strong,
                  weak_fn=lambda v: v < 0.5)

            # 9. Clustering index (NETWORK)
            _vote("Clustering index", row.get("Clustering index"),
                  strong_fn=lambda v: v < 1.0,
                  weak_fn=lambda v: v > cfg.clustering_weak)

            # 10. Fracture index (MECHANICAL)
            _vote("Fracture index", row.get("Fracture index"),
                  strong_fn=lambda v: v < 0.01,
                  weak_fn=lambda v: v > cfg.fracture_index_weak,
                  base_conf="MEDIUM")  # Canny-based; inherently noisier

            # 11. GLCM homogeneity (TEXTURE)
            _vote("Homogeneity", row.get("Homogeneity"),
                  strong_fn=lambda v: v > cfg.homogeneity_strong,
                  weak_fn=lambda v: v < cfg.homogeneity_weak,
                  base_conf="MEDIUM")  # GLCM computed on masked crop; moderate reliability

            # ── Check if enough reliable metrics to classify ──
            reliable_fraction = reliable_count / total_rules if total_rules > 0 else 0.0
            can_classify = reliable_fraction >= cfg.classification_min_reliable_fraction

            # ── Score: map votes to 0–100 ──
            if max_votes > 0:
                raw_score = float(votes + max_votes) / float(2 * max_votes) * 100.0
            else:
                raw_score = 50.0
            raw_score = float(np.clip(raw_score, 0.0, 100.0))

            # ── Classify ──
            if not can_classify:
                classification = "Cannot Determine"
            elif raw_score >= cfg.strong_threshold:
                classification = "Strong"
            elif raw_score >= cfg.moderate_threshold:
                classification = "Moderate"
            else:
                classification = "Weak / Crumbly"

            return pd.Series({
                "Crumb Strength Score": raw_score,
                "Classification": classification,
                "Metric Confidence": str(conf_map),
                "_reliable_fraction": reliable_fraction,
            })

        score_cols = out.apply(_score_row, axis=1)
        out["Crumb Strength Score"] = score_cols["Crumb Strength Score"]
        out["Classification"] = score_cols["Classification"]
        out["Metric Confidence"] = score_cols["Metric Confidence"]
        return out

    def _interpret_row(self, row: pd.Series) -> str:
        """Generate a mechanistic narrative using spec-aligned thresholds.
        Uses only metrics that are not flagged as NOT RELIABLE or LOW confidence.
        """
        cfg = self.config
        reasons_bad: List[str] = []
        reasons_good: List[str] = []

        # Helper: skip unreliable metrics
        def _val(col: str):
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return v

        # ── Weak signals (spec-aligned thresholds) ──────────────────
        fi = _val("Fracture index")
        if fi is not None and fi > cfg.fracture_index_weak:
            reasons_bad.append(f"high fracture index ({fi:.4f}) — pre-existing micro-tears")
        cr = _val("Connectivity ratio")
        if cr is not None and cr < 0.50:
            reasons_bad.append(f"low inter-pore connectivity ({cr:.2f}) — isolated pore network")
        pore_cv = _val("Pore CV")
        if pore_cv is not None and pore_cv > cfg.cell_size_cv_weak:
            reasons_bad.append(f"high cell-size CV ({pore_cv:.2f} > {cfg.cell_size_cv_weak}) — highly irregular pore sizes")
        trf = _val("Thin region fraction")
        if trf is not None and trf > 0.40:
            reasons_bad.append("large thin-wall fraction — structural support compromised")
        por = _val("Porosity")
        if por is not None and por > cfg.porosity_weak:
            reasons_bad.append(f"high porosity ({por:.3f} > {cfg.porosity_weak}) — fragile aerated structure")
        ci = _val("Clustering index")
        if ci is not None and ci > cfg.clustering_weak:
            reasons_bad.append(f"pore clustering (index={ci:.2f}) — heterogeneous pore distribution")
        circ = _val("Circularity")
        if circ is not None and circ < cfg.circularity_weak:
            reasons_bad.append(f"low pore circularity ({circ:.2f} < {cfg.circularity_weak}) — collapsed/deformed cells")
        hom = _val("Homogeneity")
        if hom is not None and hom < cfg.homogeneity_weak:
            reasons_bad.append(f"low texture homogeneity ({hom:.3f} < {cfg.homogeneity_weak}) — irregular texture")

        # ── Strong signals (spec-aligned thresholds) ──────────────────
        mwt = _val("Mean wall thickness")
        if mwt is not None and mwt > cfg.wall_thickness_strong:
            reasons_good.append(f"thick crumb walls ({mwt:.2f} px) — strong load-bearing structure")
        if hom is not None and hom > cfg.homogeneity_strong:
            reasons_good.append(f"high texture homogeneity ({hom:.3f} > {cfg.homogeneity_strong}) — uniform texture")
        if cr is not None and cr > cfg.connectivity_strong:
            reasons_good.append(f"high connectivity ({cr:.2f} > {cfg.connectivity_strong}) — continuous pore network")
        if circ is not None and circ > cfg.circularity_strong:
            reasons_good.append(f"high pore circularity ({circ:.2f} > {cfg.circularity_strong}) — stable, round cells")
        pu = _val("Porosity uniformity")
        if pu is not None and pu < cfg.porosity_uniformity_strong:
            reasons_good.append("uniform porosity distribution")
        if pore_cv is not None and pore_cv < 0.50:
            reasons_good.append(f"consistent cell sizes (CV={pore_cv:.2f})")
        wtv = _val("Wall thickness variance")
        if wtv is not None and mwt is not None and wtv < cfg.wall_variance_strong and mwt > cfg.wall_thickness_strong:
            reasons_good.append("thick and uniform walls")

        classification = row.get("Classification", "")

        if classification == "Cannot Determine":
            return "Classification cannot be determined: insufficient reliable metrics due to image quality or segmentation issues."

        if classification == "Weak / Crumbly":
            if reasons_bad:
                return (
                    f"Weak / Crumbly: {', '.join(reasons_bad[:3])}. "
                    "Likely to crumble or fracture under stress."
                )
            return "Weak / Crumbly: low score across structural metrics with unfavorable morphology."

        if classification == "Strong":
            if reasons_good:
                return (
                    f"Strong crumb: {', '.join(reasons_good[:3])}. "
                    "Pore network is stable and well-supported."
                )
            return "Strong crumb: favorable wall support and connectivity across all metrics."

        # Moderate
        if reasons_bad and reasons_good:
            return (
                f"Moderate crumb: {reasons_good[0]}, but {reasons_bad[0]}. "
                f"Borderline structure — consider addressing {reasons_bad[0]}."
            )
        return "Moderate structure with intermediate porosity-wall-connectivity balance."

    def analyze(self, input_path: Path, output_dir: Path, save_styled_html: bool = True) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = self._collect_images(input_path)
        if not images:
            raise ValueError("No images found in input path.")

        rows = []

        for img_path in images:
            sample = img_path.stem
            rgb = self._load_image(img_path)
            gray, norm, blur = self._preprocess(rgb)
            crumb, pores, roi = self._segment_crumb(blur)

            pore = self._pore_features(pores, roi)
            wall, thickness_map, thin_mask = self._wall_features(crumb)
            conn, skel, _, _ = self._connectivity_features(crumb)
            frac, fractures = self._fracture_features(blur, crumb, pores)
            tex = self._texture_features(norm, roi, crumb)

            self._save_visuals(sample, output_dir / "visual_outputs", rgb, crumb, skel, thin_mask, fractures)

            spatial = self._spatial_features(pores, roi)

            rows.append(
                {
                    "Sample name": sample,
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

        df = pd.DataFrame(rows)
        df = self._compute_scores(df)
        df["Interpretation"] = df.apply(self._interpret_row, axis=1)

        # Keep requested output columns and sort.
        out_cols = [
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
            "Crumb Strength Score",
            "Classification",
            "Interpretation",
        ]

        df = df[out_cols].sort_values("Crumb Strength Score", ascending=False).reset_index(drop=True)

        # Round for clean readability.
        num_cols = [
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
            "Crumb Strength Score",
        ]
        df[num_cols] = df[num_cols].round(4)

        csv_path = output_dir / "crumb_structure_summary.csv"
        df.to_csv(csv_path, index=False)

        print("\n=== Crumb Structure Summary (sorted by Crumb Strength Score) ===")
        print(df.to_string(index=False))
        print(f"\nSaved CSV: {csv_path}")

        if save_styled_html:
            style = (
                df.style.background_gradient(subset=["Crumb Strength Score"], cmap="YlGn")
                .format({
                    "Porosity": "{:.3f}",
                    "Mean pore size": "{:.2f}",
                    "Pore CV": "{:.2f}",
                    "Mean wall thickness": "{:.2f}",
                    "Thin region fraction": "{:.3f}",
                    "Connectivity ratio": "{:.3f}",
                    "Fracture index": "{:.4f}",
                    "Homogeneity": "{:.3f}",
                    "Crumb Strength Score": "{:.1f}",
                })
            )
            html_path = output_dir / "crumb_structure_summary_styled.html"
            style.to_html(html_path)
            print(f"Saved styled table: {html_path}")

        # Bonus: compare best and weakest sample.
        best = df.iloc[0]
        worst = df.iloc[-1]
        print("\n=== Comparison ===")
        print(f"Best structure: {best['Sample name']} (Score: {best['Crumb Strength Score']:.1f}, {best['Classification']})")
        print(f"Weakest structure: {worst['Sample name']} (Score: {worst['Crumb Strength Score']:.1f}, {worst['Classification']})")

        print("\n=== Final Conclusions per Sample ===")
        for _, row in df.iterrows():
            print(f"- {row['Sample name']}: {row['Classification']} (Score {row['Crumb Strength Score']:.1f}). {row['Interpretation']}")

        return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quantitative cake crumb structure analysis (classical CV only).")
    p.add_argument("--input", required=True, help="Path to a single image file OR folder of images")
    p.add_argument("--output", default="crumb_analysis_output", help="Output directory")
    p.add_argument("--debug", action="store_true", help="Save debug segmentation visualizations")
    p.add_argument("--no-style", action="store_true", help="Disable styled HTML table output")
    p.add_argument("--use-imagej", action="store_true", help="Try optional ImageJ initialization (pyimagej)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = AnalyzerConfig(debug_visualization=args.debug, try_imagej=args.use_imagej)
    analyzer = CrumbAnalyzer(cfg)
    analyzer.analyze(Path(args.input), Path(args.output), save_styled_html=not args.no_style)


if __name__ == "__main__":
    main()
