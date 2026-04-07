import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
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

    # Composite scoring weights (must sum to 1.0)
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "wall_thickness": 0.20,
            "connectivity": 0.20,
            "homogeneity": 0.15,
            "pore_cv": 0.15,
            "thin_fraction": 0.10,
            "fracture_index": 0.15,
            "excess_porosity": 0.05,
        }
    )

    # Excessive porosity threshold (fraction, not percent)
    porosity_high_threshold: float = 0.35

    # Classification thresholds
    strong_threshold: float = 70.0
    moderate_threshold: float = 40.0

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
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if self._ij is not None:
            # Optional ImageJ hook point. Kept minimal to avoid runtime dependency when not installed.
            pass

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )
        norm = clahe.apply(gray)
        blur = cv2.GaussianBlur(norm, self.config.gaussian_kernel, 0)
        return gray, norm, blur

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        labeled = measure.label(mask, connectivity=2)
        props = measure.regionprops(labeled)
        if not props:
            return np.zeros_like(mask, dtype=bool)
        largest = max(props, key=lambda p: p.area)
        return labeled == largest.label

    def _segment_crumb(self, blur_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Otsu threshold and choose polarity with most plausible largest connected crumb body.
        _, th = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b1 = th > 0
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

    def _pore_features(self, pores: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
        roi_area = float(roi.sum())
        air_area = float(pores.sum())
        porosity = air_area / roi_area if roi_area > 0 else np.nan

        labeled = measure.label(pores, connectivity=2)
        areas = np.array([r.area for r in measure.regionprops(labeled)], dtype=float)

        if areas.size == 0:
            return {
                "porosity": porosity,
                "mean_pore_size": 0.0,
                "std_pore_size": 0.0,
                "pore_cv": 0.0,
            }

        mean_a = float(np.mean(areas))
        std_a = float(np.std(areas))
        cv_a = float(std_a / mean_a) if mean_a > 1e-9 else 0.0
        return {
            "porosity": porosity,
            "mean_pore_size": mean_a,
            "std_pore_size": std_a,
            "pore_cv": cv_a,
        }

    def _wall_features(self, crumb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        dist = ndi.distance_transform_edt(crumb)
        local_thickness = 2.0 * dist
        crumb_pixels = crumb.sum()
        mean_thick = float(local_thickness[crumb].mean()) if crumb_pixels > 0 else np.nan

        thin_mask = crumb & (local_thickness < self.config.thin_thickness_px)
        thin_frac = float(thin_mask.sum() / crumb_pixels) if crumb_pixels > 0 else np.nan

        return {
            "mean_wall_thickness": mean_thick,
            "thin_region_fraction": thin_frac,
        }, local_thickness, thin_mask

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
            reshaped = z.reshape(h, size, h, size)
            blocks = reshaped.any(axis=(1, 3))
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

    def _compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Derived penalty for excessive porosity only.
        out["excess_porosity"] = np.clip(out["Porosity"] - self.config.porosity_high_threshold, a_min=0, a_max=None)

        features = {
            "wall_thickness": ("Mean wall thickness", True),
            "connectivity": ("Connectivity ratio", True),
            "homogeneity": ("Homogeneity", True),
            "pore_cv": ("Pore CV", False),
            "thin_fraction": ("Thin region fraction", False),
            "fracture_index": ("Fracture index", False),
            "excess_porosity": ("excess_porosity", False),
        }

        norm_cols = {}
        for key, (col, high_good) in features.items():
            vals = out[col].astype(float)
            vmin, vmax = vals.min(), vals.max()
            if np.isclose(vmin, vmax, equal_nan=True):
                norm = pd.Series(np.full(len(vals), 0.5), index=vals.index)
            else:
                norm = (vals - vmin) / (vmax - vmin)
            if not high_good:
                norm = 1.0 - norm
            norm_cols[key] = norm

        score = np.zeros(len(out), dtype=float)
        for key, w in self.config.score_weights.items():
            score += w * norm_cols[key].values

        out["Crumb Strength Score"] = np.clip(score * 100.0, 0.0, 100.0)

        def classify(s: float) -> str:
            if s > self.config.strong_threshold:
                return "Strong"
            if s >= self.config.moderate_threshold:
                return "Moderate"
            return "Weak / Crumbly"

        out["Classification"] = out["Crumb Strength Score"].apply(classify)
        return out

    def _interpret_row(self, row: pd.Series) -> str:
        reasons_bad: List[str] = []
        reasons_good: List[str] = []

        if row["Fracture index"] > 0.025:
            reasons_bad.append("high fracture index")
        if row["Connectivity ratio"] < 0.20:
            reasons_bad.append("low connectivity")
        if row["Pore CV"] > 1.0:
            reasons_bad.append("high pore-size variability")
        if row["Thin region fraction"] > 0.40:
            reasons_bad.append("large thin-wall fraction")
        if row["Porosity"] > self.config.porosity_high_threshold:
            reasons_bad.append("excessive porosity")

        if row["Mean wall thickness"] > 4.0:
            reasons_good.append("thicker crumb walls")
        if row["Homogeneity"] > 0.45:
            reasons_good.append("higher texture homogeneity")
        if row["Connectivity ratio"] >= 0.25:
            reasons_good.append("better network connectivity")

        if row["Classification"] == "Weak / Crumbly":
            if reasons_bad:
                return (
                    f"Likely weak/crumbly structure driven by {', '.join(reasons_bad[:2])}. "
                    f"Micro-tears or discontinuities may be present."
                )
            return "Likely weak/crumbly structure based on low composite score and unfavorable morphology."

        if row["Classification"] == "Strong":
            if reasons_good:
                return (
                    f"Strong structure with {', '.join(reasons_good[:2])}. "
                    f"Pore pattern appears relatively stable and well-supported."
                )
            return "Strong structure indicated by favorable wall support and connectivity metrics."

        # Moderate
        if reasons_bad and reasons_good:
            return (
                f"Moderate structure: mixed signals with {reasons_good[0]} but also {reasons_bad[0]}. "
                f"Borderline crumb robustness."
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

            rows.append(
                {
                    "Sample name": sample,
                    "Porosity": pore["porosity"],
                    "Mean pore size": pore["mean_pore_size"],
                    "Pore CV": pore["pore_cv"],
                    "Mean wall thickness": wall["mean_wall_thickness"],
                    "Thin region fraction": wall["thin_region_fraction"],
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
            "Mean wall thickness",
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
            "Mean wall thickness",
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
