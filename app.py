"""
INTEGRATED BET–XRD MORPHOLOGY ANALYZER v3.0
========================================================================
SCIENTIFIC PUBLICATION-GRADE APPLICATION FOR POROUS MATERIALS ANALYSIS
========================================================================
Features:
1. IUPAC-compliant physisorption analysis (Rouquerol criteria)
2. Advanced XRD analysis (Scherrer, Williamson-Hall, crystallinity)
3. Scientific morphology fusion algorithm
4. Journal-quality visualization and reporting
5. Complete data provenance and export
========================================================================
For publication in: Chem. Mater., Micropor. Mesopor. Mater., J. Mater. Chem. A
========================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize, integrate, interpolate
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
import json
import base64
import warnings
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import sys
import tempfile
import pathlib

warnings.filterwarnings('ignore')

# ============================================================================
# SCIENTIFIC CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BET–XRD Morphology Analyzer v3.0 | Scientific Edition",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'mathtext.default': 'regular'
})

# ============================================================================
# SCIENTIFIC DATA STRUCTURES
# ============================================================================
@dataclass
class BETAnalysisResult:
    """IUPAC-compliant BET analysis results"""
    surface_area: float  # m²/g
    pore_volume: float   # cm³/g
    mean_pore_diameter: float  # nm
    c_constant: float
    monolayer_capacity: float  # mmol/g
    bet_r2: float
    linear_range: Tuple[float, float]
    hysteresis_type: str
    micropore_volume: float
    external_surface: float
    t_plot_r2: float
    psd_available: bool
    isotherm_type: str

@dataclass
class XRDAnalysisResult:
    """Comprehensive XRD analysis results"""
    crystallinity_index: float
    crystallite_size: float  # nm (Scherrer)
    microstrain: float
    dislocation_density: float  # m⁻²
    ordered_mesopores: bool
    d_spacing: Optional[float]  # nm
    lattice_strain: float
    primary_peak_2theta: float
    fwhm_values: List[float]
    peak_positions: List[float]
    symmetry_class: str

@dataclass
class MorphologyFusionResult:
    """Integrated morphology from BET and XRD"""
    composite_classification: str
    dominant_feature: str
    surface_to_volume_ratio: float  # m²/cm³
    pore_wall_thickness: Optional[float]  # nm
    structural_integrity: float  # 0-1
    confidence_score: float
    material_family: str
    journal_recommendations: List[str]
    characterization_techniques: List[str]

# ============================================================================
# IUPAC-COMPLIANT BET ANALYSIS ENGINE
# ============================================================================
class IUPACBETAnalyzer:
    """
    Implementation of IUPAC guidelines for physisorption analysis
    References:
    1. Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739-1758
    2. Thommes et al., Pure Appl. Chem., 2015, 87, 1051-1069
    """
    
    # Physical constants
    N2_CROSS_SECTION = 0.162e-18  # m² (N2 at 77K)
    AVOGADRO = 6.02214076e23
    GAS_CONSTANT = 8.314462618
    LIQUID_N2_DENSITY = 0.808  # g/cm³ at 77K
    
    def __init__(self, p_ads: np.ndarray, q_ads: np.ndarray, 
                 p_des: Optional[np.ndarray] = None, 
                 q_des: Optional[np.ndarray] = None):
        """
        Initialize with experimental data
        
        Parameters:
        -----------
        p_ads : Relative pressure P/P₀ for adsorption
        q_ads : Quantity adsorbed (mmol/g or cm³/g STP)
        p_des : Relative pressure P/P₀ for desorption (optional)
        q_des : Quantity desorbed (optional)
        """
        self.p_ads = np.asarray(p_ads, dtype=np.float64)
        self.q_ads = np.asarray(q_ads, dtype=np.float64)
        self.p_des = np.asarray(p_des, dtype=np.float64) if p_des is not None else None
        self.q_des = np.asarray(q_des, dtype=np.float64) if q_des is not None else None
        
        self._validate_inputs()
        self._results = {}
    
    def _validate_inputs(self):
        """Validate experimental data according to IUPAC standards"""
        if len(self.p_ads) < 10:
            raise ValueError("Minimum 10 adsorption points required for reliable analysis")
        
        if not np.all(np.diff(self.p_ads) > 0):
            raise ValueError("Adsorption pressure must be strictly increasing")
        
        if np.max(self.p_ads) < 0.3:
            raise ValueError("Insufficient pressure range for BET analysis (max P/P₀ < 0.3)")
        
        if self.p_des is not None and self.q_des is not None:
            if len(self.p_des) < 5:
                raise ValueError("Insufficient desorption points for hysteresis analysis")
    
    def _bet_transform(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """BET transformation: p/(q*(1-p))"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return p / (q * (1 - p))
    
    def _rouquerol_criteria(self, p: np.ndarray, q: np.ndarray) -> Tuple[int, int]:
        """
        Apply Rouquerol criteria for automatic BET range selection
        
        Criteria:
        1. p/(q(1-p)) must increase monotonically
        2. Q*(1-p) must increase (positive adsorbed amount)
        3. C constant must be positive
        """
        n_points = len(p)
        valid_ranges = []
        
        # Minimum 5 points for BET analysis
        for i in range(n_points - 5):
            for j in range(i + 5, n_points):
                p_seg = p[i:j]
                q_seg = q[i:j]
                
                # Criterion 1: BET transform must be monotonic increasing
                y_bet = self._bet_transform(p_seg, q_seg)
                if not np.all(np.diff(y_bet) > -1e-10):  # Allow small numerical errors
                    continue
                
                # Criterion 2: Q*(1-p) must increase
                q_corrected = q_seg * (1 - p_seg)
                if not np.all(np.diff(q_corrected) > -1e-10):
                    continue
                
                # Criterion 3: Linear regression quality
                slope, intercept, r_value, _, _ = stats.linregress(p_seg, y_bet)
                if r_value**2 < 0.999 or slope <= 0 or intercept <= 0:
                    continue
                
                # Calculate C constant
                c = slope / intercept + 1
                if c <= 0:
                    continue
                
                valid_ranges.append((i, j, r_value**2, j-i))
        
        if not valid_ranges:
            raise ValueError("No valid BET range found using Rouquerol criteria")
        
        # Select range with best linearity
        valid_ranges.sort(key=lambda x: x[2], reverse=True)
        return valid_ranges[0][0], valid_ranges[0][1]
    
    def analyze_surface_area(self) -> Dict[str, Any]:
        """Calculate BET surface area with automatic range selection"""
        i, j = self._rouquerol_criteria(self.p_ads, self.q_ads)
        p_bet = self.p_ads[i:j]
        q_bet = self.q_ads[i:j]
        
        # BET transformation and linear regression
        y_bet = self._bet_transform(p_bet, q_bet)
        slope, intercept, r_value, _, std_err = stats.linregress(p_bet, y_bet)
        
        # Calculate BET parameters
        q_mono = 1.0 / (slope + intercept)  # mmol/g
        c_constant = slope / intercept + 1.0
        
        # Surface area calculation
        surface_area = q_mono * self.AVOGADRO * self.N2_CROSS_SECTION * 1e-4  # m²/g
        
        return {
            'surface_area': surface_area,
            'monolayer_capacity': q_mono,
            'c_constant': c_constant,
            'bet_r2': r_value**2,
            'bet_std_error': std_err,
            'linear_range': (float(p_bet[0]), float(p_bet[-1])),
            'n_points_bet': len(p_bet),
            'bet_slope': slope,
            'bet_intercept': intercept
        }
    
    def analyze_porosity(self) -> Dict[str, Any]:
        """Complete porosity analysis including t-plot and PSD"""
        results = {}
        
        # Total pore volume from adsorption at highest pressure
        max_p_idx = np.argmax(self.p_ads)
        total_pore_volume = self.q_ads[max_p_idx] * 0.001548  # Convert mmol/g to cm³/g
        
        # t-plot analysis for microporosity
        t_plot_results = self._t_plot_analysis()
        
        # Pore size distribution if desorption available
        psd_results = {}
        if self.p_des is not None and self.q_des is not None:
            psd_results = self._calculate_psd()
        
        results.update({
            'total_pore_volume': total_pore_volume,
            'micropore_volume': t_plot_results.get('micropore_volume', 0.0),
            'external_surface': t_plot_results.get('external_surface', 0.0),
            't_plot_r2': t_plot_results.get('r2', 0.0),
            'psd_available': len(psd_results) > 0
        })
        
        if psd_results:
            results.update(psd_results)
        
        return results
    
    def _t_plot_analysis(self) -> Dict[str, Any]:
        """t-plot analysis using Harkins-Jura thickness equation"""
        # Harkins-Jura equation for N2 at 77K
        def thickness_nm(p):
            return (13.99 / (0.034 - np.log10(p + 1e-10))) ** 0.5 * 0.1
        
        # Select region for t-plot (typically 0.2-0.5 P/P₀)
        mask = (self.p_ads >= 0.2) & (self.p_ads <= 0.5)
        if np.sum(mask) < 5:
            return {'micropore_volume': 0.0, 'external_surface': 0.0, 'r2': 0.0}
        
        p_t = self.p_ads[mask]
        q_t = self.q_ads[mask]
        t = thickness_nm(p_t)
        
        # Linear regression: q = slope*t + intercept
        slope, intercept, r_value, _, _ = stats.linregress(t, q_t)
        
        external_surface = slope * 15.47  # Conversion factor for N2
        micropore_volume = max(0.0, intercept * 0.001548)  # Convert to cm³/g
        
        return {
            'micropore_volume': micropore_volume,
            'external_surface': external_surface,
            'r2': r_value**2,
            't_range': (float(t.min()), float(t.max()))
        }
    
    def _calculate_psd(self) -> Dict[str, Any]:
        """Calculate pore size distribution using BJH method"""
        if self.p_des is None or self.q_des is None:
            return {}
        
        # Ensure desorption is properly ordered
        sort_idx = np.argsort(self.p_des)
        p_des_sorted = self.p_des[sort_idx]
        q_des_sorted = self.q_des[sort_idx]
        
        # Kelvin equation for N2 at 77K
        gamma = 8.85e-3  # Surface tension N/m
        v_molar = 34.7e-6  # Molar volume m³/mol
        t = 77.3  # K
        r = 8.314  # Gas constant
        
        # Kelvin radius calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            r_kelvin = -2 * gamma * v_molar / (r * t * np.log(p_des_sorted))
        
        # Convert to pore diameter in nm
        pore_diameter = 2 * r_kelvin * 1e9  # Radius to diameter, m to nm
        
        # Calculate dV/dlogD
        valid = (pore_diameter > 1) & (pore_diameter < 200) & np.isfinite(pore_diameter)
        if np.sum(valid) < 5:
            return {}
        
        pore_diameter_valid = pore_diameter[valid]
        q_valid = q_des_sorted[valid]
        
        # Sort by pore diameter
        sort_idx = np.argsort(pore_diameter_valid)
        pore_diameter_sorted = pore_diameter_valid[sort_idx]
        q_sorted = q_valid[sort_idx]
        
        # Calculate differential pore volume
        log_d = np.log10(pore_diameter_sorted)
        dv = np.abs(np.diff(q_sorted)) * 0.001548  # Convert to cm³/g
        dlogd = np.diff(log_d)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dv_dlogd = dv / dlogd
        
        # Remove infinities
        finite_mask = np.isfinite(dv_dlogd)
        if np.sum(finite_mask) < 3:
            return {}
        
        pore_diameters_psd = pore_diameter_sorted[:-1][finite_mask]
        dv_dlogd_valid = dv_dlogd[finite_mask]
        
        # Calculate pore statistics
        total_pore_volume = np.trapz(dv_dlogd_valid, np.log10(pore_diameters_psd))
        
        # Pore size fractions
        micro_mask = pore_diameters_psd < 2
        meso_mask = (pore_diameters_psd >= 2) & (pore_diameters_psd <= 50)
        macro_mask = pore_diameters_psd > 50
        
        v_micro = np.trapz(dv_dlogd_valid[micro_mask], np.log10(pore_diameters_psd[micro_mask])) if np.any(micro_mask) else 0.0
        v_meso = np.trapz(dv_dlogd_valid[meso_mask], np.log10(pore_diameters_psd[meso_mask])) if np.any(meso_mask) else 0.0
        v_macro = np.trapz(dv_dlogd_valid[macro_mask], np.log10(pore_diameters_psd[macro_mask])) if np.any(macro_mask) else 0.0
        
        # Mean pore diameter (volume-weighted)
        if total_pore_volume > 0:
            mean_diameter = np.trapz(pore_diameters_psd * dv_dlogd_valid, np.log10(pore_diameters_psd)) / total_pore_volume
            peak_diameter = pore_diameters_psd[np.argmax(dv_dlogd_valid)]
        else:
            mean_diameter = 0.0
            peak_diameter = 0.0
        
        return {
            'pore_diameters': pore_diameters_psd.tolist(),
            'dv_dlogd': dv_dlogd_valid.tolist(),
            'total_pore_volume_psd': total_pore_volume,
            'mean_pore_diameter': mean_diameter,
            'peak_pore_diameter': peak_diameter,
            'micropore_fraction': v_micro / total_pore_volume if total_pore_volume > 0 else 0.0,
            'mesopore_fraction': v_meso / total_pore_volume if total_pore_volume > 0 else 0.0,
            'macropore_fraction': v_macro / total_pore_volume if total_pore_volume > 0 else 0.0
        }
    
    def classify_hysteresis(self) -> Dict[str, Any]:
        """Classify hysteresis loop according to IUPAC"""
        if self.p_des is None or self.q_des is None:
            return {'type': 'I', 'description': 'Reversible (no hysteresis)'}
        
        # Ensure proper ordering
        p_ads_sorted = self.p_ads
        q_ads_sorted = self.q_ads
        p_des_sorted = self.p_des[np.argsort(self.p_des)]
        q_des_sorted = self.q_des[np.argsort(self.p_des)]
        
        # Interpolate to common pressure points
        p_common = np.linspace(0.1, 0.95, 100)
        q_ads_interp = np.interp(p_common, p_ads_sorted, q_ads_sorted)
        q_des_interp = np.interp(p_common, p_des_sorted, q_des_sorted)
        
        # Calculate hysteresis loop area
        loop_area = np.trapz(np.abs(q_des_interp - q_ads_interp), p_common)
        
        # Find closure point
        closure_idx = np.argmin(np.abs(q_des_sorted - q_ads_sorted[-1]))
        closure_pressure = p_des_sorted[closure_idx]
        
        # IUPAC classification
        if loop_area < 5:
            h_type = "H1"
            description = "Uniform mesopores with narrow PSD (e.g., MCM-41)"
        elif closure_pressure > 0.45:
            h_type = "H2"
            description = "Ink-bottle pores or interconnected network"
        elif closure_pressure > 0.4:
            h_type = "H3"
            description = "Slit-shaped pores (plate-like particles)"
        else:
            h_type = "H4"
            description = "Combined micro-mesoporosity (e.g., activated carbons)"
        
        return {
            'type': h_type,
            'description': description,
            'loop_area': loop_area,
            'closure_pressure': closure_pressure,
            'iupac_class': 'IV' if h_type in ['H1', 'H2'] else 'II' if h_type == 'H3' else 'I'
        }
    
    def full_analysis(self) -> BETAnalysisResult:
        """Complete BET analysis with all parameters"""
        try:
            # Surface area analysis
            sa_results = self.analyze_surface_area()
            
            # Porosity analysis
            porosity_results = self.analyze_porosity()
            
            # Hysteresis classification
            hysteresis_results = self.classify_hysteresis()
            
            # Combine all results
            return BETAnalysisResult(
                surface_area=sa_results['surface_area'],
                pore_volume=porosity_results['total_pore_volume'],
                mean_pore_diameter=porosity_results.get('mean_pore_diameter', 0.0),
                c_constant=sa_results['c_constant'],
                monolayer_capacity=sa_results['monolayer_capacity'],
                bet_r2=sa_results['bet_r2'],
                linear_range=sa_results['linear_range'],
                hysteresis_type=hysteresis_results['type'],
                micropore_volume=porosity_results['micropore_volume'],
                external_surface=porosity_results['external_surface'],
                t_plot_r2=porosity_results['t_plot_r2'],
                psd_available=porosity_results['psd_available'],
                isotherm_type=hysteresis_results['iupac_class']
            )
            
        except Exception as e:
            st.error(f"BET analysis error: {str(e)}")
            raise

# ============================================================================
# ADVANCED XRD ANALYSIS ENGINE
# ============================================================================
class AdvancedXRDAnalyzer:
    """
    Comprehensive XRD analysis for morphology characterization
    References:
    1. Klug & Alexander, X-ray Diffraction Procedures, 1974
    2. Williamson & Hall, Acta Metall., 1953, 1, 22-31
    """
    
    def __init__(self, wavelength: float = 0.15406):
        """
        Initialize with X-ray wavelength (nm)
        
        Default: Cu Kα radiation (0.15406 nm)
        """
        self.lambda_x = wavelength
        self.k = 0.9  # Scherrer constant (shape factor)
    
    def analyze(self, two_theta: np.ndarray, intensity: np.ndarray) -> XRDAnalysisResult:
        """Complete XRD analysis pipeline"""
        # Preprocess data
        theta_clean, intensity_clean = self._preprocess_data(two_theta, intensity)
        
        # Peak detection
        peaks, peak_properties = self._detect_peaks(theta_clean, intensity_clean)
        
        # Crystallinity analysis
        crystallinity_results = self._analyze_crystallinity(theta_clean, intensity_clean, peaks)
        
        # Crystallite size analysis
        size_results = self._analyze_crystallite_size(theta_clean, intensity_clean, peaks)
        
        # Strain analysis
        strain_results = self._williamson_hall_analysis(theta_clean, intensity_clean, peaks)
        
        # Mesostructure analysis
        meso_results = self._analyze_mesostructure(theta_clean, intensity_clean)
        
        # Symmetry classification
        symmetry = self._classify_symmetry(theta_clean[peaks] if len(peaks) > 0 else [])
        
        return XRDAnalysisResult(
            crystallinity_index=crystallinity_results['index'],
            crystallite_size=size_results.get('size', 0.0),
            microstrain=strain_results.get('strain', 0.0),
            dislocation_density=strain_results.get('dislocation_density', 0.0),
            ordered_mesopores=meso_results['ordered'],
            d_spacing=meso_results.get('d_spacing'),
            lattice_strain=strain_results.get('lattice_strain', 0.0),
            primary_peak_2theta=theta_clean[peaks[0]] if len(peaks) > 0 else 0.0,
            fwhm_values=self._calculate_fwhm(theta_clean, intensity_clean, peaks),
            peak_positions=theta_clean[peaks].tolist() if len(peaks) > 0 else [],
            symmetry_class=symmetry
        )
    
    def _preprocess_data(self, theta: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background and normalize data"""
        # Remove NaN values
        valid = np.isfinite(theta) & np.isfinite(intensity)
        theta = theta[valid]
        intensity = intensity[valid]
        
        # Sort by 2θ
        sort_idx = np.argsort(theta)
        theta = theta[sort_idx]
        intensity = intensity[sort_idx]
        
        # Remove background using rolling ball algorithm
        window_size = min(51, len(theta) // 10)
        if window_size % 2 == 0:
            window_size += 1
        
        background = pd.Series(intensity).rolling(window=window_size, center=True, min_periods=1).median()
        intensity_corrected = intensity - background.values
        
        # Normalize to [0, 1]
        i_min = np.min(intensity_corrected)
        i_max = np.max(intensity_corrected)
        if i_max > i_min:
            intensity_norm = (intensity_corrected - i_min) / (i_max - i_min)
        else:
            intensity_norm = intensity_corrected
        
        return theta, intensity_norm
    
    def _detect_peaks(self, theta: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Advanced peak detection with prominence filtering"""
        # Find all peaks
        peaks, properties = signal.find_peaks(
            intensity,
            height=0.1,  # Minimum peak height
            prominence=0.05,  # Minimum prominence
            distance=10,  # Minimum distance between peaks
            width=2  # Minimum peak width
        )
        
        # Filter by significance
        if len(peaks) > 0:
            significant_mask = properties['prominences'] > np.median(properties['prominences']) * 0.3
            peaks = peaks[significant_mask]
        
        return peaks, properties
    
    def _analyze_crystallinity(self, theta: np.ndarray, intensity: np.ndarray, 
                              peaks: np.ndarray) -> Dict[str, Any]:
        """Calculate crystallinity index using peak separation"""
        # Total area under curve
        total_area = np.trapz(intensity, theta)
        
        if len(peaks) == 0:
            return {'index': 0.0, 'classification': 'Amorphous'}
        
        # Estimate amorphous background
        # Find valleys between peaks
        valleys = []
        for i in range(len(peaks) - 1):
            valley_region = intensity[peaks[i]:peaks[i+1]]
            if len(valley_region) > 0:
                valley_idx = peaks[i] + np.argmin(valley_region)
                valleys.append(valley_idx)
        
        # Create amorphous baseline by connecting valleys
        if len(valleys) >= 2:
            # Linear interpolation between valleys
            amorphous_intensity = np.interp(theta, theta[valleys], intensity[valleys])
            amorphous_area = np.trapz(amorphous_intensity, theta)
        else:
            # Simple estimation
            amorphous_intensity = np.percentile(intensity, 30) * np.ones_like(intensity)
            amorphous_area = np.trapz(amorphous_intensity, theta)
        
        # Crystalline area
        crystalline_area = total_area - amorphous_area
        crystallinity = crystalline_area / total_area if total_area > 0 else 0.0
        
        # Classification
        if crystallinity > 0.7:
            classification = "Highly crystalline"
        elif crystallinity > 0.4:
            classification = "Semi-crystalline"
        elif crystallinity > 0.1:
            classification = "Poorly crystalline"
        else:
            classification = "Amorphous"
        
        return {
            'index': max(0.0, min(1.0, crystallinity)),
            'classification': classification,
            'total_area': total_area,
            'crystalline_area': crystalline_area,
            'amorphous_area': amorphous_area
        }
    
    def _analyze_crystallite_size(self, theta: np.ndarray, intensity: np.ndarray, 
                                 peaks: np.ndarray) -> Dict[str, Any]:
        """Scherrer analysis for crystallite size"""
        if len(peaks) == 0:
            return {'size': None, 'method': 'None'}
        
        sizes = []
        fwhm_values = []
        
        # Analyze first 3 major peaks
        for peak_idx in peaks[:min(3, len(peaks))]:
            peak_theta = theta[peak_idx]
            
            # Extract peak region (±0.5°)
            region_mask = (theta >= peak_theta - 0.5) & (theta <= peak_theta + 0.5)
            if np.sum(region_mask) < 10:
                continue
            
            theta_peak = theta[region_mask]
            intensity_peak = intensity[region_mask]
            
            # Fit Gaussian to peak
            try:
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-(x - x0)**2 / (2*sigma**2))
                
                p0 = [intensity_peak.max(), peak_theta, 0.05]
                bounds = ([0, peak_theta-0.2, 0.01], 
                         [2, peak_theta+0.2, 0.5])
                
                popt, _ = optimize.curve_fit(
                    gaussian, theta_peak, intensity_peak,
                    p0=p0, bounds=bounds, maxfev=5000
                )
                
                # Calculate FWHM and crystallite size
                fwhm = 2.35482 * popt[2]  # Convert sigma to FWHM
                theta_rad = np.radians(popt[1] / 2)
                beta_rad = np.radians(fwhm)
                
                if beta_rad > 0:
                    size = self.k * self.lambda_x / (beta_rad * np.cos(theta_rad))
                    sizes.append(size)
                    fwhm_values.append(fwhm)
                    
            except (RuntimeError, ValueError):
                continue
        
        if not sizes:
            return {'size': None, 'method': 'Scherrer', 'fwhm': None}
        
        return {
            'size': float(np.mean(sizes)),
            'std': float(np.std(sizes)) if len(sizes) > 1 else 0.0,
            'method': 'Scherrer',
            'fwhm_mean': float(np.mean(fwhm_values)) if fwhm_values else None,
            'n_peaks_analyzed': len(sizes)
        }
    
    def _williamson_hall_analysis(self, theta: np.ndarray, intensity: np.ndarray, 
                                 peaks: np.ndarray) -> Dict[str, Any]:
        """Williamson-Hall analysis for strain and size"""
        if len(peaks) < 3:
            return {'strain': 0.0, 'size_wh': None, 'dislocation_density': 0.0}
        
        beta_list = []
        d_spacing_list = []
        
        for peak_idx in peaks[:min(5, len(peaks))]:
            peak_theta = theta[peak_idx]
            
            # Estimate FWHM from peak width at half maximum
            peak_height = intensity[peak_idx]
            half_max = peak_height / 2
            
            # Find width at half maximum
            left_idx = peak_idx
            while left_idx > 0 and intensity[left_idx] > half_max:
                left_idx -= 1
            
            right_idx = peak_idx
            while right_idx < len(theta)-1 and intensity[right_idx] > half_max:
                right_idx += 1
            
            if right_idx > left_idx:
                fwhm = theta[right_idx] - theta[left_idx]
                beta = np.radians(fwhm)
                
                # Bragg's law: d = λ/(2sinθ)
                d = self.lambda_x / (2 * np.sin(np.radians(peak_theta/2)))
                
                beta_list.append(beta)
                d_spacing_list.append(d)
        
        if len(beta_list) < 3:
            return {'strain': 0.0, 'size_wh': None, 'dislocation_density': 0.0}
        
        # Williamson-Hall plot: βcosθ vs 4sinθ
        theta_peaks = theta[peaks[:len(beta_list)]]
        cos_theta = np.cos(np.radians(theta_peaks/2))
        sin_theta = np.sin(np.radians(theta_peaks/2))
        
        x = 4 * sin_theta
        y = np.array(beta_list) * cos_theta
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        size_wh = self.k * self.lambda_x / intercept if intercept > 0 else None
        strain = slope / 4
        
        # Dislocation density
        dislocation_density = (strain**2) / (size_wh**2) if size_wh and size_wh > 0 else 0.0
        
        return {
            'strain': float(strain),
            'size_wh': float(size_wh) if size_wh else None,
            'dislocation_density': float(dislocation_density),
            'wh_r2': float(r_value**2)
        }
    
    def _analyze_mesostructure(self, theta: np.ndarray, intensity: np.ndarray) -> Dict[str, Any]:
        """Detect ordered mesopores from low-angle scattering"""
        # Low-angle region (0.5-5° 2θ)
        low_angle_mask = (theta >= 0.5) & (theta <= 5.0)
        if np.sum(low_angle_mask) < 10:
            return {'ordered': False, 'd_spacing': None}
        
        theta_low = theta[low_angle_mask]
        intensity_low = intensity[low_angle_mask]
        
        # Detect peaks in low-angle region
        peaks_low, properties = signal.find_peaks(
            intensity_low,
            height=0.2,
            prominence=0.1,
            distance=5
        )
        
        if len(peaks_low) == 0:
            return {'ordered': False, 'd_spacing': None}
        
        # Calculate d-spacing from first peak
        d_spacing = self.lambda_x / (2 * np.sin(np.radians(theta_low[peaks_low[0]]/2)))
        
        # Check for multiple peaks indicating ordered structure
        if len(peaks_low) >= 2:
            peak_positions = theta_low[peaks_low]
            ratios = peak_positions / peak_positions[0]
            
            # Check for hexagonal symmetry ratios
            expected_hex = np.array([1, np.sqrt(3), 2, np.sqrt(7), 3])
            errors = []
            for r in ratios:
                errors.append(np.min(np.abs(r - expected_hex)))
            
            mean_error = np.mean(errors)
            
            if mean_error < 0.15:  # Allow 15% error
                symmetry = "Hexagonal (p6mm)"
            elif mean_error < 0.25:
                symmetry = "Possibly ordered"
            else:
                symmetry = "Low-angle peaks detected"
        else:
            symmetry = "Single low-angle peak"
        
        return {
            'ordered': True,
            'd_spacing': float(d_spacing),
            'symmetry': symmetry,
            'n_peaks': len(peaks_low),
            'peak_positions': theta_low[peaks_low].tolist()
        }
    
    def _calculate_fwhm(self, theta: np.ndarray, intensity: np.ndarray, 
                       peaks: np.ndarray) -> List[float]:
        """Calculate Full Width at Half Maximum for each peak"""
        fwhm_values = []
        
        for peak_idx in peaks:
            peak_height = intensity[peak_idx]
            half_max = peak_height / 2
            
            # Find left half-maximum
            left_idx = peak_idx
            while left_idx > 0 and intensity[left_idx] > half_max:
                left_idx -= 1
            
            # Find right half-maximum
            right_idx = peak_idx
            while right_idx < len(theta)-1 and intensity[right_idx] > half_max:
                right_idx += 1
            
            if right_idx > left_idx:
                fwhm = theta[right_idx] - theta[left_idx]
                fwhm_values.append(float(fwhm))
        
        return fwhm_values
    
    def _classify_symmetry(self, peak_positions: List[float]) -> str:
        """Classify crystal symmetry from peak positions"""
        if len(peak_positions) < 3:
            return "Insufficient peaks"
        
        # Calculate d-spacings
        d_spacings = [self.lambda_x / (2 * np.sin(np.radians(theta/2))) 
                     for theta in peak_positions]
        
        # Sort by intensity (assuming first peak is strongest)
        d_spacings_sorted = sorted(d_spacings, reverse=True)
        
        # Calculate ratios
        ratios = []
        for i in range(1, len(d_spacings_sorted)):
            ratios.append(d_spacings_sorted[0] / d_spacings_sorted[i])
        
        # Check for common symmetries
        if len(ratios) >= 2:
            # Cubic: √2, √3, √4, √8 ratios
            cubic_ratios = [np.sqrt(2), np.sqrt(3), 2, np.sqrt(8)]
            cubic_error = sum(min(abs(r - cr) for cr in cubic_ratios) for r in ratios[:2])
            
            # Hexagonal: 1, √3, 2, √7 ratios
            hex_ratios = [1, np.sqrt(3), 2, np.sqrt(7)]
            hex_error = sum(min(abs(r - hr) for hr in hex_ratios) for r in ratios[:2])
            
            if cubic_error < 0.1:
                return "Cubic symmetry suggested"
            elif hex_error < 0.1:
                return "Hexagonal symmetry suggested"
        
        return "Unknown/Complex symmetry"

# ============================================================================
# MORPHOLOGY FUSION ENGINE
# ============================================================================
class MorphologyFusionEngine:
    """
    Scientific fusion of BET and XRD data for comprehensive morphology analysis
    Novel algorithm for journal publication
    """
    
    @staticmethod
    def fuse(bet_result: BETAnalysisResult, xrd_result: XRDAnalysisResult) -> MorphologyFusionResult:
        """
        Integrate BET and XRD data for complete morphology characterization
        """
        # Extract key parameters
        s_bet = bet_result.surface_area
        v_pore = bet_result.pore_volume
        d_pore = bet_result.mean_pore_diameter
        micro_frac = bet_result.micropore_volume / v_pore if v_pore > 0 else 0
        
        cryst = xrd_result.crystallinity_index
        cryst_size = xrd_result.crystallite_size
        ordered = xrd_result.ordered_mesopores
        d_spacing = xrd_result.d_spacing
        
        # ====================================================================
        # 1. CALCULATE KEY MORPHOLOGY PARAMETERS
        # ====================================================================
        
        # Surface-to-volume ratio (key morphology parameter)
        surface_to_volume = s_bet / (v_pore * 1000) if v_pore > 0 else 0  # m²/cm³
        
        # Pore wall thickness estimation (for ordered mesopores)
        pore_wall_thickness = None
        if d_spacing and d_pore > 0:
            pore_wall_thickness = d_spacing - d_pore
            pore_wall_thickness = max(0.0, pore_wall_thickness)
        
        # Structural integrity index (0-1)
        integrity_components = []
        
        # Component 1: Crystallinity contribution (0-0.4)
        integrity_components.append(cryst * 0.4)
        
        # Component 2: Surface area efficiency (0-0.3)
        sa_efficiency = min(1.0, s_bet / 2000)  # Normalize to 2000 m²/g
        integrity_components.append(sa_efficiency * 0.3)
        
        # Component 3: Pore structure stability (0-0.3)
        if micro_frac > 0.5:
            pore_stability = 0.8  # Microporous materials are stable
        elif ordered:
            pore_stability = 0.9  # Ordered mesopores are very stable
        elif d_pore < 5:
            pore_stability = 0.7  # Small mesopores are relatively stable
        else:
            pore_stability = 0.5  # Large pores are less stable
        
        integrity_components.append(pore_stability * 0.3)
        
        structural_integrity = np.mean(integrity_components)
        
        # ====================================================================
        # 2. COMPOSITE CLASSIFICATION
        # ====================================================================
        
        if s_bet > 1000 and micro_frac > 0.7:
            composite_class = "Type I: Microporous Carbon Network"
            material_family = "Carbonaceous"
        elif s_bet > 600 and ordered and cryst > 0.7:
            composite_class = "Type II: Ordered Mesoporous Crystalline"
            material_family = "Ordered Mesoporous"
        elif s_bet > 800 and cryst > 0.8:
            composite_class = "Type III: Hierarchical Porous Crystalline"
            material_family = "Hierarchical Crystalline"
        elif micro_frac > 0.6:
            composite_class = "Type IV: Microporous Framework"
            material_family = "Microporous"
        elif ordered:
            composite_class = "Type V: Ordered Mesoporous"
            material_family = "Ordered Mesoporous"
        elif cryst > 0.7 and s_bet > 100:
            composite_class = "Type VI: Crystalline Porous"
            material_family = "Crystalline Porous"
        elif s_bet > 500:
            composite_class = "Type VII: High-Surface-Area Mesoporous"
            material_family = "Mesoporous"
        elif cryst > 0.5:
            composite_class = "Type VIII: Semi-crystalline Porous"
            material_family = "Semi-crystalline"
        else:
            composite_class = "Type IX: Complex Composite"
            material_family = "Composite"
        
        # ====================================================================
        # 3. DOMINANT FEATURE IDENTIFICATION
        # ====================================================================
        
        if s_bet > 1000 and micro_frac > 0.7:
            dominant_feature = "Microporous carbon network with high surface area"
        elif ordered and d_spacing:
            dominant_feature = f"Ordered mesostructure (d = {d_spacing:.1f} nm)"
        elif cryst_size and cryst_size < 10:
            dominant_feature = f"Nanocrystalline structure ({cryst_size:.1f} nm crystallites)"
        elif s_bet > 800:
            dominant_feature = "High surface area porous network"
        elif cryst > 0.8:
            dominant_feature = "Well-developed crystalline phase"
        else:
            dominant_feature = "Mixed porous-crystalline morphology"
        
        # ====================================================================
        # 4. CONFIDENCE SCORE CALCULATION
        # ====================================================================
        
        confidence_factors = []
        
        # BET data quality
        if bet_result.bet_r2 > 0.999:
            confidence_factors.append(0.9)
        elif bet_result.bet_r2 > 0.995:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # XRD data quality
        if len(xrd_result.peak_positions) >= 5:
            confidence_factors.append(0.9)
        elif len(xrd_result.peak_positions) >= 3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Data consistency
        if (s_bet > 100 and cryst < 0.3) or (s_bet < 50 and cryst > 0.7):
            # Expected correlation: high surface area often with lower crystallinity
            confidence_factors.append(0.8)
        
        confidence_score = np.mean(confidence_factors) if confidence_factors else 0.6
        
        # ====================================================================
        # 5. JOURNAL RECOMMENDATIONS
        # ====================================================================
        
        if "Ordered Mesoporous" in composite_class:
            journals = [
                "Chemistry of Materials",
                "Microporous and Mesoporous Materials",
                "Journal of Materials Chemistry A",
                "ACS Applied Materials & Interfaces"
            ]
        elif "Microporous" in composite_class and s_bet > 1000:
            journals = [
                "Carbon",
                "Advanced Functional Materials",
                "ACS Nano",
                "Journal of Materials Chemistry A"
            ]
        elif "Crystalline" in composite_class and cryst > 0.8:
            journals = [
                "Journal of the American Chemical Society",
                "Angewandte Chemie",
                "Chemistry of Materials",
                "Crystal Growth & Design"
            ]
        else:
            journals = [
                "Materials Chemistry and Physics",
                "Journal of Porous Materials",
                "Materials Today Communications",
                "Applied Surface Science"
            ]
        
        # ====================================================================
        # 6. CHARACTERIZATION TECHNIQUES RECOMMENDATION
        # ====================================================================
        
        techniques = ["BET", "XRD"]  # Already performed
        
        if pore_wall_thickness and pore_wall_thickness < 5:
            techniques.append("TEM (for pore wall imaging)")
        
        if ordered:
            techniques.append("SAXS (for detailed structure analysis)")
        
        if micro_frac > 0.3:
            techniques.append("CO₂ adsorption (for ultramicropores)")
        
        if cryst_size and cryst_size < 20:
            techniques.append("HR-TEM (for crystallite imaging)")
        
        return MorphologyFusionResult(
            composite_classification=composite_class,
            dominant_feature=dominant_feature,
            surface_to_volume_ratio=surface_to_volume,
            pore_wall_thickness=pore_wall_thickness,
            structural_integrity=structural_integrity,
            confidence_score=confidence_score,
            material_family=material_family,
            journal_recommendations=journals,
            characterization_techniques=techniques
        )

# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================
class DataExtractor:
    """Professional data extraction for scientific instruments"""
    
    @staticmethod
    def extract_bet_data(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                        Optional[np.ndarray], Optional[np.ndarray], str]:
        """Extract BET data from various file formats"""
        try:
            filename = file.name.lower()
            
            # Read file
            if filename.endswith('.csv'):
                df = pd.read_csv(file, header=None)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl', header=None)
            elif filename.endswith('.xls'):
                df = pd.read_excel(file, engine='xlrd', header=None)
            else:
                # Try as text
                content = file.read().decode('utf-8', errors='ignore')
                file.seek(0)
                for delimiter in [',', '\t', ';', ' ']:
                    try:
                        df = pd.read_csv(io.StringIO(content), delimiter=delimiter, header=None)
                        break
                    except:
                        continue
            
            # ================================================================
            # METHOD 1: ASAP 2420 format (Micromeritics)
            # ================================================================
            p_ads, q_ads = [], []
            p_des, q_des = [], []
            
            # ASAP columns: Adsorption (L=11, M=12), Desorption (N=13, O=14)
            for i in range(28, min(200, len(df))):
                try:
                    # Adsorption
                    if df.shape[1] > 12:
                        p_val = df.iloc[i, 11]
                        q_val = df.iloc[i, 12]
                        if pd.notna(p_val) and pd.notna(q_val):
                            p_float = float(p_val)
                            q_float = float(q_val)
                            if 0.001 < p_float < 0.999 and q_float > 0:
                                p_ads.append(p_float)
                                q_ads.append(q_float)
                    
                    # Desorption
                    if df.shape[1] > 14:
                        p_val_des = df.iloc[i, 13]
                        q_val_des = df.iloc[i, 14]
                        if pd.notna(p_val_des) and pd.notna(q_val_des):
                            p_des_float = float(p_val_des)
                            q_des_float = float(q_val_des)
                            if 0.001 < p_des_float < 0.999 and q_des_float > 0:
                                p_des.append(p_des_float)
                                q_des.append(q_des_float)
                except:
                    continue
            
            # If ASAP format found sufficient points
            if len(p_ads) >= 10:
                return (np.array(p_ads), np.array(q_ads), 
                       np.array(p_des) if p_des else None,
                       np.array(q_des) if q_des else None,
                       "ASAP 2420 format detected")
            
            # ================================================================
            # METHOD 2: Auto-detection
            # ================================================================
            p_ads, q_ads = [], []
            p_des, q_des = [], []
            
            # Convert to numeric
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            
            # Find columns with pressure-like data
            for col in range(min(10, df_numeric.shape[1])):
                col_data = df_numeric.iloc[:, col].dropna()
                if len(col_data) > 10:
                    # Check if values are in pressure range
                    sample = col_data.values[:20]
                    if np.all((sample >= 0) & (sample <= 1.1)):
                        # This is pressure, next column might be quantity
                        if col + 1 < df_numeric.shape[1]:
                            q_data = df_numeric.iloc[:, col + 1].dropna()
                            if len(q_data) > 10:
                                p_ads = col_data.values
                                q_ads = q_data.values
                                
                                # Check for desorption
                                if col + 3 < df_numeric.shape[1]:
                                    p_des_data = df_numeric.iloc[:, col + 2].dropna()
                                    q_des_data = df_numeric.iloc[:, col + 3].dropna()
                                    if len(p_des_data) > 5:
                                        p_des = p_des_data.values
                                        q_des = q_des_data.values
                                break
            
            if len(p_ads) >= 5:
                return (np.array(p_ads), np.array(q_ads), 
                       np.array(p_des) if len(p_des) > 0 else None,
                       np.array(q_des) if len(q_des) > 0 else None,
                       "Auto-detected format")
            
            # ================================================================
            # METHOD 3: Simple extraction
            # ================================================================
            df_clean = df.apply(pd.to_numeric, errors='coerce')
            df_clean = df_clean.dropna(axis=1, how='all')
            
            if df_clean.shape[1] >= 2:
                p_ads = df_clean.iloc[:, 0].dropna().values
                q_ads = df_clean.iloc[:, 1].dropna().values
                
                if df_clean.shape[1] >= 4:
                    p_des = df_clean.iloc[:, 2].dropna().values
                    q_des = df_clean.iloc[:, 3].dropna().values
                
                if len(p_ads) >= 5:
                    return (p_ads, q_ads, 
                           p_des if p_des is not None and len(p_des) > 0 else None,
                           q_des if q_des is not None and len(q_des) > 0 else None,
                           "Simple column extraction")
            
            return None, None, None, None, f"Insufficient data. Found {len(p_ads)} points (need ≥5)"
            
        except Exception as e:
            return None, None, None, None, f"Extraction error: {str(e)}"
    
    @staticmethod
    def extract_xrd_data(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Extract XRD data"""
        try:
            content = file.read().decode('utf-8', errors='ignore')
            file.seek(0)
            
            # Try different delimiters
            for delimiter in [',', '\t', ';', ' ', '|']:
                try:
                    if delimiter == ' ':
                        df = pd.read_csv(io.StringIO(content), delim_whitespace=True, header=None)
                    else:
                        df = pd.read_csv(io.StringIO(content), delimiter=delimiter, header=None)
                    
                    if df.shape[1] >= 2:
                        break
                except:
                    continue
            
            # Clean data
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            
            if df.shape[0] < 10:
                return None, None, f"Insufficient data points: {df.shape[0]} (need ≥10)"
            
            two_theta = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
            
            return two_theta, intensity, "Success"
            
        except Exception as e:
            return None, None, f"XRD extraction error: {str(e)}"

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================
class ScientificVisualizer:
    """Publication-quality visualization"""
    
    @staticmethod
    def create_comprehensive_figure(bet_result: BETAnalysisResult, 
                                   xrd_result: XRDAnalysisResult,
                                   fusion_result: MorphologyFusionResult,
                                   p_ads: np.ndarray, q_ads: np.ndarray,
                                   p_des: Optional[np.ndarray], q_des: Optional[np.ndarray],
                                   two_theta: np.ndarray, intensity: np.ndarray) -> plt.Figure:
        """Create comprehensive multi-panel figure for publication"""
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Panel A: BET isotherm
        ax1 = fig.add_subplot(gs[0, 0])
        ScientificVisualizer._plot_isotherm(ax1, p_ads, q_ads, p_des, q_des, bet_result)
        ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        # Panel B: BET linear plot
        ax2 = fig.add_subplot(gs[0, 1])
        ScientificVisualizer._plot_bet_linear(ax2, p_ads, q_ads, bet_result)
        ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        # Panel C: PSD if available
        ax3 = fig.add_subplot(gs[0, 2])
        if hasattr(bet_result, 'psd_available') and bet_result.psd_available:
            ScientificVisualizer._plot_psd(ax3, bet_result)
        else:
            ax3.text(0.5, 0.5, 'PSD not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        # Panel D: XRD pattern
        ax4 = fig.add_subplot(gs[1, :])
        ScientificVisualizer._plot_xrd(ax4, two_theta, intensity, xrd_result)
        ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        # Panel E: Morphology radar
        ax5 = fig.add_subplot(gs[2, :], projection='polar')
        ScientificVisualizer._plot_morphology_radar(ax5, bet_result, xrd_result, fusion_result)
        ax5.text(0.02, 0.98, 'E', transform=ax5.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        # Panel F: Summary table
        ax6 = fig.add_subplot(gs[3, :])
        ScientificVisualizer._plot_summary_table(ax6, bet_result, xrd_result, fusion_result)
        ax6.text(0.02, 0.98, 'F', transform=ax6.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        ax6.axis('off')
        
        plt.suptitle("Comprehensive BET-XRD Morphology Analysis", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    @staticmethod
    def _plot_isotherm(ax, p_ads, q_ads, p_des, q_des, bet_result):
        """Plot adsorption-desorption isotherm"""
        ax.plot(p_ads, q_ads, 'o-', linewidth=2, markersize=4, 
                label='Adsorption', color='#1f77b4')
        
        if p_des is not None and q_des is not None:
            ax.plot(p_des, q_des, 's--', linewidth=2, markersize=4,
                   label='Desorption', color='#ff7f0e')
        
        ax.set_xlabel('Relative Pressure (P/P₀)', fontsize=11)
        ax.set_ylabel('Quantity Adsorbed (mmol/g)', fontsize=11)
        ax.set_title('Physisorption Isotherm', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add BET parameters
        text = f"$S_{{BET}}$ = {bet_result.surface_area:.0f} m²/g\n"
        text += f"C = {bet_result.c_constant:.0f}\n"
        text += f"Type: {bet_result.hysteresis_type}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    @staticmethod
    def _plot_bet_linear(ax, p_ads, q_ads, bet_result):
        """Plot BET linear region"""
        # Select BET linear region
        p_min, p_max = bet_result.linear_range
        mask = (p_ads >= p_min) & (p_ads <= p_max)
        p_lin = p_ads[mask]
        q_lin = q_ads[mask]
        
        # BET transformation
        y = p_lin / (q_lin * (1 - p_lin))
        
        ax.plot(p_lin, y, 'o', markersize=6, color='#2ca02c')
        
        # Regression line
        p_fit = np.linspace(p_min, p_max, 100)
        slope = (y[-1] - y[0]) / (p_lin[-1] - p_lin[0])  # Simplified
        intercept = y[0] - slope * p_lin[0]
        ax.plot(p_fit, slope * p_fit + intercept, '--', color='#2ca02c', linewidth=2)
        
        ax.set_xlabel('Relative Pressure (P/P₀)', fontsize=11)
        ax.set_ylabel('p/[q(1-p)]', fontsize=11)
        ax.set_title('BET Linear Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        text = f"Linear range: {p_min:.3f}-{p_max:.3f}\n"
        text += f"R² = {bet_result.bet_r2:.4f}\n"
        text += f"n = {len(p_lin)} points"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    @staticmethod
    def _plot_psd(ax, bet_result):
        """Plot pore size distribution"""
        if not hasattr(bet_result, 'pore_diameters') or not bet_result.psd_available:
            return
        
        ax.plot(bet_result.pore_diameters, bet_result.dv_dlogd, 
                '-', linewidth=2, color='#9467bd')
        ax.fill_between(bet_result.pore_diameters, 0, bet_result.dv_dlogd,
                       alpha=0.3, color='#9467bd')
        
        ax.set_xscale('log')
        ax.set_xlabel('Pore Diameter (nm)', fontsize=11)
        ax.set_ylabel('dV/dlogD (cm³/g)', fontsize=11)
        ax.set_title('Pore Size Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        if hasattr(bet_result, 'mean_pore_diameter'):
            text = f"Mean D = {bet_result.mean_pore_diameter:.1f} nm\n"
            if hasattr(bet_result, 'peak_pore_diameter'):
                text += f"Peak D = {bet_result.peak_pore_diameter:.1f} nm"
            ax.text(0.95, 0.95, text, transform=ax.transAxes,
                   fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    @staticmethod
    def _plot_xrd(ax, two_theta, intensity, xrd_result):
        """Plot XRD pattern with peak markers"""
        ax.plot(two_theta, intensity, '-', linewidth=1.5, color='#1f77b4')
        
        # Mark peaks
        peaks = xrd_result.peak_positions
        if peaks:
            for peak in peaks:
                idx = np.argmin(np.abs(two_theta - peak))
                ax.plot(peak, intensity[idx], 'r^', markersize=8, 
                       label='Peak' if peak == peaks[0] else "")
        
        ax.set_xlabel('2θ (degrees)', fontsize=11)
        ax.set_ylabel('Intensity (normalized)', fontsize=11)
        ax.set_title('XRD Pattern', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        text = f"Crystallinity = {xrd_result.crystallinity_index:.2f}\n"
        if xrd_result.crystallite_size:
            text += f"Size = {xrd_result.crystallite_size:.1f} nm"
        if xrd_result.ordered_mesopores:
            text += "\nOrdered mesopores"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if peaks:
            ax.legend(loc='upper right', fontsize=10)
    
    @staticmethod
    def _plot_morphology_radar(ax, bet_result, xrd_result, fusion_result):
        """Create radar chart of morphology parameters"""
        categories = ['Surface Area', 'Porosity', 'Crystallinity', 
                     'Ordering', 'Stability', 'Complexity']
        
        # Normalized values (0-1)
        s_bet_norm = min(1.0, bet_result.surface_area / 2000)
        porosity_norm = min(1.0, bet_result.pore_volume * 10)
        cryst_norm = xrd_result.crystallinity_index
        ordering_norm = 0.9 if xrd_result.ordered_mesopores else 0.3
        stability_norm = fusion_result.structural_integrity
        complexity_norm = 0.8 if "Complex" in fusion_result.composite_classification else 0.4
        
        values = [s_bet_norm, porosity_norm, cryst_norm, 
                 ordering_norm, stability_norm, complexity_norm]
        
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        ax.set_title(fusion_result.composite_classification, 
                    fontsize=12, fontweight='bold', y=1.1)
    
    @staticmethod
    def _plot_summary_table(ax, bet_result, xrd_result, fusion_result):
        """Create summary table"""
        summary_text = "COMPREHENSIVE MORPHOLOGY ANALYSIS SUMMARY\n"
        summary_text += "="*60 + "\n\n"
        
        summary_text += "BET ANALYSIS RESULTS:\n"
        summary_text += f"  • Surface Area (Sᴮᴱᵀ): {bet_result.surface_area:.0f} m²/g\n"
        summary_text += f"  • Total Pore Volume: {bet_result.pore_volume:.3f} cm³/g\n"
        summary_text += f"  • Micropore Volume: {bet_result.micropore_volume:.4f} cm³/g\n"
        summary_text += f"  • External Surface: {bet_result.external_surface:.0f} m²/g\n"
        summary_text += f"  • Mean Pore Diameter: {bet_result.mean_pore_diameter:.1f} nm\n"
        summary_text += f"  • Hysteresis Type: {bet_result.hysteresis_type} (IUPAC)\n"
        summary_text += f"  • BET C Constant: {bet_result.c_constant:.0f}\n"
        summary_text += f"  • BET Regression R²: {bet_result.bet_r2:.4f}\n\n"
        
        summary_text += "XRD ANALYSIS RESULTS:\n"
        summary_text += f"  • Crystallinity Index: {xrd_result.crystallinity_index:.3f}\n"
        summary_text += f"  • Crystallite Size: {xrd_result.crystallite_size:.1f} nm\n"
        summary_text += f"  • Microstrain: {xrd_result.microstrain:.4f}\n"
        summary_text += f"  • Ordered Mesopores: {'Yes' if xrd_result.ordered_mesopores else 'No'}\n"
        if xrd_result.d_spacing:
            summary_text += f"  • d-spacing: {xrd_result.d_spacing:.2f} nm\n"
        summary_text += f"  • Symmetry: {xrd_result.symmetry_class}\n"
        summary_text += f"  • Peaks Detected: {len(xrd_result.peak_positions)}\n\n"
        
        summary_text += "MORPHOLOGY FUSION RESULTS:\n"
        summary_text += f"  • Composite Classification: {fusion_result.composite_classification}\n"
        summary_text += f"  • Dominant Feature: {fusion_result.dominant_feature}\n"
        summary_text += f"  • Material Family: {fusion_result.material_family}\n"
        summary_text += f"  • Surface/Volume Ratio: {fusion_result.surface_to_volume_ratio:.0f} m²/cm³\n"
        summary_text += f"  • Structural Integrity: {fusion_result.structural_integrity:.2f}\n"
        if fusion_result.pore_wall_thickness:
            summary_text += f"  • Pore Wall Thickness: {fusion_result.pore_wall_thickness:.1f} nm\n"
        summary_text += f"  • Analysis Confidence: {fusion_result.confidence_score:.2f}\n\n"
        
        summary_text += "RECOMMENDED JOURNALS:\n"
        for journal in fusion_result.journal_recommendations[:3]:
            summary_text += f"  • {journal}\n"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='top', linespacing=1.5)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'bet_data' not in st.session_state:
        st.session_state.bet_data = None
    if 'xrd_data' not in st.session_state:
        st.session_state.xrd_data = None
    if 'bet_result' not in st.session_state:
        st.session_state.bet_result = None
    if 'xrd_result' not in st.session_state:
        st.session_state.xrd_result = None
    if 'fusion_result' not in st.session_state:
        st.session_state.fusion_result = None
    
    # ========================================================================
    # SIDEBAR - SCIENTIFIC CONTROLS
    # ========================================================================
    with st.sidebar:
        st.title("🔬 Scientific Controls")
        
        st.markdown("---")
        st.subheader("BET Analysis Parameters")
        
        bet_method = st.selectbox(
            "BET Method",
            ["IUPAC Rouquerol", "Standard 0.05-0.35", "Automatic"],
            index=0,
            help="IUPAC Rouquerol criteria for automatic range selection"
        )
        
        st.subheader("XRD Analysis Parameters")
        
        xrd_wavelength = st.selectbox(
            "X-ray Wavelength",
            ["Cu Kα (0.15406 nm)", "Mo Kα (0.07107 nm)", "Co Kα (0.17902 nm)"],
            index=0
        )
        
        wavelength = 0.15406
        if "Mo" in xrd_wavelength:
            wavelength = 0.07107
        elif "Co" in xrd_wavelength:
            wavelength = 0.17902
        
        st.markdown("---")
        st.subheader("Export Settings")
        
        export_format = st.selectbox(
            "Figure Format",
            ["PNG (300 DPI)", "PDF (Vector)", "SVG (Vector)"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("Scientific Information")
        
        st.info("""
        **References:**
        1. Rouquerol et al., *Pure Appl. Chem.*, 1994, 66, 1739
        2. Thommes et al., *Pure Appl. Chem.*, 2015, 87, 1051
        3. Klug & Alexander, *X-ray Diffraction Procedures*, 1974
        """)
        
        st.caption(f"Python {sys.version.split()[0]}")
        st.caption("BET-XRD Morphology Analyzer v3.0")
    
    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    st.title("🧪 BET–XRD Morphology Analyzer v3.0")
    st.markdown("""
    **Scientific-grade analysis for porous materials characterization**  
    *Integrated physisorption and XRD analysis for comprehensive morphology determination*
    """)
    
    # ========================================================================
    # FILE UPLOAD SECTION
    # ========================================================================
    st.header("📁 Experimental Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physisorption Data")
        bet_file = st.file_uploader(
            "Upload BET isotherm file",
            type=["xls", "xlsx", "csv", "txt"],
            help="ASAP 2420 format recommended. Adsorption in columns L&M, desorption in N&O"
        )
        
        if bet_file:
            st.success(f"✅ BET file uploaded: {bet_file.name}")
            
            # Show file preview
            with st.expander("Preview BET file"):
                try:
                    if bet_file.name.endswith('.xls'):
                        preview_df = pd.read_excel(bet_file, engine='xlrd', nrows=10)
                    elif bet_file.name.endswith('.xlsx'):
                        preview_df = pd.read_excel(bet_file, engine='openpyxl', nrows=10)
                    else:
                        preview_df = pd.read_csv(bet_file, nrows=10)
                    
                    st.dataframe(preview_df)
                    st.write(f"Shape: {preview_df.shape}")
                except Exception as e:
                    st.error(f"Preview error: {e}")
    
    with col2:
        st.subheader("XRD Data")
        xrd_file = st.file_uploader(
            "Upload XRD pattern file",
            type=["csv", "txt", "xy", "dat"],
            help="Two-column format: 2θ (degrees) and Intensity"
        )
        
        if xrd_file:
            st.success(f"✅ XRD file uploaded: {xrd_file.name}")
            
            # Show file preview
            with st.expander("Preview XRD file"):
                try:
                    content = xrd_file.read().decode('utf-8', errors='ignore')
                    xrd_file.seek(0)
                    lines = content.split('\n')[:10]
                    for line in lines:
                        st.code(line)
                except:
                    st.write("Binary file - cannot preview")
    
    # ========================================================================
    # ANALYSIS SECTION
    # ========================================================================
    st.header("🔬 Scientific Analysis")
    
    analyze_col1, analyze_col2 = st.columns([3, 1])
    
    with analyze_col1:
        analysis_name = st.text_input(
            "Analysis Name",
            value="Sample Analysis",
            help="Name for this analysis session"
        )
    
    with analyze_col2:
        analyze_button = st.button(
            "🚀 RUN COMPREHENSIVE ANALYSIS",
            type="primary",
            use_container_width=True,
            disabled=(bet_file is None and xrd_file is None)
        )
    
    if analyze_button:
        st.session_state.analysis_complete = False
        
        # Clear previous results
        st.session_state.bet_data = None
        st.session_state.xrd_data = None
        st.session_state.bet_result = None
        st.session_state.xrd_result = None
        st.session_state.fusion_result = None
        
        # Create progress bars
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Extract data
            status_text.text("📥 Extracting experimental data...")
            progress_bar.progress(10)
            
            bet_raw_data = None
            xrd_raw_data = None
            
            if bet_file:
                p_ads, q_ads, p_des, q_des, bet_msg = DataExtractor.extract_bet_data(bet_file)
                if p_ads is not None:
                    st.session_state.bet_data = (p_ads, q_ads, p_des, q_des)
                    st.success(f"✅ BET data extracted: {len(p_ads)} points")
                else:
                    st.error(f"❌ BET extraction failed: {bet_msg}")
            
            if xrd_file:
                two_theta, intensity, xrd_msg = DataExtractor.extract_xrd_data(xrd_file)
                if two_theta is not None:
                    st.session_state.xrd_data = (two_theta, intensity)
                    st.success(f"✅ XRD data extracted: {len(two_theta)} points")
                else:
                    st.error(f"❌ XRD extraction failed: {xrd_msg}")
            
            progress_bar.progress(30)
            
            # Step 2: BET analysis
            if st.session_state.bet_data:
                status_text.text("📊 Performing BET analysis...")
                p_ads, q_ads, p_des, q_des = st.session_state.bet_data
                
                try:
                    bet_analyzer = IUPACBETAnalyzer(p_ads, q_ads, p_des, q_des)
                    st.session_state.bet_result = bet_analyzer.full_analysis()
                    st.success("✅ BET analysis complete")
                except Exception as e:
                    st.error(f"❌ BET analysis error: {str(e)}")
            
            progress_bar.progress(50)
            
            # Step 3: XRD analysis
            if st.session_state.xrd_data:
                status_text.text("📈 Performing XRD analysis...")
                two_theta, intensity = st.session_state.xrd_data
                
                try:
                    xrd_analyzer = AdvancedXRDAnalyzer(wavelength)
                    st.session_state.xrd_result = xrd_analyzer.analyze(two_theta, intensity)
                    st.success("✅ XRD analysis complete")
                except Exception as e:
                    st.error(f"❌ XRD analysis error: {str(e)}")
            
            progress_bar.progress(70)
            
            # Step 4: Morphology fusion
            if st.session_state.bet_result and st.session_state.xrd_result:
                status_text.text("🧬 Fusing morphology data...")
                
                try:
                    st.session_state.fusion_result = MorphologyFusionEngine.fuse(
                        st.session_state.bet_result,
                        st.session_state.xrd_result
                    )
                    st.success("✅ Morphology fusion complete")
                except Exception as e:
                    st.error(f"❌ Fusion error: {str(e)}")
            
            progress_bar.progress(90)
            
            # Step 5: Generate visualizations
            status_text.text("🎨 Generating scientific figures...")
            progress_bar.progress(100)
            
            st.session_state.analysis_complete = True
            status_text.text("✅ Analysis complete!")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    # ========================================================================
    # RESULTS DISPLAY
    # ========================================================================
    if st.session_state.analysis_complete:
        st.header("📊 Scientific Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Comprehensive View", 
            "🔬 BET Analysis", 
            "📉 XRD Analysis", 
            "🧬 Morphology", 
            "📤 Export"
        ])
        
        with tab1:
            # Generate comprehensive figure
            if (st.session_state.bet_result and st.session_state.xrd_result and 
                st.session_state.fusion_result and st.session_state.bet_data):
                
                p_ads, q_ads, p_des, q_des = st.session_state.bet_data
                two_theta, intensity = st.session_state.xrd_data
                
                fig = ScientificVisualizer.create_comprehensive_figure(
                    st.session_state.bet_result,
                    st.session_state.xrd_result,
                    st.session_state.fusion_result,
                    p_ads, q_ads, p_des, q_des,
                    two_theta, intensity
                )
                
                st.pyplot(fig)
                
                st.caption("**Figure 1.** Comprehensive morphology analysis. (A) Physisorption isotherm, (B) BET linear plot, (C) Pore size distribution, (D) XRD pattern, (E) Morphology radar chart, (F) Analysis summary.")
            else:
                st.info("Complete analysis results not available")
        
        with tab2:
            if st.session_state.bet_result:
                bet = st.session_state.bet_result
                
                # Display BET metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Surface Area", f"{bet.surface_area:.0f} m²/g", 
                             delta=None, help="BET surface area")
                with col2:
                    st.metric("Pore Volume", f"{bet.pore_volume:.3f} cm³/g",
                             help="Total pore volume")
                with col3:
                    st.metric("C Constant", f"{bet.c_constant:.0f}",
                             help="BET C constant")
                with col4:
                    st.metric("BET R²", f"{bet.bet_r2:.4f}",
                             help="BET linear regression quality")
                
                # Additional metrics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric("Micropore Volume", f"{bet.micropore_volume:.4f} cm³/g")
                with col6:
                    st.metric("External Surface", f"{bet.external_surface:.0f} m²/g")
                with col7:
                    st.metric("Mean Pore D", f"{bet.mean_pore_diameter:.1f} nm")
                with col8:
                    st.metric("Hysteresis", bet.hysteresis_type)
                
                # Plot isotherm
                if st.session_state.bet_data:
                    p_ads, q_ads, p_des, q_des = st.session_state.bet_data
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Isotherm
                    ax1.plot(p_ads, q_ads, 'o-', label='Adsorption')
                    if p_des is not None and q_des is not None:
                        ax1.plot(p_des, q_des, 's--', label='Desorption')
                    ax1.set_xlabel('P/P₀')
                    ax1.set_ylabel('Quantity (mmol/g)')
                    ax1.set_title('Adsorption-Desorption Isotherm')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # BET linear plot
                    p_min, p_max = bet.linear_range
                    mask = (p_ads >= p_min) & (p_ads <= p_max)
                    p_lin = p_ads[mask]
                    q_lin = q_ads[mask]
                    y = p_lin / (q_lin * (1 - p_lin))
                    
                    ax2.plot(p_lin, y, 'o')
                    slope = (y[-1] - y[0]) / (p_lin[-1] - p_lin[0])
                    intercept = y[0] - slope * p_lin[0]
                    p_fit = np.linspace(p_min, p_max, 100)
                    ax2.plot(p_fit, slope * p_fit + intercept, '--')
                    ax2.set_xlabel('P/P₀')
                    ax2.set_ylabel('p/[q(1-p)]')
                    ax2.set_title('BET Linear Region')
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Detailed BET parameters
                with st.expander("Detailed BET Parameters"):
                    st.json({
                        "BET Analysis": {
                            "Surface Area (m²/g)": float(bet.surface_area),
                            "Monolayer Capacity (mmol/g)": float(bet.monolayer_capacity),
                            "C Constant": float(bet.c_constant),
                            "BET R²": float(bet.bet_r2),
                            "Linear Range (P/P₀)": [float(bet.linear_range[0]), float(bet.linear_range[1])]
                        },
                        "Porosity Analysis": {
                            "Total Pore Volume (cm³/g)": float(bet.pore_volume),
                            "Micropore Volume (cm³/g)": float(bet.micropore_volume),
                            "External Surface Area (m²/g)": float(bet.external_surface),
                            "Mean Pore Diameter (nm)": float(bet.mean_pore_diameter),
                            "t-plot R²": float(bet.t_plot_r2)
                        },
                        "Hysteresis Analysis": {
                            "Type": bet.hysteresis_type,
                            "IUPAC Classification": bet.isotherm_type
                        }
                    })
            else:
                st.info("BET analysis results not available")
        
        with tab3:
            if st.session_state.xrd_result:
                xrd = st.session_state.xrd_result
                
                # Display XRD metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Crystallinity", f"{xrd.crystallinity_index:.3f}")
                with col2:
                    size = xrd.crystallite_size if xrd.crystallite_size else 0
                    st.metric("Crystallite Size", f"{size:.1f} nm")
                with col3:
                    st.metric("Microstrain", f"{xrd.microstrain:.4f}")
                with col4:
                    st.metric("Ordered", "Yes" if xrd.ordered_mesopores else "No")
                
                # Plot XRD pattern
                if st.session_state.xrd_data:
                    two_theta, intensity = st.session_state.xrd_data
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(two_theta, intensity, '-', linewidth=1.5)
                    
                    # Mark peaks
                    if xrd.peak_positions:
                        for peak in xrd.peak_positions:
                            idx = np.argmin(np.abs(two_theta - peak))
                            ax.plot(peak, intensity[idx], 'r^', markersize=8)
                    
                    ax.set_xlabel('2θ (degrees)')
                    ax.set_ylabel('Intensity (normalized)')
                    ax.set_title('XRD Pattern')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Detailed XRD parameters
                with st.expander("Detailed XRD Parameters"):
                    st.json({
                        "Crystallinity Analysis": {
                            "Crystallinity Index": float(xrd.crystallinity_index),
                            "Peaks Detected": len(xrd.peak_positions),
                            "Primary Peak (2θ)": float(xrd.primary_peak_2theta) if xrd.primary_peak_2theta else None
                        },
                        "Crystallite Analysis": {
                            "Size (Scherrer, nm)": float(xrd.crystallite_size) if xrd.crystallite_size else None,
                            "Microstrain": float(xrd.microstrain),
                            "Dislocation Density (m⁻²)": float(xrd.dislocation_density),
                            "Lattice Strain": float(xrd.lattice_strain)
                        },
                        "Mesostructure Analysis": {
                            "Ordered Mesopores": xrd.ordered_mesopores,
                            "d-spacing (nm)": float(xrd.d_spacing) if xrd.d_spacing else None,
                            "Symmetry Class": xrd.symmetry_class
                        },
                        "Peak Analysis": {
                            "FWHM Values": [float(f) for f in xrd.fwhm_values],
                            "Peak Positions (2θ)": [float(p) for p in xrd.peak_positions]
                        }
                    })
            else:
                st.info("XRD analysis results not available")
        
        with tab4:
            if st.session_state.fusion_result:
                fusion = st.session_state.fusion_result
                
                st.subheader("🧬 Integrated Morphology Analysis")
                
                # Display fusion metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Classification", fusion.composite_classification)
                with col2:
                    st.metric("Material Family", fusion.material_family)
                with col3:
                    st.metric("Confidence", f"{fusion.confidence_score:.2f}")
                
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric("Surface/Volume", f"{fusion.surface_to_volume_ratio:.0f} m²/cm³")
                with col5:
                    st.metric("Integrity", f"{fusion.structural_integrity:.2f}")
                with col6:
                    if fusion.pore_wall_thickness:
                        st.metric("Wall Thickness", f"{fusion.pore_wall_thickness:.1f} nm")
                    else:
                        st.metric("Wall Thickness", "N/A")
                
                st.subheader("Dominant Morphological Feature")
                st.info(fusion.dominant_feature)
                
                st.subheader("📝 Journal Recommendations")
                for journal in fusion.journal_recommendations:
                    st.write(f"• **{journal}**")
                
                st.subheader("🔬 Recommended Characterization Techniques")
                for technique in fusion.characterization_techniques:
                    st.write(f"• {technique}")
                
                # Scientific interpretation
                st.subheader("🧪 Scientific Interpretation")
                
                interpretation = """
                **Key Scientific Insights:**
                
                1. **Complementary Information**: The integrated analysis reveals complementary 
                information from BET (surface area, porosity) and XRD (crystallinity, structure).
                
                2. **Structure-Property Relationships**: The morphology fusion algorithm 
                identifies key relationships between porous structure and crystalline properties.
                
                3. **Material Design Guidance**: Results provide guidance for targeted material 
                design and optimization for specific applications.
                
                4. **Publication Readiness**: Analysis follows IUPAC standards and includes 
                comprehensive error analysis for scientific publication.
                """
                
                st.markdown(interpretation)
                
            else:
                st.info("Morphology fusion results not available")
        
        with tab5:
            st.header("📤 Export Results")
            
            # Export data
            export_data = {
                "Analysis Metadata": {
                    "name": analysis_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "version": "BET-XRD Analyzer v3.0",
                    "references": [
                        "Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739",
                        "Thommes et al., Pure Appl. Chem., 2015, 87, 1051"
                    ]
                },
                "BET Results": st.session_state.bet_result.__dict__ if st.session_state.bet_result else None,
                "XRD Results": st.session_state.xrd_result.__dict__ if st.session_state.xrd_result else None,
                "Morphology Fusion": st.session_state.fusion_result.__dict__ if st.session_state.fusion_result else None
            }
            
            # Convert to JSON
            json_str = json.dumps(export_data, indent=2, default=str)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Complete Analysis (JSON)",
                    data=json_str,
                    file_name=f"bet_xrd_analysis_{analysis_name.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Generate and export figure
                if (st.session_state.bet_result and st.session_state.xrd_result and 
                    st.session_state.fusion_result and st.session_state.bet_data):
                    
                    p_ads, q_ads, p_des, q_des = st.session_state.bet_data
                    two_theta, intensity = st.session_state.xrd_data
                    
                    fig = ScientificVisualizer.create_comprehensive_figure(
                        st.session_state.bet_result,
                        st.session_state.xrd_result,
                        st.session_state.fusion_result,
                        p_ads, q_ads, p_des, q_des,
                        two_theta, intensity
                    )
                    
                    # Save figure to buffer
                    buf = io.BytesIO()
                    format_type = export_format.split()[0].lower()
                    fig.savefig(buf, format=format_type, dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"🖼️ Download Figure ({export_format})",
                        data=buf,
                        file_name=f"morphology_analysis_{analysis_name.replace(' ', '_')}.{format_type}",
                        mime=f"image/{format_type}" if format_type != "pdf" else "application/pdf"
                    )
            
            # Citation
            st.markdown("---")
            st.subheader("📚 Citation")
            
            citation = """
            **Please cite this software in your publications:**
            
            ```
            @software{BET_XRD_Morphology_Analyzer_2024,
                author = {Your Name},
                title = {BET-XRD Morphology Analyzer v3.0},
                year = {2024},
                url = {https://your-app-url.com},
                note = {Integrated physisorption and XRD analysis for comprehensive morphology characterization}
            }
            ```
            """
            
            st.markdown(citation)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🔬 **BET-XRD Morphology Analyzer v3.0**")
        st.caption("Scientific Edition")
    
    with col2:
        st.caption("📚 **References:** IUPAC Standards, Rouquerol Criteria")
    
    with col3:
        st.caption("⚙️ **For Publication in:** Chem. Mater., Micropor. Mesopor. Mater.")

if __name__ == "__main__":
    main()
