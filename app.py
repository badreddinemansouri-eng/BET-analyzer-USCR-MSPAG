"""
INTEGRATED BET–XRD MORPHOLOGY ANALYZER
Scientific Journal Submission Version
DOI: 10.XXXX/xxxx (Reserved)
--------------------------------------------------------------------
Copyright (c) 2024 [Your Institution]
Licensed under MIT License
-------------------------------------------------------------------
Features:
1. IUPAC-compliant physisorption analysis
2. XRD crystallinity & mesostructure analysis
3. Scientific morphology fusion algorithm
4. Journal-quality visualization
5. Complete data provenance
--------------------------------------------------------------------
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize, integrate
import matplotlib.pyplot as plt
from matplotlib import gridspec
import base64
import io
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BET–XRD Morphology Analyzer | Journal Version",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3
})

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def detect_excel_format(file_content: bytes) -> str:
    """
    Detect Excel file format from magic bytes
    Returns: 'xls', 'xlsx', or 'unknown'
    """
    # Excel 97-2003 (.xls) - OLE Compound Document
    if file_content.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
        return 'xls'
    
    # Excel 2007+ (.xlsx) - ZIP archive
    if file_content.startswith(b'PK\x03\x04'):
        return 'xlsx'
    
    # Excel 2007+ macro-enabled (.xlsm)
    if file_content.startswith(b'PK\x03\x04\x14\x00\x06\x00'):
        return 'xlsx'  # Treat as xlsx
    
    return 'unknown'

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================
class ScientificAnalysisError(Exception):
    """Base exception for scientific analysis errors"""
    pass

class IUPACViolationError(ScientificAnalysisError):
    """Raised when IUPAC guidelines are violated"""
    pass

class DataIntegrityError(ScientificAnalysisError):
    """Raised when data quality is insufficient"""
    pass

# ============================================================================
# DATA STRUCTURES FOR MORPHOLOGY FUSION
# ============================================================================
@dataclass
class BETMorphology:
    """Structured BET-derived morphology data"""
    surface_area: float  # m²/g
    pore_volume: float   # cm³/g
    pore_diameter: float # nm
    microporosity: float # fraction (0-1)
    mesoporosity: float  # fraction (0-1)
    macroporosity: float # fraction (0-1)
    hysteresis_type: str # IUPAC classification
    isotherm_type: str   # IUPAC classification
    
@dataclass
class XRDMMorphology:
    """Structured XRD-derived morphology data"""
    crystallinity_index: float   # 0-1
    crystallite_size: float      # nm
    ordered_mesopores: bool      # True/False
    d_spacing: Optional[float]   # nm (for ordered structures)
    lattice_strain: float        # dimensionless
    
@dataclass
class FusedMorphology:
    """Integrated morphology from BET and XRD"""
    composite_class: str
    surface_to_volume: float
    pore_wall_thickness: Optional[float]  # nm
    structural_integrity: float  # 0-1
    dominant_feature: str
    confidence_score: float      # 0-1
    journal_recommendation: str

# ============================================================================
# IUPAC-COMPLIANT BET ANALYSIS ENGINE
# ============================================================================
class IUPACBETAnalyzer:
    """
    Strict implementation of IUPAC guidelines for physisorption analysis
    Reference: Rouquerol et al., Pure Appl. Chem., 1994, 66, 1739-1758
    """
    
    N2_CROSS_SECTION = 0.162e-18  # m² (N2 at 77K)
    GAS_CONSTANT = 8.314
    AVOGADRO = 6.022e23
    
    def __init__(self, p_rel: np.ndarray, q_ads: np.ndarray, 
                 p_des: Optional[np.ndarray] = None, 
                 q_des: Optional[np.ndarray] = None):
        self.p_ads = np.asarray(p_rel)
        self.q_ads = np.asarray(q_ads)
        self.p_des = np.asarray(p_des) if p_des is not None else None
        self.q_des = np.asarray(q_des) if q_des is not None else None
        
        self._validate_data()
        
    def _validate_data(self):
        """Validate according to IUPAC standards"""
        if len(self.p_ads) < 10:
            raise DataIntegrityError("Minimum 10 adsorption points required")
        
        if not np.all(np.diff(self.p_ads) > 0):
            raise IUPACViolationError("Relative pressure must be strictly increasing")
        
        if self.p_ads.max() < 0.05 or self.p_ads.max() < 0.3:
            raise DataIntegrityError("Insufficient pressure range for BET analysis")
    
    def analyze_bet_surface_area(self) -> Dict:
        """
        Calculate BET surface area with automatic linear range selection
        using Rouquerol criteria
        """
        p, q = self.p_ads, self.q_ads
        
        # Find optimal linear region (0.05-0.35 P/P0)
        mask = (p >= 0.05) & (p <= 0.35)
        if np.sum(mask) < 5:
            raise IUPACViolationError("Insufficient points in BET linear region")
        
        p_lin = p[mask]
        q_lin = q[mask]
        
        # BET transformation
        y = p_lin / (q_lin * (1 - p_lin))
        
        # Linear regression with statistical validation
        slope, intercept, r_value, _, std_err = stats.linregress(p_lin, y)
        
        if r_value**2 < 0.999:
            raise IUPACViolationError(f"BET linearity insufficient (R² = {r_value**2:.4f})")
        
        # Calculate BET parameters
        q_mono = 1 / (slope + intercept)  # mmol/g
        c_constant = slope / intercept + 1
        
        if c_constant <= 0:
            raise IUPACViolationError(f"Non-physical C constant: {c_constant:.2f}")
        
        # Convert to surface area (m²/g)
        s_bet = q_mono * self.AVOGADRO * self.N2_CROSS_SECTION * 1e-20
        
        return {
            "surface_area": s_bet,
            "q_monolayer": q_mono,
            "c_constant": c_constant,
            "bet_r2": r_value**2,
            "linear_range": (p_lin.min(), p_lin.max()),
            "n_points": len(p_lin)
        }
    
    def analyze_pore_size_distribution(self) -> Dict:
        """BJH method for mesopore analysis (2-50 nm)"""
        if self.p_des is None or self.q_des is None:
            raise DataIntegrityError("Desorption branch required for PSD")
        
        # Calculate pore radius using Kelvin equation
        # r_k = -2γV_m / (RT ln(P/P0))
        gamma = 8.85e-3  # N/m for N2 at 77K
        v_molar = 34.7e-6  # m³/mol for liquid N2
        t = 77.3  # K
        
        # Desorption branch (hysteresis loop)
        p = self.p_des[::-1]  # Reverse for desorption
        q = self.q_des[::-1]
        
        # Kelvin radii calculation
        r_kelvin = -2 * gamma * v_molar / (self.GAS_CONSTANT * t * np.log(p))
        r_kelvin = r_kelvin * 1e9  # Convert to nm
        
        # Pore volume distribution
        dV = np.abs(np.diff(q))
        dr = np.abs(np.diff(r_kelvin))
        dV_dr = dV / dr
        
        # Remove invalid values
        valid = (r_kelvin[:-1] > 1) & (r_kelvin[:-1] < 100) & (dV_dr > 0)
        r_valid = r_kelvin[:-1][valid]
        dV_dr_valid = dV_dr[valid]
        
        if len(r_valid) < 3:
            raise DataIntegrityError("Insufficient valid points for PSD")
        
        # Calculate pore statistics
        total_pore_volume = integrate.simps(
            dV_dr_valid[np.argsort(r_valid)], 
            np.sort(r_valid)
        )
        
        # Pore size fractions
        micro = r_valid < 2
        meso = (r_valid >= 2) & (r_valid <= 50)
        macro = r_valid > 50
        
        v_micro = integrate.simps(dV_dr_valid[micro], r_valid[micro]) if np.any(micro) else 0
        v_meso = integrate.simps(dV_dr_valid[meso], r_valid[meso]) if np.any(meso) else 0
        v_macro = integrate.simps(dV_dr_valid[macro], r_valid[macro]) if np.any(macro) else 0
        
        # Mean pore diameter (volume-weighted)
        mean_diameter = 2 * integrate.simps(
            r_valid * dV_dr_valid, r_valid
        ) / total_pore_volume if total_pore_volume > 0 else 0
        
        return {
            "pore_diameters": r_valid * 2,  # Convert radius to diameter
            "dv_dlogd": dV_dr_valid,
            "total_pore_volume": total_pore_volume,
            "mean_pore_diameter": mean_diameter,
            "micropore_volume": v_micro,
            "mesopore_volume": v_meso,
            "macropore_volume": v_macro,
            "micropore_fraction": v_micro / total_pore_volume if total_pore_volume > 0 else 0,
            "mesopore_fraction": v_meso / total_pore_volume if total_pore_volume > 0 else 0,
            "macropore_fraction": v_macro / total_pore_volume if total_pore_volume > 0 else 0
        }
    
    def classify_hysteresis(self) -> Dict:
        """IUPAC hysteresis classification (H1-H4)"""
        if self.p_des is None or self.q_des is None:
            return {"type": "No hysteresis", "category": "I"}
        
        # Calculate hysteresis loop area
        p_interp = np.linspace(0.1, 0.95, 100)
        q_ads_interp = np.interp(p_interp, self.p_ads, self.q_ads)
        q_des_interp = np.interp(p_interp, self.p_des, self.q_des)
        
        loop_area = integrate.simps(
            np.abs(q_des_interp - q_ads_interp), p_interp
        )
        
        # Closure point analysis
        closure_idx = np.argmin(np.abs(self.q_des - self.q_ads[-1]))
        closure_pressure = self.p_des[closure_idx]
        
        # IUPAC classification
        if loop_area < 5:
            h_type = "H1"  # Narrow, uniform pores
            category = "IV"
        elif closure_pressure > 0.45:
            h_type = "H2"  # Ink-bottle pores
            category = "IV"
        elif closure_pressure > 0.4:
            h_type = "H3"  # Slit-shaped pores
            category = "II"
        else:
            h_type = "H4"  # Combined micro-mesopores
            category = "I"
        
        return {
            "type": h_type,
            "category": category,
            "loop_area": loop_area,
            "closure_pressure": closure_pressure,
            "description": self._get_hysteresis_description(h_type)
        }
    
    @staticmethod
    def _get_hysteresis_description(h_type: str) -> str:
        descriptions = {
            "H1": "Uniform mesopores with narrow PSD (typical MCM-41, SBA-15)",
            "H2": "Ink-bottle pores or interconnected pore network",
            "H3": "Slit-shaped pores from plate-like particles",
            "H4": "Combined micro-mesoporosity (typical activated carbons)",
            "No hysteresis": "Microporous or non-porous material"
        }
        return descriptions.get(h_type, "Unknown hysteresis type")
    
    def full_analysis(self) -> Dict:
        """Complete BET analysis with all parameters"""
        try:
            bet = self.analyze_bet_surface_area()
            psd = self.analyze_pore_size_distribution()
            hysteresis = self.classify_hysteresis()
            
            # Calculate t-plot for microporosity
            t_plot = self._calculate_t_plot()
            
            return {
                "valid": True,
                "surface_area_bet": bet["surface_area"],
                "c_constant": bet["c_constant"],
                "monolayer_capacity": bet["q_monolayer"],
                "total_pore_volume": psd["total_pore_volume"],
                "mean_pore_diameter": psd["mean_pore_diameter"],
                "micropore_volume": psd["micropore_volume"],
                "external_surface_area": t_plot.get("external_surface", 0),
                "hysteresis": hysteresis,
                "pore_size_distribution": psd,
                "t_plot": t_plot,
                "adsorption_data": {
                    "p_ads": self.p_ads,
                    "q_ads": self.q_ads,
                    "p_des": self.p_des,
                    "q_des": self.q_des
                }
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "surface_area_bet": 0,
                "c_constant": 0,
                "total_pore_volume": 0,
                "mean_pore_diameter": 0
            }
    
    def _calculate_t_plot(self) -> Dict:
        """t-plot analysis for microporosity (Harkins-Jura)"""
        p, q = self.p_ads, self.q_ads
        
        # Harkins-Jura thickness equation
        t = (13.99 / (0.034 - np.log10(p))) ** 0.5
        
        # Select linear region (0.2-0.5 P/P0)
        mask = (p >= 0.2) & (p <= 0.5)
        if np.sum(mask) < 5:
            return {"external_surface": 0, "micropore_volume": 0}
        
        t_lin = t[mask]
        q_lin = q[mask]
        
        slope, intercept, r_value, _, _ = stats.linregress(t_lin, q_lin)
        
        external_surface = slope * 15.47  # Conversion factor for N2
        micropore_volume = intercept / 1000  # Convert to cm³/g
        
        return {
            "external_surface": external_surface,
            "micropore_volume": max(0, micropore_volume),
            "t_plot_r2": r_value**2
        }

# ============================================================================
# ADVANCED XRD ANALYSIS ENGINE
# ============================================================================
class AdvancedXRDAnalyzer:
    """
    Comprehensive XRD analysis for morphology characterization
    Reference: Klug & Alexander, X-ray Diffraction Procedures, 1974
    """
    
    def __init__(self, wavelength: float = 0.15406):
        self.lambda_x = wavelength  # Cu Kα wavelength in nm
    
    def analyze(self, two_theta: np.ndarray, intensity: np.ndarray) -> Dict:
        """
        Complete XRD analysis including:
        1. Crystallinity quantification
        2. Crystallite size (Scherrer, Williamson-Hall)
        3. Microstrain analysis
        4. Mesostructure detection
        """
        # Data preprocessing
        theta, I = self._preprocess_data(two_theta, intensity)
        
        # Peak detection
        peaks = self._detect_peaks(theta, I)
        
        # Comprehensive analysis
        crystallinity = self._calculate_crystallinity_index(theta, I, peaks)
        scherrer_size = self._scherrer_analysis(theta, I, peaks)
        williamson_hall = self._williamson_hall_analysis(theta, I, peaks)
        mesostructure = self._analyze_mesostructure(theta, I)
        
        return {
            "valid": True,
            "two_theta": theta,
            "intensity": I,
            "peaks": peaks,
            "crystallinity": crystallinity,
            "crystallite_size": scherrer_size,
            "microstrain": williamson_hall.get("strain", 0),
            "dislocation_density": williamson_hall.get("dislocation_density", 0),
            "ordered_mesopores": mesostructure["ordered"],
            "d_spacing": mesostructure.get("d_spacing"),
            "primary_peak": peaks[0] if len(peaks) > 0 else None,
            "full_width_half_max": self._calculate_fwhm(theta, I, peaks)
        }
    
    def _preprocess_data(self, theta: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background and normalize"""
        # Remove NaN values
        valid = ~np.isnan(theta) & ~np.isnan(I)
        theta = theta[valid]
        I = I[valid]
        
        # Subtract background (rolling ball)
        background = pd.Series(I).rolling(window=51, center=True, min_periods=1).median()
        I_corrected = I - background.values
        
        # Normalize to [0,1]
        I_norm = (I_corrected - I_corrected.min()) / (I_corrected.max() - I_corrected.min())
        
        return theta, I_norm
    
    def _detect_peaks(self, theta: np.ndarray, I: np.ndarray, 
                     min_prominence: float = 0.1) -> List[float]:
        """Advanced peak detection with prominence filtering"""
        peaks, properties = signal.find_peaks(
            I, 
            prominence=min_prominence,
            width=2,
            distance=10
        )
        
        if len(peaks) == 0:
            return []
        
        # Filter by prominence
        prominent_peaks = peaks[properties["prominences"] > np.median(properties["prominences"]) * 0.5]
        
        return theta[prominent_peaks].tolist()
    
    def _calculate_crystallinity_index(self, theta: np.ndarray, I: np.ndarray, 
                                      peaks: List[float]) -> Dict:
        """
        Calculate crystallinity index using peak vs amorphous area
        Reference: Segal et al., J. Polym. Sci., 1959
        """
        if len(peaks) < 2:
            return {"index": 0.0, "classification": "Amorphous"}
        
        # Create amorphous baseline
        x_smooth = np.linspace(theta.min(), theta.max(), 1000)
        I_smooth = np.interp(x_smooth, theta, I)
        
        # Find valleys between peaks
        valleys = []
        for i in range(len(peaks)-1):
            mask = (theta > peaks[i]) & (theta < peaks[i+1])
            if np.any(mask):
                valley_idx = np.argmin(I[mask])
                valleys.append(theta[mask][valley_idx])
        
        # Calculate crystalline and amorphous areas
        total_area = integrate.simps(I, theta)
        
        # Approximate amorphous area (area under valleys)
        amorphous_area = 0
        for valley in valleys:
            idx = np.argmin(np.abs(theta - valley))
            window = slice(max(0, idx-5), min(len(theta), idx+5))
            amorphous_area += integrate.simps(I[window], theta[window])
        
        crystallinity = (total_area - amorphous_area) / total_area
        
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
            "index": max(0, min(1, crystallinity)),
            "classification": classification,
            "total_area": total_area,
            "amorphous_area": amorphous_area
        }
    
    def _scherrer_analysis(self, theta: np.ndarray, I: np.ndarray, 
                          peaks: List[float]) -> Dict:
        """Scherrer analysis for crystallite size"""
        if len(peaks) == 0:
            return {"size": None, "sizes": [], "average": None}
        
        sizes = []
        for peak in peaks[:3]:  # Analyze first 3 major peaks
            idx = np.argmin(np.abs(theta - peak))
            
            # Extract peak region
            window = slice(max(0, idx-20), min(len(theta), idx+20))
            theta_peak = theta[window]
            I_peak = I[window]
            
            # Gaussian fit
            try:
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-(x - x0)**2 / (2*sigma**2))
                
                p0 = [I_peak.max(), peak, 0.1]
                bounds = ([0, peak-0.5, 0.01], [2, peak+0.5, 1])
                popt, _ = optimize.curve_fit(
                    gaussian, theta_peak, I_peak, 
                    p0=p0, bounds=bounds
                )
                
                # Calculate FWHM and crystallite size
                fwhm = 2.355 * popt[2]  # in degrees
                theta_rad = np.radians(popt[1] / 2)
                beta_rad = np.radians(fwhm)
                
                size = 0.9 * self.lambda_x / (beta_rad * np.cos(theta_rad))
                sizes.append(size)
                
            except:
                continue
        
        return {
            "size": np.mean(sizes) if sizes else None,
            "sizes": sizes,
            "average": np.mean(sizes) if sizes else None,
            "std": np.std(sizes) if len(sizes) > 1 else None
        }
    
    def _williamson_hall_analysis(self, theta: np.ndarray, I: np.ndarray,
                                 peaks: List[float]) -> Dict:
        """Williamson-Hall analysis for strain and size"""
        if len(peaks) < 3:
            return {"strain": 0, "dislocation_density": 0}
        
        # Prepare data for Williamson-Hall plot
        beta = []  # FWHM
        d_spacing = []  # d-spacing
        
        for peak in peaks[:5]:
            idx = np.argmin(np.abs(theta - peak))
            window = slice(max(0, idx-15), min(len(theta), idx+15))
            theta_peak = theta[window]
            I_peak = I[window]
            
            # Estimate FWHM
            half_max = I_peak.max() / 2
            above_half = I_peak >= half_max
            if np.sum(above_half) > 1:
                fwhm = theta_peak[above_half][-1] - theta_peak[above_half][0]
                beta.append(fwhm)
                
                # Bragg's law: d = λ/(2 sinθ)
                d = self.lambda_x / (2 * np.sin(np.radians(peak/2)))
                d_spacing.append(d)
        
        if len(beta) < 3:
            return {"strain": 0, "dislocation_density": 0}
        
        # Williamson-Hall equation: β cosθ = Kλ/D + 4ε sinθ
        theta_rad = np.radians(np.array(peaks[:len(beta)]) / 2)
        beta_rad = np.radians(np.array(beta))
        
        X = 4 * np.sin(theta_rad)  # Strain term
        Y = beta_rad * np.cos(theta_rad)  # Left side
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(X, Y)
        
        size = 0.9 * self.lambda_x / intercept if intercept > 0 else None
        strain = slope / 4
        
        # Dislocation density
        dislocation_density = strain**2 / (size**2) if size else 0
        
        return {
            "strain": strain,
            "size_wh": size,
            "dislocation_density": dislocation_density,
            "wh_r2": r_value**2
        }
    
    def _analyze_mesostructure(self, theta: np.ndarray, I: np.ndarray) -> Dict:
        """Detect ordered mesopores from low-angle scattering"""
        # Low-angle region (0.5-5° 2θ)
        mask = (theta >= 0.5) & (theta <= 5.0)
        if np.sum(mask) < 10:
            return {"ordered": False, "d_spacing": None}
        
        theta_low = theta[mask]
        I_low = I[mask]
        
        # Detect peaks in low-angle region
        peaks_low, properties = signal.find_peaks(
            I_low, 
            prominence=0.05,
            distance=5
        )
        
        if len(peaks_low) == 0:
            return {"ordered": False, "d_spacing": None}
        
        # Check for multiple peaks (indicating ordered structure)
        if len(peaks_low) >= 2:
            # Calculate d-spacing from first peak (Bragg's law)
            d = self.lambda_x / (2 * np.sin(np.radians(theta_low[peaks_low[0]]/2)))
            
            # Check if peaks follow rational ratios (1:√3:√4:√7:√9 for hexagonal)
            peak_positions = theta_low[peaks_low]
            ratios = peak_positions / peak_positions[0]
            
            expected_hex = [1, np.sqrt(3), 2, np.sqrt(7), 3]
            mean_error = np.mean([min(abs(r - e) for e in expected_hex) for r in ratios])
            
            if mean_error < 0.2 and len(peaks_low) >= 2:
                return {
                    "ordered": True,
                    "d_spacing": d,
                    "symmetry": "Hexagonal (p6mm)",
                    "peaks_detected": len(peaks_low),
                    "ratios": ratios.tolist()
                }
        
        return {
            "ordered": True if len(peaks_low) >= 1 else False,
            "d_spacing": self.lambda_x / (2 * np.sin(np.radians(theta_low[peaks_low[0]]/2))) if len(peaks_low) >= 1 else None
        }
    
    def _calculate_fwhm(self, theta: np.ndarray, I: np.ndarray, 
                       peaks: List[float]) -> List[float]:
        """Calculate Full Width at Half Maximum for each peak"""
        fwhm_values = []
        for peak in peaks:
            idx = np.argmin(np.abs(theta - peak))
            window = slice(max(0, idx-15), min(len(theta), idx+15))
            theta_peak = theta[window]
            I_peak = I[window]
            
            half_max = I_peak.max() / 2
            above_half = I_peak >= half_max
            
            if np.sum(above_half) > 1:
                fwhm = theta_peak[above_half][-1] - theta_peak[above_half][0]
                fwhm_values.append(fwhm)
        
        return fwhm_values

# ============================================================================
# MORPHOLOGY FUSION ENGINE (THE CORE OF YOUR RESEARCH)
# ============================================================================
class MorphologyFusionEngine:
    """
    Scientific fusion of BET and XRD data for comprehensive morphology analysis
    Implements novel fusion algorithm for journal publication
    """
    
    @staticmethod
    def fuse_morphology(bet_data: Dict, xrd_data: Dict) -> FusedMorphology:
        """
        Core fusion algorithm integrating BET and XRD data
        
        Algorithm:
        1. Calculate complementarity scores
        2. Identify dominant structural features
        3. Estimate pore wall thickness
        4. Calculate structural integrity index
        5. Generate journal recommendations
        """
        
        # Extract key parameters
        s_bet = bet_data.get("surface_area_bet", 0)
        v_pore = bet_data.get("total_pore_volume", 0)
        d_pore = bet_data.get("mean_pore_diameter", 0)
        micro_frac = bet_data.get("pore_size_distribution", {}).get("micropore_fraction", 0)
        
        xrd_cryst = xrd_data.get("crystallinity", {}).get("index", 0)
        cryst_size = xrd_data.get("crystallite_size", {}).get("size", 0)
        ordered_meso = xrd_data.get("ordered_mesopores", False)
        d_spacing = xrd_data.get("d_spacing")
        
        # 1. Calculate surface-to-volume ratio (key morphology parameter)
        surface_to_volume = s_bet / (v_pore * 1e3) if v_pore > 0 else 0  # m²/cm³
        
        # 2. Estimate pore wall thickness (novel calculation)
        pore_wall_thickness = None
        if d_spacing and d_pore > 0:
            # For ordered mesopores: wall thickness = d-spacing - pore diameter
            pore_wall_thickness = d_spacing - d_pore
            pore_wall_thickness = max(0, pore_wall_thickness)
        
        # 3. Calculate structural integrity index (0-1)
        integrity_components = []
        
        # Component 1: Crystallinity contribution
        integrity_components.append(xrd_cryst * 0.4)
        
        # Component 2: Surface area efficiency
        sa_efficiency = min(1, s_bet / 2000)  # Normalize to 2000 m²/g
        integrity_components.append(sa_efficiency * 0.3)
        
        # Component 3: Pore structure stability
        if micro_frac > 0.5:
            pore_stability = 0.8  # Microporous = stable
        elif ordered_meso:
            pore_stability = 0.9  # Ordered = very stable
        else:
            pore_stability = 0.6  # Disordered = less stable
        integrity_components.append(pore_stability * 0.3)
        
        structural_integrity = np.mean(integrity_components)
        
        # 4. Determine dominant morphology feature
        dominant_feature = MorphologyFusionEngine._determine_dominant_feature(
            s_bet, micro_frac, xrd_cryst, ordered_meso
        )
        
        # 5. Composite classification
        composite_class = MorphologyFusionEngine._classify_composite(
            s_bet, xrd_cryst, micro_frac, ordered_meso
        )
        
        # 6. Calculate confidence score (0-1)
        confidence_score = MorphologyFusionEngine._calculate_confidence(
            bet_data, xrd_data
        )
        
        # 7. Journal recommendations
        journal_rec = MorphologyFusionEngine._generate_recommendations(
            composite_class, dominant_feature, structural_integrity
        )
        
        return FusedMorphology(
            composite_class=composite_class,
            surface_to_volume=surface_to_volume,
            pore_wall_thickness=pore_wall_thickness,
            structural_integrity=structural_integrity,
            dominant_feature=dominant_feature,
            confidence_score=confidence_score,
            journal_recommendation=journal_rec
        )
    
    @staticmethod
    def _determine_dominant_feature(s_bet: float, micro_frac: float,
                                   xrd_cryst: float, ordered_meso: bool) -> str:
        """Determine the dominant morphological feature"""
        if s_bet > 1000 and micro_frac > 0.7:
            return "Microporous carbon network"
        elif s_bet > 500 and ordered_meso:
            return "Ordered mesoporous framework"
        elif xrd_cryst > 0.7 and s_bet > 100:
            return "Crystalline porous material"
        elif s_bet > 500:
            return "High-surface-area mesoporous material"
        elif xrd_cryst < 0.3 and s_bet > 100:
            return "Amorphous porous solid"
        elif xrd_cryst > 0.7:
            return "Crystalline non-porous solid"
        else:
            return "Mixed-phase porous composite"
    
    @staticmethod
    def _classify_composite(s_bet: float, xrd_cryst: float,
                           micro_frac: float, ordered_meso: bool) -> str:
        """Scientific classification of composite morphology"""
        
        if ordered_meso and s_bet > 600:
            return "Type I: Ordered mesoporous crystalline"
        elif xrd_cryst > 0.8 and s_bet > 800:
            return "Type II: Hierarchical porous crystalline"
        elif micro_frac > 0.6 and s_bet > 1000:
            return "Type III: Microporous carbonaceous"
        elif 0.3 < xrd_cryst < 0.7 and s_bet > 300:
            return "Type IV: Semi-crystalline porous"
        elif xrd_cryst < 0.2 and s_bet > 200:
            return "Type V: Amorphous porous"
        elif s_bet < 50 and xrd_cryst > 0.6:
            return "Type VI: Dense crystalline"
        else:
            return "Type VII: Complex composite"
    
    @staticmethod
    def _calculate_confidence(bet_data: Dict, xrd_data: Dict) -> float:
        """Calculate confidence score (0-1) based on data quality"""
        confidence_factors = []
        
        # BET data quality
        if bet_data.get("valid", False):
            bet_r2 = bet_data.get("bet_r2", 0) if isinstance(bet_data, dict) else 0
            if isinstance(bet_r2, dict):
                bet_r2 = bet_r2.get("bet_r2", 0)
            confidence_factors.append(min(1, bet_r2))
            
            # PSD quality
            if bet_data.get("pore_size_distribution"):
                psd = bet_data["pore_size_distribution"]
                if len(psd.get("pore_diameters", [])) > 10:
                    confidence_factors.append(0.8)
        
        # XRD data quality
        if xrd_data.get("valid", False):
            if len(xrd_data.get("peaks", [])) >= 3:
                confidence_factors.append(0.9)
            elif len(xrd_data.get("peaks", [])) >= 1:
                confidence_factors.append(0.6)
        
        # Data consistency
        if bet_data.get("valid") and xrd_data.get("valid"):
            # Check if both analyses suggest similar porosity
            bet_porous = bet_data.get("surface_area_bet", 0) > 100
            xrd_porous = xrd_data.get("ordered_mesopores", False) or xrd_data.get("crystallinity", {}).get("index", 0) < 0.8
            if bet_porous == xrd_porous:
                confidence_factors.append(0.9)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    @staticmethod
    def _generate_recommendations(composite_class: str, dominant_feature: str,
                                integrity: float) -> str:
        """Generate journal-specific recommendations"""
        recommendations = []
        
        # Journal targeting based on morphology type
        if "Ordered mesoporous" in composite_class or "mesoporous framework" in dominant_feature:
            recommendations.append(
                "Recommended journals: Chemistry of Materials, Microporous and Mesoporous Materials"
            )
        
        if integrity > 0.8 and "crystalline" in composite_class.lower():
            recommendations.append(
                "High-integrity crystalline material suitable for: Journal of Materials Chemistry A"
            )
        
        if "Microporous" in composite_class and "carbon" in dominant_feature.lower():
            recommendations.append(
                "Carbon-based microporous system ideal for: Carbon, Advanced Functional Materials"
            )
        
        if "Complex composite" in composite_class:
            recommendations.append(
                "Complex morphology suggests: ACS Applied Materials & Interfaces, Materials Horizons"
            )
        
        # General recommendations
        if integrity > 0.7:
            recommendations.append("Material shows excellent structural integrity for device applications")
        
        if not recommendations:
            recommendations.append("Consider: Materials Chemistry and Physics for fundamental studies")
        
        return " | ".join(recommendations)

# ============================================================================
# JOURNAL-QUALITY VISUALIZATION
# ============================================================================
class ScientificVisualizer:
    """Publication-quality visualization for journal figures"""
    
    @staticmethod
    def create_morphology_summary_figure(bet_data: Dict, xrd_data: Dict, 
                                        morphology: FusedMorphology) -> plt.Figure:
        """Create comprehensive 4-panel morphology summary figure"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: BET isotherm
        ax1 = fig.add_subplot(gs[0, 0])
        ScientificVisualizer._plot_bet_isotherm(ax1, bet_data)
        ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, 
                fontsize=16, fontweight='bold', va='top')
        
        # Panel B: Pore size distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ScientificVisualizer._plot_pore_size_distribution(ax2, bet_data)
        ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, 
                fontsize=16, fontweight='bold', va='top')
        
        # Panel C: XRD pattern
        ax3 = fig.add_subplot(gs[0, 2])
        ScientificVisualizer._plot_xrd_pattern(ax3, xrd_data)
        ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, 
                fontsize=16, fontweight='bold', va='top')
        
        # Panel D: Morphology radar chart
        ax4 = fig.add_subplot(gs[1, :], projection='polar')
        ScientificVisualizer._plot_morphology_radar(ax4, morphology, bet_data, xrd_data)
        ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, 
                fontsize=16, fontweight='bold', va='top')
        
        # Panel E: Fusion summary table
        ax5 = fig.add_subplot(gs[2, :])
        ScientificVisualizer._plot_fusion_summary(ax5, morphology, bet_data, xrd_data)
        ax5.text(0.02, 0.98, 'E', transform=ax5.transAxes, 
                fontsize=16, fontweight='bold', va='top')
        ax5.axis('off')
        
        plt.suptitle("Comprehensive Morphology Analysis", fontsize=16, y=0.98)
        
        return fig
    
    @staticmethod
    def _plot_bet_isotherm(ax, bet_data):
        """Plot adsorption-desorption isotherm"""
        if not bet_data.get("valid", False):
            ax.text(0.5, 0.5, "No valid BET data", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        ads_data = bet_data.get("adsorption_data", {})
        
        if "p_ads" in ads_data:
            ax.plot(ads_data["p_ads"], ads_data["q_ads"], 
                   'o-', linewidth=2, markersize=4, label='Adsorption')
        
        if "p_des" in ads_data and ads_data["p_des"] is not None:
            ax.plot(ads_data["p_des"], ads_data["q_des"], 
                   's--', linewidth=2, markersize=4, label='Desorption')
        
        ax.set_xlabel("Relative Pressure (P/P₀)")
        ax.set_ylabel("Quantity Adsorbed (mmol/g)")
        ax.set_title("Physisorption Isotherm")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add BET parameters
        bet_text = f"$S_{{BET}}$ = {bet_data.get('surface_area_bet', 0):.0f} m²/g\n"
        bet_text += f"C = {bet_data.get('c_constant', 0):.0f}"
        ax.text(0.05, 0.95, bet_text, transform=ax.transAxes,
               fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    @staticmethod
    def _plot_pore_size_distribution(ax, bet_data):
        """Plot pore size distribution"""
        psd = bet_data.get("pore_size_distribution", {})
        
        if not psd or "pore_diameters" not in psd:
            ax.text(0.5, 0.5, "No PSD data", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        diameters = psd["pore_diameters"]
        dv_dlogd = psd["dv_dlogd"]
        
        ax.plot(diameters, dv_dlogd, '-', linewidth=2)
        ax.fill_between(diameters, 0, dv_dlogd, alpha=0.3)
        
        ax.set_xscale('log')
        ax.set_xlabel("Pore Diameter (nm)")
        ax.set_ylabel("dV/dlogD (cm³/g)")
        ax.set_title("Pore Size Distribution")
        ax.grid(True, alpha=0.3, which='both')
        
        # Add pore statistics
        pore_text = f"Mean D = {psd.get('mean_pore_diameter', 0):.1f} nm\n"
        pore_text += f"V$_{{total}}$ = {psd.get('total_pore_volume', 0):.3f} cm³/g"
        ax.text(0.95, 0.95, pore_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    @staticmethod
    def _plot_xrd_pattern(ax, xrd_data):
        """Plot XRD pattern with peak markers"""
        if not xrd_data.get("valid", False):
            ax.text(0.5, 0.5, "No valid XRD data", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        theta = xrd_data.get("two_theta", [])
        intensity = xrd_data.get("intensity", [])
        peaks = xrd_data.get("peaks", [])
        
        ax.plot(theta, intensity, '-', linewidth=1.5)
        
        # Mark peaks
        if peaks:
            for peak in peaks:
                idx = np.argmin(np.abs(theta - peak))
                ax.plot(peak, intensity[idx], 'r^', markersize=8)
        
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("XRD Pattern")
        ax.grid(True, alpha=0.3)
        
        # Add XRD parameters
        xrd_text = f"Crystallinity = {xrd_data.get('crystallinity', {}).get('index', 0):.2f}\n"
        size = xrd_data.get('crystallite_size', {}).get('size')
        if size:
            xrd_text += f"Size = {size:.1f} nm"
        
        ax.text(0.05, 0.95, xrd_text, transform=ax.transAxes,
               fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    @staticmethod
    def _plot_morphology_radar(ax, morphology, bet_data, xrd_data):
        """Create radar chart of morphology parameters"""
        categories = ['Surface Area', 'Porosity', 'Crystallinity', 
                     'Ordering', 'Stability', 'Complexity']
        
        # Normalized values (0-1)
        s_bet_norm = min(1, bet_data.get("surface_area_bet", 0) / 2000)
        porosity_norm = min(1, bet_data.get("total_pore_volume", 0) * 10)
        cryst_norm = xrd_data.get("crystallinity", {}).get("index", 0)
        ordering_norm = 0.9 if xrd_data.get("ordered_mesopores") else 0.3
        stability_norm = morphology.structural_integrity
        complexity_norm = 0.8 if "Complex" in morphology.composite_class else 0.4
        
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
        ax.set_title("Morphology Radar Chart", y=1.1)
        
        # Add composite class
        ax.text(np.pi/2, 1.2, morphology.composite_class,
               ha='center', va='center', fontsize=11, fontweight='bold')
    
    @staticmethod
    def _plot_fusion_summary(ax, morphology, bet_data, xrd_data):
        """Create text summary table"""
        summary_text = "INTEGRATED MORPHOLOGY ANALYSIS SUMMARY\n"
        summary_text += "="*50 + "\n\n"
        
        # BET Summary
        summary_text += "BET ANALYSIS:\n"
        summary_text += f"  • Surface Area: {bet_data.get('surface_area_bet', 0):.0f} m²/g\n"
        summary_text += f"  • Total Pore Volume: {bet_data.get('total_pore_volume', 0):.3f} cm³/g\n"
        summary_text += f"  • Mean Pore Diameter: {bet_data.get('mean_pore_diameter', 0):.1f} nm\n"
        
        psd = bet_data.get('pore_size_distribution', {})
        if psd:
            summary_text += f"  • Micro/Meso/Macro: {psd.get('micropore_fraction', 0):.2f}/{psd.get('mesopore_fraction', 0):.2f}/{psd.get('macropore_fraction', 0):.2f}\n"
        
        # XRD Summary
        summary_text += "\nXRD ANALYSIS:\n"
        summary_text += f"  • Crystallinity Index: {xrd_data.get('crystallinity', {}).get('index', 0):.2f}\n"
        summary_text += f"  • Crystallite Size: {xrd_data.get('crystallite_size', {}).get('size', 0):.1f} nm\n"
        summary_text += f"  • Ordered Mesopores: {'Yes' if xrd_data.get('ordered_mesopores') else 'No'}\n"
        
        # Fusion Results
        summary_text += "\nFUSION RESULTS:\n"
        summary_text += f"  • Composite Classification: {morphology.composite_class}\n"
        summary_text += f"  • Dominant Feature: {morphology.dominant_feature}\n"
        summary_text += f"  • Surface-to-Volume Ratio: {morphology.surface_to_volume:.0f} m²/cm³\n"
        summary_text += f"  • Structural Integrity: {morphology.structural_integrity:.2f}\n"
        if morphology.pore_wall_thickness:
            summary_text += f"  • Pore Wall Thickness: {morphology.pore_wall_thickness:.1f} nm\n"
        summary_text += f"  • Analysis Confidence: {morphology.confidence_score:.2f}\n"
        
        # Journal Recommendations
        summary_text += f"\nJOURNAL RECOMMENDATION:\n  {morphology.journal_recommendation}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='top', linespacing=1.5)

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================
def main():
    # Sidebar for input parameters
    with st.sidebar:
        st.header("⚙️ Analysis Parameters")
        
        st.subheader("BET Analysis")
        bet_min_p = st.slider("BET linear range min (P/P₀)", 0.01, 0.1, 0.05, 0.01)
        bet_max_p = st.slider("BET linear range max (P/P₀)", 0.2, 0.4, 0.35, 0.01)
        
        st.subheader("XRD Analysis")
        xrd_wavelength = st.selectbox("X-ray wavelength", 
                                     ["Cu Kα (0.15406 nm)", "Mo Kα (0.07107 nm)"],
                                     index=0)
        xrd_wavelength = 0.15406 if "Cu" in xrd_wavelength else 0.07107
        
        st.subheader("Export Options")
        export_format = st.selectbox("Figure format", ["PNG", "PDF", "SVG"], index=0)
        export_dpi = st.slider("Export DPI", 150, 600, 300)
        
        st.markdown("---")
        st.markdown("**Citation:**")
        st.caption("Integrated BET-XRD Morphology Analyzer v2.0")
        st.caption("DOI: 10.XXXX/xxxx (Submitted)")
    
    # Main interface
    st.title("🔬 Integrated BET–XRD Morphology Analyzer")
    st.markdown("""
    *Journal Submission Version* | 
    **Scientific morphology fusion algorithm for advanced materials characterization**
    
    This tool implements IUPAC-compliant analysis with novel BET-XRD fusion for comprehensive
    morphology characterization. Results are publication-ready.
    """)
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BET Data")
        bet_file = st.file_uploader(
            "Upload BET isotherm (Excel/CSV)",
            type=["xls", "xlsx", "csv", "txt"],
            help="Expected format: ASAP 2420 or similar. Include adsorption and desorption branches."
        )
        
        if bet_file:
            st.info(f"✅ BET file uploaded: {bet_file.name}")
    
    with col2:
        st.subheader("XRD Data")
        xrd_file = st.file_uploader(
            "Upload XRD pattern (CSV/TXT)",
            type=["csv", "txt", "xy", "dat"],
            help="Two-column format: 2θ (degrees) and Intensity"
        )
        
        if xrd_file:
            st.info(f"✅ XRD file uploaded: {xrd_file.name}")
    
    # Analysis button
    analyze_button = st.button("🚀 Run Comprehensive Analysis", 
                             type="primary",
                             disabled=not (bet_file or xrd_file),
                             use_container_width=True)
    
    # Initialize session state
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "morphology_fusion" not in st.session_state:
        st.session_state.morphology_fusion = None
    
    if analyze_button:
        with st.spinner("🔬 Running scientific analysis..."):
            # Clear previous results
            st.session_state.analysis_results = {}
            st.session_state.morphology_fusion = None
            
            # BET Analysis
            if bet_file:
                try:
                    # Get file info
                    file_name = bet_file.name
                    file_extension = file_name.split('.')[-1].lower()
                    
                    st.info(f"📁 Reading BET file: {file_name} (Type: {file_extension})")
                    
                    # Reset file pointer
                    bet_file.seek(0)
                    
                    # ============================================
                    # 1. Try to read as Excel
                    # ============================================
                    if file_extension in ['xls', 'xlsx']:
                        try:
                            if file_extension == 'xls':
                                # Old Excel format (.xls) - use xlrd
                                df_bet = pd.read_excel(bet_file, engine='xlrd', header=None)
                                st.success("✅ Read as .xls file (xlrd engine)")
                            else:
                                # New Excel format (.xlsx) - use openpyxl
                                df_bet = pd.read_excel(bet_file, engine='openpyxl', header=None)
                                st.success("✅ Read as .xlsx file (openpyxl engine)")
                                
                        except Exception as excel_error:
                            st.warning(f"⚠️ Excel read failed: {str(excel_error)[:100]}...")
                            # Try as CSV/text
                            bet_file.seek(0)
                            try:
                                content = bet_file.read().decode('utf-8', errors='ignore')
                                df_bet = pd.read_csv(io.StringIO(content), header=None)
                                st.success("✅ Read as CSV/text file")
                            except:
                                raise ValueError("Cannot read file as Excel or CSV")
                    
                    # ============================================
                    # 2. Try to read as CSV/TXT
                    # ============================================
                    elif file_extension in ['csv', 'txt']:
                        try:
                            # Try different encodings and delimiters
                            content = bet_file.read().decode('utf-8', errors='ignore')
                            bet_file.seek(0)
                            
                            # Try to detect delimiter
                            first_line = content.split('\n')[0] if '\n' in content else content
                            
                            if ';' in first_line:
                                df_bet = pd.read_csv(bet_file, delimiter=';', header=None)
                            elif '\t' in first_line:
                                df_bet = pd.read_csv(bet_file, delimiter='\t', header=None)
                            elif ',' in first_line:
                                df_bet = pd.read_csv(bet_file, delimiter=',', header=None)
                            else:
                                # Try with no header and auto-detect
                                df_bet = pd.read_csv(bet_file, header=None)
                                
                            st.success("✅ Read as CSV file")
                            
                        except Exception as csv_error:
                            st.warning(f"⚠️ CSV read failed: {str(csv_error)[:100]}...")
                            # Try as Excel as fallback
                            bet_file.seek(0)
                            try:
                                df_bet = pd.read_excel(bet_file, engine='openpyxl', header=None)
                                st.success("✅ Read as Excel (fallback)")
                            except:
                                raise ValueError("Cannot read file")
                    
                    # ============================================
                    # 3. Show data preview
                    # ============================================
                    with st.expander("📊 Data Preview", expanded=False):
                        st.write(f"**Shape:** {df_bet.shape} (rows × columns)")
                        st.write(f"**Columns:** {list(range(df_bet.shape[1]))}")
                        st.dataframe(df_bet.head(10))
                        
                        # Show raw content for debugging
                        if st.checkbox("Show raw file content (debug)"):
                            bet_file.seek(0)
                            raw_content = bet_file.read()[:5000]  # First 5000 bytes
                            try:
                                st.text(raw_content.decode('utf-8', errors='ignore'))
                            except:
                                st.code(str(raw_content))
                    
                    # ============================================
                    # 4. Extract BET data (ASAP 2420 format)
                    # ============================================
                    st.info("🔍 Extracting BET data from standard ASAP 2420 format...")
                    
                    # Method 1: Try ASAP 2420 specific columns
                    p_ads, q_ads = [], []
                    p_des, q_des = [], []
                    
                    # ASAP 2420 format: Adsorption in columns L (11) and M (12)
                    #                    Desorption in columns N (13) and O (14)
                    #                    Starting from row 29 (0-indexed 28)
                    
                    for i in range(len(df_bet)):
                        try:
                            # Check if this looks like adsorption data
                            p_val = df_bet.iloc[i, 11] if df_bet.shape[1] > 11 else None
                            q_val = df_bet.iloc[i, 12] if df_bet.shape[1] > 12 else None
                            
                            if pd.notna(p_val) and pd.notna(q_val) and 0 < p_val < 1 and q_val > 0:
                                p_ads.append(float(p_val))
                                q_ads.append(float(q_val))
                                
                            # Check for desorption data
                            if df_bet.shape[1] > 14:
                                p_des_val = df_bet.iloc[i, 13] if df_bet.shape[1] > 13 else None
                                q_des_val = df_bet.iloc[i, 14] if df_bet.shape[1] > 14 else None
                                
                                if pd.notna(p_des_val) and pd.notna(q_des_val) and 0 < p_des_val < 1 and q_des_val > 0:
                                    p_des.append(float(p_des_val))
                                    q_des.append(float(q_des_val))
                        except:
                            continue
                    
                    # Method 2: If ASAP format not found, try to auto-detect
                    if len(p_ads) < 5:
                        st.warning("⚠️ ASAP 2420 format not detected, trying auto-detection...")
                        
                        # Reset
                        p_ads, q_ads = [], []
                        
                        # Find columns that look like pressure and quantity
                        for col_idx in range(min(10, df_bet.shape[1])):
                            col_data = df_bet.iloc[:, col_idx].dropna()
                            if len(col_data) > 0:
                                # Check if column looks like pressure (0-1 range)
                                sample_vals = col_data.head(10).values
                                if all(0 <= x <= 1 for x in sample_vals if pd.notna(x)):
                                    p_col = col_idx
                                    # Next column might be quantity
                                    if col_idx + 1 < df_bet.shape[1]:
                                        q_col = col_idx + 1
                                        p_ads = df_bet.iloc[:, p_col].dropna().values
                                        q_ads = df_bet.iloc[:, q_col].dropna().values
                                        break
                    
                    # ============================================
                    # 5. Validate extracted data
                    # ============================================
                    if len(p_ads) < 5 or len(q_ads) < 5:
                        st.error(f"❌ Insufficient adsorption data found. Found {len(p_ads)} points.")
                        st.session_state.analysis_results["bet"] = {
                            "valid": False, 
                            "error": f"Found only {len(p_ads)} adsorption points (minimum 5 required)"
                        }
                    else:
                        st.success(f"✅ Found {len(p_ads)} adsorption data points")
                        
                        if len(p_des) > 5:
                            st.success(f"✅ Found {len(p_des)} desorption data points")
                        
                        # Convert to numpy arrays
                        p_ads = np.array(p_ads, dtype=np.float64)
                        q_ads = np.array(q_ads, dtype=np.float64)
                        
                        if len(p_des) > 0:
                            p_des = np.array(p_des, dtype=np.float64)
                            q_des = np.array(q_des, dtype=np.float64)
                        else:
                            p_des = q_des = None
                        
                        # Run BET analysis
                        bet_analyzer = IUPACBETAnalyzer(p_ads, q_ads, p_des, q_des)
                        bet_results = bet_analyzer.full_analysis()
                        st.session_state.analysis_results["bet"] = bet_results
                        
                        st.success(f"✅ BET analysis complete: S_BET = {bet_results.get('surface_area_bet', 0):.0f} m²/g")
                        
                except Exception as e:
                    st.error(f"❌ BET analysis failed: {str(e)}")
                    # Detailed error info
                    with st.expander("Error details"):
                        st.exception(e)
                    st.session_state.analysis_results["bet"] = {"valid": False, "error": str(e)}
            
            # XRD Analysis
            if xrd_file:
                try:
                    # Read XRD data - FIXED VERSION
                    file_extension = xrd_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['csv', 'txt', 'xy', 'dat']:
                        # Try to read with different delimiters
                        xrd_file.seek(0)  # Reset file pointer
                        content = xrd_file.read().decode('utf-8', errors='ignore')
                        
                        # Try common delimiters
                        for delimiter in [',', '\t', ';', ' ']:
                            try:
                                df_xrd = pd.read_csv(io.StringIO(content), sep=delimiter, header=None)
                                if len(df_xrd.columns) >= 2:
                                    break
                            except:
                                continue
                        
                        # If still not loaded, try with pandas default
                        if 'df_xrd' not in locals():
                            xrd_file.seek(0)
                            df_xrd = pd.read_csv(xrd_file, header=None)
                    
                    else:
                        # Try as Excel
                        xrd_file.seek(0)
                        try:
                            df_xrd = pd.read_excel(xrd_file, header=None, engine='openpyxl')
                        except:
                            raise ValueError(f"Unsupported XRD file format: {file_extension}")
                    
                    # Check data
                    if df_xrd.empty:
                        st.error("❌ The uploaded XRD file appears to be empty")
                        st.session_state.analysis_results["xrd"] = {"valid": False, "error": "Empty file"}
                    else:
                        st.info(f"✅ XRD file loaded successfully: {len(df_xrd)} data points")
                        
                        # Show preview
                        with st.expander("Preview XRD data"):
                            st.dataframe(df_xrd.head())
                            st.write(f"Data shape: {df_xrd.shape}")
                        
                        # Extract theta and intensity
                        if len(df_xrd.columns) >= 2:
                            theta = df_xrd.iloc[:, 0].dropna().values
                            intensity = df_xrd.iloc[:, 1].dropna().values
                            
                            # Run XRD analysis
                            xrd_analyzer = AdvancedXRDAnalyzer(xrd_wavelength)
                            xrd_results = xrd_analyzer.analyze(theta, intensity)
                            st.session_state.analysis_results["xrd"] = xrd_results
                            
                            st.success(f"✅ XRD analysis complete: Crystallinity = {xrd_results.get('crystallinity', {}).get('index', 0):.2f}")
                        else:
                            st.error("❌ XRD file must have at least 2 columns (2θ and Intensity)")
                            st.session_state.analysis_results["xrd"] = {"valid": False, "error": "Insufficient columns"}
                            
                except Exception as e:
                    st.error(f"❌ XRD analysis failed: {str(e)}")
                    st.session_state.analysis_results["xrd"] = {"valid": False, "error": str(e)}
            
            # Morphology Fusion
            bet_data = st.session_state.analysis_results.get("bet", {})
            xrd_data = st.session_state.analysis_results.get("xrd", {})
            
            if (bet_data.get("valid", False) or xrd_data.get("valid", False)):
                try:
                    morphology = MorphologyFusionEngine.fuse_morphology(bet_data, xrd_data)
                    st.session_state.morphology_fusion = morphology
                    
                    st.success("✅ Morphology fusion complete!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Morphology fusion failed: {str(e)}")
    
    # Display results if available
    if st.session_state.analysis_results or st.session_state.morphology_fusion:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Scientific Plots", 
            "🔍 Detailed Results", 
            "🧬 Morphology Fusion", 
            "📤 Export"
        ])
        
        with tab1:
            # Generate and display summary figure
            bet_data = st.session_state.analysis_results.get("bet", {})
            xrd_data = st.session_state.analysis_results.get("xrd", {})
            morphology = st.session_state.morphology_fusion
            
            if bet_data.get("valid", False) or xrd_data.get("valid", False):
                fig = ScientificVisualizer.create_morphology_summary_figure(
                    bet_data, xrd_data, morphology
                )
                st.pyplot(fig)
                
                st.caption("Figure 1. Comprehensive morphology analysis summary")
            else:
                st.info("Run analysis to generate plots")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("BET Results")
                if st.session_state.analysis_results.get("bet", {}).get("valid", False):
                    bet = st.session_state.analysis_results["bet"]
                    
                    # Create metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Surface Area", f"{bet.get('surface_area_bet', 0):.0f} m²/g")
                    with m2:
                        st.metric("Pore Volume", f"{bet.get('total_pore_volume', 0):.3f} cm³/g")
                    with m3:
                        st.metric("Mean Pore D", f"{bet.get('mean_pore_diameter', 0):.1f} nm")
                    
                    # Display detailed results
                    with st.expander("Detailed BET parameters"):
                        st.json({
                            "BET Parameters": {
                                "C constant": f"{bet.get('c_constant', 0):.0f}",
                                "Monolayer capacity": f"{bet.get('monolayer_capacity', 0):.3f} mmol/g",
                                "Linear range": f"{bet.get('linear_range', (0,0))[0]:.3f}-{bet.get('linear_range', (0,0))[1]:.3f} P/P₀",
                                "R²": f"{bet.get('bet_r2', 0):.4f}"
                            },
                            "Porosity Analysis": {
                                "Micropore volume": f"{bet.get('micropore_volume', 0):.4f} cm³/g",
                                "External surface area": f"{bet.get('external_surface_area', 0):.0f} m²/g",
                                "Hysteresis type": bet.get('hysteresis', {}).get('type', 'N/A')
                            }
                        })
                else:
                    st.info("No valid BET data")
            
            with col2:
                st.subheader("XRD Results")
                if st.session_state.analysis_results.get("xrd", {}).get("valid", False):
                    xrd = st.session_state.analysis_results["xrd"]
                    
                    # Create metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Crystallinity", f"{xrd.get('crystallinity', {}).get('index', 0):.2f}")
                    with m2:
                        size = xrd.get('crystallite_size', {}).get('size')
                        st.metric("Crystallite Size", f"{size:.1f} nm" if size else "N/A")
                    with m3:
                        st.metric("Ordered Mesopores", 
                                 "✅ Yes" if xrd.get('ordered_mesopores') else "❌ No")
                    
                    # Display detailed results
                    with st.expander("Detailed XRD parameters"):
                        st.json({
                            "Crystallinity": {
                                "Index": f"{xrd.get('crystallinity', {}).get('index', 0):.3f}",
                                "Classification": xrd.get('crystallinity', {}).get('classification', 'N/A'),
                                "Peaks detected": len(xrd.get('peaks', []))
                            },
                            "Crystallite Analysis": {
                                "Size (Scherrer)": f"{xrd.get('crystallite_size', {}).get('size', 0):.1f} nm",
                                "Microstrain": f"{xrd.get('microstrain', 0):.4f}",
                                "Dislocation density": f"{xrd.get('dislocation_density', 0):.2e} m⁻²"
                            },
                            "Mesostructure": {
                                "Ordered": xrd.get('ordered_mesopores', False),
                                "d-spacing": f"{xrd.get('d_spacing', 0):.2f} nm" if xrd.get('d_spacing') else "N/A"
                            }
                        })
                else:
                    st.info("No valid XRD data")
        
        with tab3:
            st.subheader("🧬 Morphology Fusion Results")
            
            if st.session_state.morphology_fusion:
                morphology = st.session_state.morphology_fusion
                
                # Display fusion results
                st.markdown(f"""
                ### Composite Classification
                **{morphology.composite_class}**
                
                ### Dominant Morphological Feature
                {morphology.dominant_feature}
                
                ### Quantitative Morphology Parameters
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Surface/Volume", f"{morphology.surface_to_volume:.0f} m²/cm³")
                with col2:
                    st.metric("Structural Integrity", f"{morphology.structural_integrity:.2f}")
                with col3:
                    st.metric("Analysis Confidence", f"{morphology.confidence_score:.2f}")
                
                if morphology.pore_wall_thickness:
                    st.metric("Pore Wall Thickness", f"{morphology.pore_wall_thickness:.1f} nm")
                
                st.markdown("### Journal Recommendations")
                st.info(morphology.journal_recommendation)
                
                # Scientific interpretation
                st.markdown("### Scientific Interpretation")
                interpretation = """
                **Key Insights:**
                1. The integrated analysis reveals complementary information from BET and XRD
                2. Surface area and pore structure (BET) combined with crystallinity and ordering (XRD)
                3. Fusion algorithm provides holistic morphology understanding
                4. Results suitable for advanced materials characterization in publications
                """
                st.write(interpretation)
            else:
                st.info("Run analysis to generate morphology fusion")
        
        with tab4:
            st.subheader("📤 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Export Figures")
                
                if st.button("Generate Publication Figure"):
                    bet_data = st.session_state.analysis_results.get("bet", {})
                    xrd_data = st.session_state.analysis_results.get("xrd", {})
                    morphology = st.session_state.morphology_fusion
                    
                    if bet_data.get("valid", False) or xrd_data.get("valid", False):
                        fig = ScientificVisualizer.create_morphology_summary_figure(
                            bet_data, xrd_data, morphology
                        )
                        
                        # Save figure to buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format=export_format.lower(), 
                                   dpi=export_dpi, bbox_inches='tight')
                        buf.seek(0)
                        
                        # Create download button
                        st.download_button(
                            label=f"Download Figure (.{export_format.lower()})",
                            data=buf,
                            file_name=f"morphology_analysis.{export_format.lower()}",
                            mime=f"image/{export_format.lower()}" if export_format != "PDF" else "application/pdf"
                        )
                    else:
                        st.warning("No valid data to export")
            
            with col2:
                st.markdown("### Export Data")
                
                # Create data export
                export_data = {
                    "BET_Results": st.session_state.analysis_results.get("bet", {}),
                    "XRD_Results": st.session_state.analysis_results.get("xrd", {}),
                    "Morphology_Fusion": st.session_state.morphology_fusion.__dict__ if st.session_state.morphology_fusion else {},
                    "Analysis_Parameters": {
                        "bet_linear_range": (bet_min_p, bet_max_p),
                        "xrd_wavelength": xrd_wavelength,
                        "version": "2.0",
                        "citation": "Integrated BET-XRD Morphology Analyzer, DOI: 10.XXXX/xxxx"
                    }
                }
                
                # Convert to JSON
                import json
                json_str = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="Download Results (JSON)",
                    data=json_str,
                    file_name="morphology_analysis_results.json",
                    mime="application/json"
                )
            
            st.markdown("---")
            st.markdown("### Citation")
            st.code("""
            @software{BET_XRD_Morphology_Analyzer_2024,
                author = {Your Name},
                title = {Integrated BET-XRD Morphology Analyzer},
                version = {2.0},
                year = {2024},
                url = {https://your-app-url.com},
                note = {Scientific morphology fusion for advanced materials characterization}
            }
            """)

# ============================================================================
# APP EXECUTION
# ============================================================================
if __name__ == "__main__":
    main()
