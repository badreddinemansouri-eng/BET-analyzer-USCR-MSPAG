"""
INTEGRATED BET-XRD MORPHOLOGY ANALYZER
Complete working version for Render deployment
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import io
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BET-XRD Morphology Analyzer",
    layout="wide",
    page_icon="🔬"
)

st.title("🔬 BET-XRD Morphology Analyzer")
st.markdown("Scientific analysis of porous materials for journal publication")

# ============================================================================
# SCIENTIFIC FUNCTIONS (FIXED)
# ============================================================================
def integrate_trapezoidal(x, y):
    """Manual trapezoidal integration to replace np.trapz"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    area = 0.0
    for i in range(len(x) - 1):
        area += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.0
    return area

def analyze_bet_data(p_ads, q_ads, p_des=None, q_des=None):
    """Simplified BET analysis with manual integration"""
    try:
        # Basic validation
        if len(p_ads) < 5 or len(q_ads) < 5:
            return {"valid": False, "error": "Insufficient data points"}
        
        # Ensure numpy arrays
        p_ads = np.array(p_ads, dtype=float)
        q_ads = np.array(q_ads, dtype=float)
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(p_ads) & np.isfinite(q_ads) & (p_ads > 0) & (p_ads < 1) & (q_ads > 0)
        p_ads = p_ads[valid_mask]
        q_ads = q_ads[valid_mask]
        
        if len(p_ads) < 5:
            return {"valid": False, "error": "Not enough valid data points after cleaning"}
        
        # BET linear region (0.05-0.35)
        mask = (p_ads >= 0.05) & (p_ads <= 0.35)
        if np.sum(mask) < 3:
            return {"valid": False, "error": "No valid BET linear region (0.05-0.35 P/P₀)"}
        
        p_lin = p_ads[mask]
        q_lin = q_ads[mask]
        
        # BET transform: p/(q*(1-p))
        with np.errstate(divide='ignore', invalid='ignore'):
            y = p_lin / (q_lin * (1 - p_lin))
        
        # Remove any infinite or NaN values from BET transform
        valid_y = np.isfinite(y)
        if np.sum(valid_y) < 3:
            return {"valid": False, "error": "Invalid BET transform values"}
        
        p_lin = p_lin[valid_y]
        y = y[valid_y]
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(p_lin, y)
        
        # Calculate BET parameters
        if (slope + intercept) == 0 or intercept == 0:
            return {"valid": False, "error": "Invalid BET regression parameters"}
        
        qm = 1.0 / (slope + intercept)
        C = slope / intercept + 1.0
        surface_area = qm * 4.35  # m²/g for N2 (0.162 nm² cross-section)
        
        # Total pore volume (approximate from adsorption)
        pore_volume = q_ads[-1] if len(q_ads) > 0 else 0.0
        
        # Process desorption if available
        p_des_array = q_des_array = None
        if p_des is not None and q_des is not None:
            p_des_array = np.array(p_des, dtype=float)
            q_des_array = np.array(q_des, dtype=float)
            
            # Clean desorption data
            valid_des = np.isfinite(p_des_array) & np.isfinite(q_des_array) & (p_des_array > 0) & (p_des_array < 1) & (q_des_array > 0)
            p_des_array = p_des_array[valid_des]
            q_des_array = q_des_array[valid_des]
        
        return {
            "valid": True,
            "surface_area": float(surface_area),
            "pore_volume": float(pore_volume),
            "c_constant": float(C),
            "r_squared": float(r_value**2),
            "adsorption_data": {
                "p_rel": p_ads.tolist(),
                "q_ads": q_ads.tolist(),
                "p_des": p_des_array.tolist() if p_des_array is not None else None,
                "q_des": q_des_array.tolist() if q_des_array is not None else None
            }
        }
    except Exception as e:
        return {"valid": False, "error": f"BET analysis error: {str(e)}"}

def analyze_xrd_data(two_theta, intensity):
    """Simplified XRD analysis with manual integration"""
    try:
        # Convert to numpy arrays
        two_theta = np.array(two_theta, dtype=float)
        intensity = np.array(intensity, dtype=float)
        
        # Remove NaN values
        valid = np.isfinite(two_theta) & np.isfinite(intensity)
        two_theta = two_theta[valid]
        intensity = intensity[valid]
        
        if len(two_theta) < 10:
            return {"valid": False, "error": "Insufficient XRD data after cleaning"}
        
        # Normalize intensity
        if np.max(intensity) > np.min(intensity):
            intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        else:
            intensity_norm = intensity
        
        # Find peaks
        try:
            peaks, properties = signal.find_peaks(intensity_norm, height=0.1, distance=10)
        except:
            peaks = np.array([])
        
        # Crystallinity index using manual integration
        if len(peaks) > 0:
            # Calculate total area
            total_area = integrate_trapezoidal(two_theta, intensity_norm)
            
            # Calculate peak areas
            peak_area = 0.0
            for peak_idx in peaks:
                idx = int(peak_idx)
                left = max(0, idx - 5)
                right = min(len(two_theta) - 1, idx + 5)
                
                if right > left:
                    peak_area += integrate_trapezoidal(
                        two_theta[left:right+1], 
                        intensity_norm[left:right+1]
                    )
            
            crystallinity = peak_area / total_area if total_area > 0 else 0.0
        else:
            crystallinity = 0.0
        
        # Check for low-angle peaks (ordered mesopores)
        low_angle_mask = (two_theta >= 0.5) & (two_theta <= 5.0)
        ordered_mesopores = False
        if np.sum(low_angle_mask) >= 5:
            low_angle_intensity = intensity_norm[low_angle_mask]
            try:
                low_angle_peaks, _ = signal.find_peaks(low_angle_intensity, height=0.2)
                ordered_mesopores = len(low_angle_peaks) > 0
            except:
                ordered_mesopores = False
        
        return {
            "valid": True,
            "two_theta": two_theta.tolist(),
            "intensity": intensity_norm.tolist(),
            "peaks": two_theta[peaks].tolist() if len(peaks) > 0 else [],
            "crystallinity": float(crystallinity),
            "ordered_mesopores": bool(ordered_mesopores)
        }
    except Exception as e:
        return {"valid": False, "error": f"XRD analysis error: {str(e)}"}

# ============================================================================
# FILE READING FUNCTIONS (FIXED FOR ASAP 2420)
# ============================================================================
def read_bet_file(file):
    """Read BET file with ASAP 2420 format support"""
    try:
        filename = file.name.lower()
        
        # Show loading message
        with st.spinner(f"Reading {filename}..."):
            # Read file based on extension
            if filename.endswith('.csv'):
                df = pd.read_csv(file, header=None)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl', header=None)
            elif filename.endswith('.xls'):
                df = pd.read_excel(file, engine='xlrd', header=None)
            else:
                # Try as text file
                content = file.read().decode('utf-8', errors='ignore')
                file.seek(0)
                for delimiter in [',', '\t', ';', ' ']:
                    try:
                        df = pd.read_csv(io.StringIO(content), delimiter=delimiter, header=None)
                        break
                    except:
                        continue
                else:
                    return None, None, None, None, "Cannot read file format"
        
        st.info(f"📊 File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # =============================================
        # METHOD 1: ASAP 2420 specific extraction
        # =============================================
        p_ads, q_ads = [], []
        p_des, q_des = [], []
        
        # ASAP 2420 format:
        # Adsorption: Column L (index 11) = P/P0, Column M (index 12) = Q_ads
        # Desorption: Column N (index 13) = P/P0, Column O (index 14) = Q_des
        # Data starts around row 29 (index 28)
        
        rows_checked = min(150, len(df))
        for i in range(28, rows_checked):  # Start from row 29
            try:
                # Check if we have enough columns
                if df.shape[1] > 12:
                    # Adsorption data
                    p_val = df.iloc[i, 11]
                    q_val = df.iloc[i, 12]
                    
                    if pd.notna(p_val) and pd.notna(q_val):
                        p_float = float(p_val)
                        q_float = float(q_val)
                        if 0.001 < p_float < 0.999 and q_float > 0:
                            p_ads.append(p_float)
                            q_ads.append(q_float)
                
                # Desorption data
                if df.shape[1] > 14:
                    p_val_des = df.iloc[i, 13]
                    q_val_des = df.iloc[i, 14]
                    
                    if pd.notna(p_val_des) and pd.notna(q_val_des):
                        p_des_float = float(p_val_des)
                        q_des_float = float(q_val_des)
                        if 0.001 < p_des_float < 0.999 and q_des_float > 0:
                            p_des.append(p_des_float)
                            q_des.append(q_des_float)
            except (ValueError, TypeError):
                continue
        
        # =============================================
        # METHOD 2: Auto-detection if ASAP fails
        # =============================================
        if len(p_ads) < 5:
            st.warning("ASAP 2420 format not detected. Trying auto-detection...")
            
            # Reset
            p_ads, q_ads = [], []
            p_des, q_des = [], []
            
            # Convert to numeric
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            
            # Find pressure-like columns (values 0-1)
            for col_idx in range(min(20, df_numeric.shape[1])):
                col_data = df_numeric.iloc[:, col_idx].dropna()
                if len(col_data) > 10:
                    # Check if values are in pressure range
                    sample = col_data.values[:20]
                    if np.all((sample >= 0) & (sample <= 1.1)):
                        # Found pressure column, look for quantity in next columns
                        for q_offset in [1, 2, 3]:
                            q_col = col_idx + q_offset
                            if q_col < df_numeric.shape[1]:
                                q_data = df_numeric.iloc[:, q_col].dropna()
                                if len(q_data) > 10 and np.mean(q_data) > 0:
                                    p_ads = col_data.values
                                    q_ads = q_data.values
                                    
                                    # Check for desorption
                                    if q_col + 2 < df_numeric.shape[1]:
                                        p_des_data = df_numeric.iloc[:, q_col + 1].dropna()
                                        q_des_data = df_numeric.iloc[:, q_col + 2].dropna()
                                        if len(p_des_data) > 5 and len(q_des_data) > 5:
                                            p_des = p_des_data.values
                                            q_des = q_des_data.values
                                    break
                        break
        
        # =============================================
        # METHOD 3: Simple column extraction
        # =============================================
        if len(p_ads) < 5:
            st.warning("Auto-detection failed. Using simple column extraction...")
            
            # Clean data
            df_clean = df.apply(pd.to_numeric, errors='coerce')
            df_clean = df_clean.dropna(axis=1, how='all')
            
            if df_clean.shape[1] >= 2:
                # Use first two columns
                col1 = df_clean.iloc[:, 0].dropna()
                col2 = df_clean.iloc[:, 1].dropna()
                
                # Determine which is pressure and which is quantity
                if len(col1) > 10 and len(col2) > 10:
                    # Check if first column looks like pressure
                    if np.mean(col1.values) < 1.0 and np.mean(col2.values) > 1.0:
                        p_ads = col1.values
                        q_ads = col2.values
                    else:
                        p_ads = col2.values
                        q_ads = col1.values
                    
                    # Check for desorption
                    if df_clean.shape[1] >= 4:
                        p_des = df_clean.iloc[:, 2].dropna().values
                        q_des = df_clean.iloc[:, 3].dropna().values
        
        # =============================================
        # VALIDATION
        # =============================================
        if len(p_ads) < 5:
            # Show debug info
            with st.expander("🔍 Debug file structure"):
                st.write("First 30 rows of data:")
                st.dataframe(df.head(30))
                
                st.write("Column summary:")
                for col in range(min(20, df.shape[1])):
                    non_null = df.iloc[:, col].notna().sum()
                    if non_null > 0:
                        sample = df.iloc[:5, col].tolist()
                        st.write(f"Column {col}: {non_null} values | Sample: {sample}")
            
            return None, None, None, None, f"Found only {len(p_ads)} valid adsorption points (need at least 5)"
        
        # Final cleaning
        p_ads = np.array(p_ads, dtype=float)
        q_ads = np.array(q_ads, dtype=float)
        
        # Filter valid ranges
        valid = (p_ads > 0.001) & (p_ads < 0.999) & (q_ads > 0) & np.isfinite(p_ads) & np.isfinite(q_ads)
        p_ads = p_ads[valid]
        q_ads = q_ads[valid]
        
        if len(p_ads) < 5:
            return None, None, None, None, "Not enough valid points after filtering"
        
        # Process desorption
        if len(p_des) > 0 and len(q_des) > 0:
            p_des = np.array(p_des, dtype=float)
            q_des = np.array(q_des, dtype=float)
            valid_des = (p_des > 0.001) & (p_des < 0.999) & (q_des > 0) & np.isfinite(p_des) & np.isfinite(q_des)
            p_des = p_des[valid_des]
            q_des = q_des[valid_des]
            
            if len(p_des) < 5:
                p_des = q_des = None
        else:
            p_des = q_des = None
        
        st.success(f"✅ Extracted {len(p_ads)} adsorption points")
        if p_des is not None:
            st.success(f"✅ Extracted {len(p_des)} desorption points")
        
        return p_ads, q_ads, p_des, q_des, None
        
    except Exception as e:
        return None, None, None, None, f"File reading error: {str(e)}"

def read_xrd_file(file):
    """Read XRD file"""
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
            return None, None, "Insufficient data points (need at least 10)"
        
        # Ensure we have at least 2 columns
        if df.shape[1] < 2:
            return None, None, "File must have at least 2 columns (2θ and Intensity)"
        
        two_theta = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        
        # Basic validation
        if len(two_theta) != len(intensity):
            return None, None, "2θ and Intensity columns have different lengths"
        
        return two_theta, intensity, None
        
    except Exception as e:
        return None, None, f"XRD file reading error: {str(e)}"

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_bet_isotherm(bet_data):
    """Plot adsorption-desorption isotherm"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if bet_data.get("valid", False):
        ads_data = bet_data.get("adsorption_data", {})
        
        p_ads = ads_data.get("p_rel", [])
        q_ads = ads_data.get("q_ads", [])
        
        if len(p_ads) > 0 and len(q_ads) > 0:
            ax.plot(p_ads, q_ads, 'o-', linewidth=2, markersize=4, label='Adsorption')
        
        p_des = ads_data.get("p_des")
        q_des = ads_data.get("q_des")
        
        if p_des is not None and q_des is not None and len(p_des) > 0 and len(q_des) > 0:
            ax.plot(p_des, q_des, 's--', linewidth=2, markersize=4, label='Desorption')
        
        ax.set_xlabel('Relative Pressure (P/P₀)', fontsize=12)
        ax.set_ylabel('Quantity Adsorbed', fontsize=12)
        ax.set_title('Physisorption Isotherm', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add BET parameters if available
        if bet_data.get('surface_area') is not None:
            text = f"Sᴮᴱᵀ = {bet_data['surface_area']:.0f} m²/g\n"
            text += f"C = {bet_data['c_constant']:.0f}\n"
            text += f"R² = {bet_data['r_squared']:.4f}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def plot_xrd_pattern(xrd_data):
    """Plot XRD pattern"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if xrd_data.get("valid", False):
        two_theta = xrd_data.get("two_theta", [])
        intensity = xrd_data.get("intensity", [])
        
        if len(two_theta) > 0 and len(intensity) > 0:
            ax.plot(two_theta, intensity, '-', linewidth=1.5, color='blue')
            
            # Mark peaks
            peaks = xrd_data.get("peaks", [])
            if len(peaks) > 0:
                for peak in peaks:
                    # Find closest point
                    idx = np.argmin(np.abs(np.array(two_theta) - peak))
                    ax.plot(peak, intensity[idx], 'r^', 
                           markersize=8, label='Peak' if peak == peaks[0] else "")
            
            ax.set_xlabel('2θ (degrees)', fontsize=12)
            ax.set_ylabel('Intensity (normalized)', fontsize=12)
            ax.set_title('XRD Pattern', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add XRD parameters
            if xrd_data.get('crystallinity') is not None:
                text = f"Crystallinity = {xrd_data['crystallinity']:.2f}\n"
                text += f"Peaks found = {len(peaks)}"
                ax.text(0.05, 0.95, text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            if len(peaks) > 0:
                ax.legend(fontsize=11)
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Initialize session state
    if 'bet_results' not in st.session_state:
        st.session_state.bet_results = None
    if 'xrd_results' not in st.session_state:
        st.session_state.xrd_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Upload BET and/or XRD data for analysis")
        
        st.markdown("---")
        st.markdown("**File Formats:**")
        st.caption("• BET: Excel (.xls, .xlsx) or CSV (ASAP 2420 format)")
        st.caption("• XRD: CSV, TXT (2 columns: 2θ and Intensity)")
        
        st.markdown("---")
        st.markdown("**About:**")
        st.caption("Scientific BET-XRD morphology analysis")
        st.caption("For journal publication")
        
        # Show Python version
        st.markdown("---")
        st.caption(f"Python {sys.version.split()[0]}")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 BET Isotherm Data")
        bet_file = st.file_uploader(
            "Upload BET file",
            type=['xls', 'xlsx', 'csv', 'txt'],
            key='bet_uploader',
            help="ASAP 2420 format recommended. Adsorption in columns L&M, desorption in N&O"
        )
        
        if bet_file:
            st.success(f"✅ BET file uploaded: {bet_file.name}")
    
    with col2:
        st.subheader("📈 XRD Pattern Data")
        xrd_file = st.file_uploader(
            "Upload XRD file",
            type=['csv', 'txt', 'dat', 'xy'],
            key='xrd_uploader',
            help="Two-column format: 2θ (degrees) and Intensity"
        )
        
        if xrd_file:
            st.success(f"✅ XRD file uploaded: {xrd_file.name}")
    
    # Analyze button
    analyze_clicked = st.button("🚀 Run Scientific Analysis", 
                               type="primary", 
                               use_container_width=True,
                               disabled=(bet_file is None and xrd_file is None))
    
    if analyze_clicked:
        # Clear previous results
        st.session_state.bet_results = None
        st.session_state.xrd_results = None
        
        # Process BET file
        if bet_file:
            with st.spinner("🔬 Analyzing BET data..."):
                p_ads, q_ads, p_des, q_des, error = read_bet_file(bet_file)
                
                if error:
                    st.error(f"❌ BET file error: {error}")
                    
                    # Show file preview for debugging
                    with st.expander("🔍 Debug: Show file preview"):
                        bet_file.seek(0)
                        if bet_file.name.endswith('.xls'):
                            preview_df = pd.read_excel(bet_file, engine='xlrd', nrows=30)
                        elif bet_file.name.endswith('.xlsx'):
                            preview_df = pd.read_excel(bet_file, engine='openpyxl', nrows=30)
                        else:
                            preview_df = pd.read_csv(bet_file, nrows=30)
                        
                        st.dataframe(preview_df)
                        
                        st.write("**Column indices for manual selection:**")
                        st.write("ASAP 2420 format: Adsorption = columns 11,12; Desorption = columns 13,14")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            p_col = st.number_input("Pressure column", 0, 50, 11, key="p_col")
                        with col2:
                            q_col = st.number_input("Quantity column", 0, 50, 12, key="q_col")
                        
                        if st.button("Try manual columns"):
                            try:
                                bet_file.seek(0)
                                if bet_file.name.endswith('.xls'):
                                    df = pd.read_excel(bet_file, engine='xlrd')
                                elif bet_file.name.endswith('.xlsx'):
                                    df = pd.read_excel(bet_file, engine='openpyxl')
                                else:
                                    df = pd.read_csv(bet_file)
                                
                                p_ads = df.iloc[:, p_col].dropna().values
                                q_ads = df.iloc[:, q_col].dropna().values
                                
                                if len(p_ads) >= 5 and len(q_ads) >= 5:
                                    bet_results = analyze_bet_data(p_ads, q_ads)
                                    st.session_state.bet_results = bet_results
                                    st.success("✅ Manual extraction successful!")
                                    st.rerun()
                                else:
                                    st.error("❌ Not enough data in selected columns")
                            except Exception as e:
                                st.error(f"❌ Manual extraction failed: {e}")
                else:
                    # Run analysis
                    bet_results = analyze_bet_data(p_ads, q_ads, p_des, q_des)
                    st.session_state.bet_results = bet_results
                    
                    if bet_results.get("valid", False):
                        st.success(f"✅ BET analysis complete! Surface area: {bet_results['surface_area']:.1f} m²/g")
                    else:
                        st.error(f"❌ BET analysis failed: {bet_results.get('error', 'Unknown error')}")
        
        # Process XRD file
        if xrd_file:
            with st.spinner("🔬 Analyzing XRD data..."):
                two_theta, intensity, error = read_xrd_file(xrd_file)
                
                if error:
                    st.error(f"❌ XRD file error: {error}")
                else:
                    xrd_results = analyze_xrd_data(two_theta, intensity)
                    st.session_state.xrd_results = xrd_results
                    
                    if xrd_results.get("valid", False):
                        st.success(f"✅ XRD analysis complete! Crystallinity: {xrd_results['crystallinity']:.2f}")
                    else:
                        st.error(f"❌ XRD analysis failed: {xrd_results.get('error', 'Unknown error')}")
        
        if bet_file is None and xrd_file is None:
            st.warning("⚠️ Please upload at least one file")
    
    # Display results
    if st.session_state.bet_results or st.session_state.xrd_results:
        st.markdown("---")
        st.header("📊 Scientific Results")
        
        tabs = st.tabs(["BET Analysis", "XRD Analysis", "Morphology Synthesis"])
        
        with tabs[0]:
            if st.session_state.bet_results:
                bet = st.session_state.bet_results
                
                if bet.get("valid", False):
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Surface Area", f"{bet['surface_area']:.1f} m²/g")
                    with col2:
                        st.metric("Pore Volume", f"{bet['pore_volume']:.3f} cm³/g")
                    with col3:
                        st.metric("C Constant", f"{bet['c_constant']:.1f}")
                    with col4:
                        st.metric("Regression R²", f"{bet['r_squared']:.4f}")
                    
                    # Plot
                    st.subheader("Isotherm Plot")
                    fig = plot_bet_isotherm(bet)
                    st.pyplot(fig)
                    
                    # Raw data
                    with st.expander("📋 View extracted data"):
                        if bet.get("adsorption_data"):
                            ads_df = pd.DataFrame({
                                "P/P₀": bet["adsorption_data"].get("p_rel", []),
                                "Q_ads": bet["adsorption_data"].get("q_ads", [])
                            })
                            st.dataframe(ads_df)
                            
                            if bet["adsorption_data"].get("p_des") is not None:
                                des_df = pd.DataFrame({
                                    "P/P₀": bet["adsorption_data"]["p_des"],
                                    "Q_des": bet["adsorption_data"]["q_des"]
                                })
                                st.write("Desorption data:")
                                st.dataframe(des_df)
                else:
                    st.error(f"❌ BET analysis error: {bet.get('error')}")
            else:
                st.info("📭 No BET results available")
        
        with tabs[1]:
            if st.session_state.xrd_results:
                xrd = st.session_state.xrd_results
                
                if xrd.get("valid", False):
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Crystallinity Index", f"{xrd['crystallinity']:.3f}")
                    with col2:
                        st.metric("Ordered Mesopores", 
                                 "✅ Yes" if xrd['ordered_mesopores'] else "❌ No")
                    
                    # Plot
                    st.subheader("XRD Pattern")
                    fig = plot_xrd_pattern(xrd)
                    st.pyplot(fig)
                    
                    # Peaks
                    if len(xrd.get("peaks", [])) > 0:
                        st.subheader("Detected Peaks")
                        peaks_df = pd.DataFrame({
                            "2θ (degrees)": xrd["peaks"]
                        })
                        st.dataframe(peaks_df)
                else:
                    st.error(f"❌ XRD analysis error: {xrd.get('error')}")
            else:
                st.info("📭 No XRD results available")
        
        with tabs[2]:
            st.subheader("🧬 Morphology Synthesis")
            
            bet_valid = st.session_state.bet_results and st.session_state.bet_results.get("valid", False)
            xrd_valid = st.session_state.xrd_results and st.session_state.xrd_results.get("valid", False)
            
            if bet_valid and xrd_valid:
                bet = st.session_state.bet_results
                xrd = st.session_state.xrd_results
                
                # Calculate composite properties
                sa = bet['surface_area']
                cryst = xrd['crystallinity']
                ordered = xrd['ordered_mesopores']
                
                # Material classification
                if sa > 800 and ordered:
                    material_class = "Ordered Mesoporous Material"
                    applications = ["Catalysis", "Adsorption", "Drug Delivery"]
                    journals = ["Chemistry of Materials", "Microporous and Mesoporous Materials"]
                elif sa > 1000:
                    material_class = "Microporous/Hierarchical Material"
                    applications = ["Gas Storage", "Filtration", "Energy Storage"]
                    journals = ["Carbon", "Advanced Functional Materials"]
                elif cryst > 0.7:
                    material_class = "Crystalline Porous Material"
                    applications = ["Catalysis", "Sensing", "Separation"]
                    journals = ["Journal of Materials Chemistry A", "Crystal Growth & Design"]
                elif sa > 500 and cryst > 0.4:
                    material_class = "Semi-crystalline Porous Material"
                    applications = ["General Adsorption", "Support Materials"]
                    journals = ["Materials Chemistry and Physics", "Journal of Porous Materials"]
                else:
                    material_class = "Mixed Morphology Material"
                    applications = ["General Applications"]
                    journals = ["Materials Today Communications"]
                
                # Display synthesis
                st.write("### Material Characterization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Classification", material_class)
                    st.metric("Surface Area", f"{sa:.0f} m²/g")
                    st.metric("Crystallinity", f"{cryst:.2f}")
                
                with col2:
                    st.metric("Porosity", "High" if sa > 500 else "Moderate" if sa > 100 else "Low")
                    st.metric("Structure", "Ordered" if ordered else "Disordered")
                    st.metric("Pore Volume", f"{bet['pore_volume']:.3f} cm³/g")
                
                st.markdown("---")
                st.write("### Suggested Applications")
                for app in applications:
                    st.write(f"• {app}")
                
                st.markdown("---")
                st.write("### Recommended Journals")
                st.info("Based on your material properties, consider submitting to:")
                for journal in journals:
                    st.write(f"• **{journal}**")
                
                # Export data
                st.markdown("---")
                st.write("### 📤 Export Results")
                
                export_data = {
                    "BET_Analysis": bet,
                    "XRD_Analysis": xrd,
                    "Morphology_Synthesis": {
                        "material_class": material_class,
                        "applications": applications,
                        "recommended_journals": journals
                    }
                }
                
                import json
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download Complete Analysis (JSON)",
                    data=json_str,
                    file_name="bet_xrd_analysis.json",
                    mime="application/json"
                )
            
            elif bet_valid:
                st.info("📊 BET data available. Upload XRD data for complete morphology synthesis.")
            elif xrd_valid:
                st.info("📈 XRD data available. Upload BET data for complete morphology synthesis.")
            else:
                st.info("📭 Upload both BET and XRD data for integrated morphology analysis.")
    
    # Footer
    st.markdown("---")
    st.caption("🔬 Scientific BET-XRD Morphology Analyzer | For Journal Publication")
    st.caption("Developed for advanced materials characterization")

if __name__ == "__main__":
    main()
