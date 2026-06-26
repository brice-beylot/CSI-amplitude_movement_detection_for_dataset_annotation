import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from csi_preprocessing import preprocess_dataframe
from scipy.signal import butter, sosfiltfilt

def apply_lowpass_filter_row(df, cutoff_freq=10, fs=100.0):
    """
    Apply a 2nd-order low-pass Butterworth filter to the CSI data contained in the "CSI_processed" column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a "CSI_processed" column.
        cutoff_freq (float, optional): Cutoff frequency for the low-pass filter in Hz. Defaults to 10.
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.

    Returns:
        pd.DataFrame: DataFrame with the filtered CSI data.
    """
    if "CSI_processed" not in df:
        raise KeyError("Missing required key: CSI_processed")
    df_out = df.copy()
    sos = butter(
        N=2,                # Filter order
        Wn=cutoff_freq,     # Single cutoff frequency for low-pass
        btype="lowpass",    # Filter type
        fs=fs,              # Sampling frequency
        output="sos",       # Second-order sections output
    )
    # Apply the filter individually to the array in each row
    df_out["CSI_processed"] = df["CSI_processed"].apply(
        lambda csi_array: sosfiltfilt(sos, csi_array, axis=0)
    )
    return df_out

def load_dataset(file_path):
    """
    Load the dataset from a compressed .pkl file and return a DataFrame.
    
    Args:
        file_path (str): Path to the .pkl file containing the dataset.
    
    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset.
    """
    print(f"Loading data from: {file_path}")

    # Load the dataframe from the .pkl file
    df = pd.read_pickle((file_path), compression="gzip")
    
    """
    print(f"Loaded {file_path} with shape {df.shape}")
    print(f"Data types: {df.dtypes}")
    print(f"Data examples (first 5 rows):\n{df[:5]}")
    
    # Calculate the minimum length across all CSI samples to ensure consistent shapes
    if "CSI" in df.columns :
        min_length = min(csi.shape[0] for csi in df["CSI"])
    else :
        print("no 'CSI' column found for shaping data")
        
    # Convert CSI column to a list of NumPy arrays
    csi_list = [np.array(csi)[:min_length] for csi in df["CSI"]]
    x = np.array(csi_list)
    print(f"Extracted CSI data shape: {x.shape}")
    
    if "labels" in df.columns : 
        labels = df["label"].values
        print(f"Extracted labels shape: {labels.shape}")
        
    for keys in df.columns:
        first_element = df[keys].iloc[0]
        if type(first_element).__name__ not in ['list', 'ndarray']:
            print(f"Unique values for {keys}: {df[keys].unique()}")
    
          
    print("Example of shapes for the first sample : ")
    if "CSI" in df.columns: print(f"Shape of the CSI : {np.array(df['CSI'][0]).shape}")
    if "compensate_gain_per_sample" in df.columns : print(f"Shape of the compensate gain per sample : {np.array(df['compensate_gain_per_sample'][0]).shape}")
    if "fft_gain_per_sample" in df.columns : print(f"Shape of the fft gain per sample : {np.array(df['fft_gain_per_sample'][0]).shape}")
    if "agc_gain_per_sample" in df.columns : print(f"Shape of the agc gain per sample : {np.array(df['agc_gain_per_sample'][0]).shape}")
    if "CSI_processed" in df.columns : print(f"Shape of the CSI processed : {np.array(df['CSI_processed'][0]).shape}")
    """
    
    return df

def change_CSI_processed(df, remove_null=True, apply_gain=True, denoise=True, bandpass=True, low_freq=0.5, high_freq=10.0, standardize=False, denoise_method="hampel_filter") :
    """
    Applies preprocessing steps to the CSI signals present int the "CSI" column of the dataset and regenerates the 'CSI_processed' column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the raw CSI data.
        remove_null (bool, optional): Whether to remove null subcarriers. Defaults to True.
        apply_gain (bool, optional): Whether to apply AGC/FFT gain compensation. Defaults to True.
        denoise (bool, optional): Whether to apply denoising. Defaults to True.
        bandpass (bool, optional): Whether to apply a bandpass filter. Defaults to True.
        low_freq (float, optional): Low cutoff frequency for bandpass in Hz. Defaults to 0.5.
        high_freq (float, optional): High cutoff frequency for bandpass in Hz. Defaults to 10.0.
        standardize (bool, optional): Whether to standardize the data. Defaults to False.
        denoise_method (str, optional): Method used for denoising. Defaults to "hampel_filter".

    Returns:
        pd.DataFrame: A new DataFrame with the updated 'CSI_processed' column.
    """
    #print(df["CSI_processed"][0])
    print("Preprocessing CSI samples from the dataset...")
    df.drop(columns=['CSI_processed'])
    new_df = preprocess_dataframe(df, remove_null=remove_null, apply_gain=apply_gain, denoise=denoise, bandpass=bandpass, low_freq=low_freq, high_freq=high_freq, standardize=standardize, denoise_method=denoise_method)
    return new_df

def join_idle_states(df, index=0) :
    """
    Concatenates the CSI data before an event starts and after an event ends to create a continuous idle state baseline.

    Args:
        df (pd.DataFrame): Input DataFrame with "CSI_processed", "event_start", and "event_end" columns.
        index (int, optional): The row index of the sample to process. Defaults to 0.

    Returns:
        np.ndarray: A 2D array of the concatenated idle state CSI data.
    """
    event_start = df["event_start"][index]
    event_end = df["event_end"][index]
    csi_start = df["CSI_processed"][index][:event_start+1]
    csi_end = df["CSI_processed"][index][event_end:]
    joined_csi = np.concatenate((csi_start,csi_end), axis=0)
    return joined_csi

def calculate_percentile(values, percentile, keep_above=False):
    """
    Calculate percentile value from a list using linear interpolation.
    
    Args:
        values (list): List of numeric values.
        percentile (float): Percentile to calculate (0-100).
        keep_above (bool, optional): If True, sort descending to keep values above the percentile. Defaults to False.
    
    Returns:
        float: The calculated percentile value. Returns 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    
    if keep_above:
        sorted_values = sorted(values, reverse=True)
    else:
        sorted_values = sorted(values)
    n = len(sorted_values)
    p = percentile / 100.0
    k = int((n - 1) * p)
    
    if k >= n - 1:
        return sorted_values[-1]
    
    # Linear interpolation
    frac = (n - 1) * p - k
    return sorted_values[k] * (1 - frac) + sorted_values[k + 1] * frac

def calculate_mean(values):
    """
    Calculate the arithmetic mean of a list of values.
    
    Args:
        values (list): List of numeric values.
    
    Returns:
        float: The calculated mean. Returns 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    return mean

def calculate_variance(values):
    """
    Calculate variance using a numerically stable two-pass algorithm.
    
    Args:
        values (list): List of numeric values.
    
    Returns:
        float: The calculated variance. Returns 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance

def calculate_spatial_turbulence(amplitudes, band, gain_locked=True):
    """
    Calculate spatial turbulence (standard deviation or coefficient of variation) from subcarrier amplitudes for a single packet.
    
    Args:
        amplitudes (list): List of amplitude values (one per subcarrier in the packet).
        band (list): List of subcarrier indices to use for turbulence calculation.
        gain_locked (bool, optional): If True, returns standard deviation. If False, returns coefficient of variation (std/mean). Defaults to True.
    
    Returns:
        float: Calculated turbulence value. Returns 0.0 if band is empty or variance is zero.
    """
    band_mags = [amplitudes[sc] for sc in band if sc < len(amplitudes)]
    
    if not band_mags:
        return 0.0
    
    variance = calculate_variance(band_mags)
    mean = calculate_mean(band_mags)
    
    if variance > 0 and gain_locked :
        turbulence = math.sqrt(variance)
    elif variance > 0 and not gain_locked :
        turbulence = math.sqrt(variance)/mean
    else :
        turbulence = 0
    return turbulence

def find_candidate_windows(csi_packets, all_subcarriers=False, gain_locked=True, window_size=300, band=None, step=25, percentile=5, threshold=None, quietest=True, method="turbulence", weighted=True, sigma=12):
    """
    Partitions the recording into overlapping windows and identifies the quietest or noisiest windows based on variance metrics and a calculated/given threshold.
    
    Args:
        csi_packets (np.ndarray or list): 2D array of shape (n_packets, num_subcarriers) containing CSI amplitudes.
        all_subcarriers (bool, optional): If True, use all subcarriers independently of the value of the 'band' argument. Defaults to False.
        gain_locked (bool, optional): Passed to spatial turbulence calculation. Defaults to True.
        window_size (int, optional): Size of each window in packets. Defaults to 300.
        band (list, optional): List of subcarrier indices to evaluate. If None, uses a default central band. Defaults to None.
        step (int, optional): Step size between window starts in packets. Defaults to 25.
        percentile (float, optional): Percentile threshold for filtering windows. Defaults to 5.
        threshold (float, optional): Hard threshold to use instead of calculating from percentile. Defaults to None.
        quietest (bool, optional): If True, filters windows below the threshold. If False, filters above. Defaults to True.
        method (str, optional): Metric to use: "turbulence" or "amplitude". The "turbulence" method consists in calculating the variance of std or std/mean of subcarriers amplitude for each packets. The "amplitude" method consists in calculating the mean of the variance of subcarriers amplitude for each subcarrier. Defaults to "turbulence".
        weighted (bool, optional): If True, applies Gaussian weighting based on distance to find the central window. Defaults to True.
        sigma (float, optional): Spread parameter for the Gaussian weighting. Defaults to 12.
    
    Returns:
        tuple: (candidate_windows, all_metrics, p_threshold)
            - candidate_windows (list): List of lists [start_idx, end_idx, metric_value] for windows passing the threshold.
            - all_metrics (list): The complete list of calculated metrics for all scanned windows, padded with zeros for the plot.
            - p_threshold (float): The threshold value used to filter the windows.
    """
    n_packets = len(csi_packets)
    
    if n_packets < window_size:
        print(f"Error: Not enough packets ({n_packets}) for window size {window_size}")
        return []
    
    if all_subcarriers :
        default_band = [i for i in range(len(csi_packets[0]))]
    elif band==None:
        default_band = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    else :
        default_band = band
    
    all_windows = []
    
    for start_idx in range(0, n_packets - window_size + 1, step):
        window_packets = csi_packets[start_idx : start_idx + window_size]
        
        if method == "turbulence" :
            turbulences = []
            for packet_idx in range(len(window_packets)):
                packet_mags = window_packets[packet_idx]
                turbulence = calculate_spatial_turbulence(packet_mags, default_band, gain_locked=gain_locked)
                turbulences.append(turbulence)
            window_metric = calculate_variance(turbulences)
            
        elif method == "amplitude" :
            subcarriers_variances = []
            for subcarriers_idx in default_band :
                if subcarriers_idx < len(window_packets[0]) :
                    subcarrier_mags = [window_packets[packet_i][subcarriers_idx] for packet_i in range(len(window_packets))]
                    subcarrier_variance = calculate_variance(subcarrier_mags)
                    subcarriers_variances.append(subcarrier_variance)
            window_metric = sum(subcarriers_variances)/len(subcarriers_variances)
        
        else : 
            print("Method name in argument not valid !")
            return [], [], np.inf
        
        end_idx = start_idx + window_size - 1
        all_windows.append([start_idx, end_idx, window_metric])
    
    if not all_windows:
        print("Error: No windows generated")
        return []

    all_metrics = [w[2] for w in all_windows]
        
    if quietest:
        if threshold==None :
            p_threshold = calculate_percentile(all_metrics, percentile, keep_above=False)
        else :
            p_threshold = threshold
        candidate_windows = [w for w in all_windows if w[2] <= p_threshold]
    else:
        if threshold==None :
            p_threshold = calculate_percentile(all_metrics, percentile, keep_above=True)
        else :
            p_threshold = threshold
        candidate_windows = [w for w in all_windows if w[2] >= p_threshold]
    
    if len(candidate_windows) == 0 :
        print("Error, no proper window was find !")
    
    if weighted : 
        metrics = [w[2] for w in candidate_windows]
        starts = [w[0] for w in candidate_windows]
        weighted_metrics = [0 for _ in range(len(candidate_windows))]

        for i in range(len(candidate_windows)):
            for j in range(len(candidate_windows)):
                distance = abs(starts[i] - starts[j]) / step
                weight = gaussian(distance,sigma=sigma)
                weighted_metrics[i] += metrics[j] * weight

            candidate_windows[i][2] = weighted_metrics[i]
            index = candidate_windows[i][0] // step
            all_metrics[index] = candidate_windows[i][2]
    
    all_metrics = [0 for _ in range(0,int(window_size//2),step)] + all_metrics + [0 for _ in range(0,int(window_size//2),step)]
    
    return candidate_windows, all_metrics, p_threshold


def calculate_nbvi_scores(subcarrier_amplitudes, apply_noise_gate_band=True, alpha=0.75):
    """
    Calculate three complementary Narrow Band Variance Index (NBVI) scores for a single subcarrier.
    
    Args:
        subcarrier_amplitudes (list): List of amplitude values for this subcarrier across all packets.
        apply_noise_gate_band (bool, optional): If True, considers subcarriers with mean < 1e-6 as dead/null. Defaults to True.
        alpha (float, optional): Weighting factor between energy and CV. Defaults to 0.75.
    
    Returns:
        dict: Dictionary containing statistical metrics and the three NBVI scores ('nbvi_classic', 'nbvi_entropy', 'nbvi_mad').
    """
    if not subcarrier_amplitudes or len(subcarrier_amplitudes) == 0:
        return {
            'mean': 0.0, 'std': float('inf'), 'mad': float('inf'), 'entropy': 0.0,
            'nbvi_classic': float('inf'), 'nbvi_entropy': float('inf'), 'nbvi_mad': float('inf'),
        }
    
    mean = sum(subcarrier_amplitudes) / len(subcarrier_amplitudes)
    
    if mean < 1e-6 and apply_noise_gate_band : 
        return {
            'mean': mean, 'std': float('inf'), 'mad': float('inf'), 'entropy': 0.0,
            'nbvi_classic': float('inf'), 'nbvi_entropy': float('inf'), 'nbvi_mad': float('inf'),
        }
    
    variance = calculate_variance(subcarrier_amplitudes)
    std = math.sqrt(variance) if variance > 0 else 0.0
    
    sorted_amps = sorted(subcarrier_amplitudes)
    n = len(sorted_amps)
    median = sorted_amps[n // 2]
    abs_deviations = sorted([abs(a - median) for a in subcarrier_amplitudes])
    mad = abs_deviations[n // 2] 
    
    min_amp = min(subcarrier_amplitudes)
    max_amp = max(subcarrier_amplitudes)
    range_amp = max_amp - min_amp
    
    entropy = 0.0
    if range_amp > 0:
        num_bins = 10
        bins = [0] * num_bins
        for amp in subcarrier_amplitudes:
            bin_idx = int((amp - min_amp) / range_amp * num_bins)
            if bin_idx == num_bins: 
                bin_idx = num_bins - 1
            bins[bin_idx] += 1
        
        for count in bins:
            if count > 0:
                probability = count / n
                entropy -= probability * math.log2(probability)
    
    cv = std / mean 
    nbvi_energy = std / (mean * mean) 
    
    nbvi_classic = alpha * nbvi_energy + (1 - alpha) * cv
    entropy_factor = max(0.5, entropy) 
    nbvi_entropy = nbvi_classic / entropy_factor
    robust_std = mad * 1.4826 if mad > 1e-6 else std 
    cv_mad = robust_std / mean
    energy_mad = robust_std / (mean * mean)
    nbvi_mad = alpha * energy_mad + (1 - alpha) * cv_mad
    
    return {
        'mean': mean, 'std': std, 'mad': mad, 'entropy': entropy,
        'nbvi_classic': nbvi_classic, 'nbvi_entropy': nbvi_entropy, 'nbvi_mad': nbvi_mad,
    }


def apply_noise_gate(subcarrier_metrics, apply_noise_gate_band=False, percentile=15, null_threshold=1.0):
    """
    Apply a noise gate to exclude weak/dead subcarriers and those with infinite NBVI scores.
    
    Args:
        subcarrier_metrics (list): List of dicts (one per subcarrier) containing NBVI scores.
        apply_noise_gate_band (bool, optional): If True, filters out subcarriers below the threshold. Defaults to False.
        percentile (float, optional): Percentile threshold for mean amplitude filtering. Defaults to 15.
        null_threshold (float, optional): Minimum mean amplitude to consider valid. Defaults to 1.0.
    
    Returns:
        list: Filtered list of subcarrier metrics.
    """
    if apply_noise_gate_band :
        valid_means = [m['mean'] for m in subcarrier_metrics if m['mean'] > null_threshold and not math.isinf(m['nbvi_classic'])]
    else :
        valid_means = [m['mean'] for m in subcarrier_metrics if not math.isinf(m['nbvi_classic'])]
    
    if not valid_means:
        print("NBVI: Noise Gate - no valid subcarriers found")
        return []
    
    threshold = calculate_percentile(valid_means, percentile, keep_above=False)
    
    if apply_noise_gate_band :
        filtered = [m for m in subcarrier_metrics if m['mean'] >= threshold and not math.isinf(m['nbvi_classic'])]
    else :
        filtered = [m for m in subcarrier_metrics if not math.isinf(m['nbvi_classic'])]
    
    return filtered


def select_with_spacing_strict(sorted_metrics, k=12, min_spacing=1):
    """
    Select subcarriers while strictly maximizing spectral diversity by enforcing spacing constraints.
    
    Args:
        sorted_metrics (list): List of subcarrier metrics, sorted by NBVI score (best first).
        k (int, optional): Number of subcarriers to select. Defaults to 12.
        min_spacing (int, optional): Minimum spacing between selected subcarrier indices. Defaults to 1.
    
    Returns:
        list: Sorted list of selected subcarrier indices.
    """
    valid_candidates = [c['subcarrier'] for c in sorted_metrics if not math.isinf(c['nbvi_classic'])]
    
    if len(valid_candidates) < k:
        print(f"Warning: Not enough valid candidates ({len(valid_candidates)}) to select {k} subcarriers")
    
    for current_spacing in range(min_spacing, -1, -1):
        selected = []
        for candidate in valid_candidates:
            if len(selected) >= k:
                break
            if selected and any(abs(candidate - s) < current_spacing for s in selected):
                continue
            selected.append(candidate)
        
        if len(selected) >= k:
            selected.sort()
            return selected
    
    selected = [c for c in valid_candidates[:k]]
    selected.sort()
    return selected


def select_with_spacing_clustered(sorted_metrics, k=12, min_spacing=1):
    """
    Select subcarriers using a clustered strategy (best 5 unrestricted, remaining spaced).
    
    Args:
        sorted_metrics (list): List of subcarrier metrics, sorted by NBVI score (best first).
        k (int, optional): Number of subcarriers to select. Defaults to 12.
        min_spacing (int, optional): Minimum spacing between remaining selected subcarrier indices. Defaults to 1.
    
    Returns:
        list: Sorted list of selected subcarrier indices.
    """
    selected = []
    
    for m in sorted_metrics:
        if len(selected) >= 5:
            break
        if not math.isinf(m['nbvi_classic']):
            selected.append(m['subcarrier'])
    
    for m in sorted_metrics[5:]:
        if len(selected) >= k:
            break
        if math.isinf(m['nbvi_classic']):
            continue
        candidate = m['subcarrier']
        if all(abs(candidate - s) >= min_spacing for s in selected):
            selected.append(candidate)
    
    if len(selected) < k:
        for m in sorted_metrics:
            if len(selected) >= k:
                break
            if not math.isinf(m['nbvi_classic']):
                candidate = m['subcarrier']
                if candidate not in selected:
                    selected.append(candidate)
    
    selected.sort()
    return selected


def generate_four_candidate_bands(csi_window, all_subcarriers=False, apply_noise_gate_band=True, guard_band_low=-1, guard_band_high=52, 
                                  dc_subcarrier=-1, band_size=12, alpha=0.75):
    """
    Generate four candidate subcarrier bands using different NBVI scoring and spacing strategies.
    
    Args:
        csi_window (np.ndarray or list): 2D array of CSI data for the baseline window.
        all_subcarriers (bool, optional): If True, returns a single band with all indices. Defaults to False.
        apply_noise_gate_band (bool, optional): If True, filters out noisy/dead subcarriers. Defaults to True.
        guard_band_low (int, optional): Lower guard band index to ignore. Defaults to -1 to ignore no subcarriers.
        guard_band_high (int, optional): Upper guard band index to ignore. Defaults to 52 to ignore no subcarriers.
        dc_subcarrier (int, optional): DC subcarrier index to ignore. Defaults to -1 to ignore no subcarriers.
        band_size (int, optional): Target number of subcarriers per band. Defaults to 12.
        alpha (float, optional): Weighting factor for NBVI scoring. Defaults to 0.75.
    
    Returns:
        dict: Dictionary containing the four generated bands ('band_entropy', 'band_mad', 'band_classic_spaced', 'band_classic') and the filtered metrics.
    """
    if not all_subcarriers :
        n_packets = len(csi_window)
        num_subcarriers = len(csi_window[0]) if n_packets > 0 else 0
        
        all_metrics = []
        for sc in range(num_subcarriers):
            subcarrier_amps = [csi_window[pkt][sc] for pkt in range(n_packets)]
            scores = calculate_nbvi_scores(subcarrier_amps, apply_noise_gate_band=apply_noise_gate_band, alpha=alpha)
            
            if sc < guard_band_low or sc > guard_band_high or sc == dc_subcarrier:
                scores['nbvi_classic'] = float('inf')
                scores['nbvi_entropy'] = float('inf')
                scores['nbvi_mad'] = float('inf')
            
            scores['subcarrier'] = sc
            all_metrics.append(scores)
        
        filtered_metrics = apply_noise_gate(all_metrics, apply_noise_gate_band=apply_noise_gate_band, percentile=15, null_threshold=1.0)
        
        if len(filtered_metrics) < band_size:
            print(f"Error: Not enough valid subcarriers ({len(filtered_metrics)}) to select {band_size}")
            return {
                'band_entropy': [], 'band_mad': [], 'band_classic_spaced': [], 'band_classic': [],
                'filtered_metrics': filtered_metrics,
            }
        
        sorted_entropy = sorted(filtered_metrics, key=lambda x: x['nbvi_entropy'])
        band_entropy = select_with_spacing_strict(sorted_entropy, k=band_size, min_spacing=1)
        
        sorted_mad = sorted(filtered_metrics, key=lambda x: x['nbvi_mad'])
        band_mad = select_with_spacing_clustered(sorted_mad, k=band_size, min_spacing=1)
        
        sorted_classic = sorted(filtered_metrics, key=lambda x: x['nbvi_classic'])
        band_classic_spaced = select_with_spacing_strict(sorted_classic, k=band_size, min_spacing=1)
        
        band_classic = select_with_spacing_clustered(sorted_classic, k=band_size, min_spacing=1)
        
        return {
            'band_entropy': band_entropy,
            'band_mad': band_mad,
            'band_classic_spaced': band_classic_spaced,
            'band_classic': band_classic,
            'filtered_metrics': filtered_metrics,
        }
    elif all_subcarriers :
        return {
            'default_band': [i for i in range(len(csi_window[0]))],
        }

def calculate_moving_variance(csi_data, band, gain_locked=True, window_size=100):
    """
    Calculate the moving variance of spatial turbulence values over a sliding window.
    
    Args:
        csi_data (np.ndarray or list): 2D array of CSI data.
        band (list): List of selected subcarrier indices.
        gain_locked (bool, optional): Passed to spatial turbulence calculation. Defaults to True.
        window_size (int, optional): Size of the sliding window. Defaults to 100.
    
    Returns:
        list: Series of calculated moving variance values.
    """
    n_packets = len(csi_data)
    turbulences = []
    for pkt_idx in range(n_packets):
        packet_mags = csi_data[pkt_idx]
        turbulence = calculate_spatial_turbulence(packet_mags, band, gain_locked=gain_locked)
        turbulences.append(turbulence)
    
    if len(turbulences) < window_size:
        return []
    
    mv_values = []
    for i in range(window_size, len(turbulences)):
        window = turbulences[i - window_size : i]
        variance = calculate_variance(window)
        mv_values.append(variance)
    return mv_values

def calculate_moving_mean(csi_data, band, window_size=100):
    """
    Calculate the moving mean of the variance of amplitude values for the specified subcarriers.
    
    Args:
        csi_data (np.ndarray or list): 2D array of CSI data.
        band (list): List of selected subcarrier indices.
        window_size (int, optional): Size of the sliding window. Defaults to 100.
    
    Returns:
        list: Series of calculated moving mean values.
    """
    if len(csi_data) < window_size:
        return []
    
    mm_values = []
    for i in range(window_size, len(csi_data)):
        subcarriers_variances = []
        for subcarriers_idx in band :
            subcarrier_mags = [csi_data[packet_i][subcarriers_idx] for packet_i in [k for k in range(i - window_size,i)]]
            subcarrier_variance = calculate_variance(subcarrier_mags)
            subcarriers_variances.append(subcarrier_variance)
        mean = calculate_mean(subcarriers_variances)
        mm_values.append(mean)
    return mm_values

def validate_band(csi_data, band, gain_locked=True, mvs_window_size=100, 
                  validation_percentile=95, validation_factor=1.1, method="turbulence"):
    """
    Validate a subcarrier band by calculating its false positive rate using a baseline moving variance or mean metric.
    
    Args:
        csi_data (np.ndarray or list): 2D array of calibration CSI data.
        band (list): List of subcarrier indices to validate.
        gain_locked (bool, optional): Passed to metrics calculations. Defaults to True.
        mvs_window_size (int, optional): Window size for moving metric calculation. Defaults to 100.
        validation_percentile (float, optional): Percentile used to establish the baseline threshold. Defaults to 95.
        validation_factor (float, optional): Multiplier applied to the threshold for sensitivity tuning. Defaults to 1.1.
        method (str, optional): Metric to use: "turbulence" or "amplitude". Defaults to "turbulence".
    
    Returns:
        tuple: (fp_rate, window_values, adaptive_threshold)
            - fp_rate (float): Calculated false positive rate (0.0 to 1.0).
            - window_values (list): Full series of the moving metrics calculated.
            - adaptive_threshold (float): The final calculated threshold.
    """
    n_packets = len(csi_data)
    
    if n_packets < mvs_window_size:
        return 0.0, [], np.inf
    
    if method=="turbulence" :
        window_values = calculate_moving_variance(csi_data, band, gain_locked=gain_locked, window_size=mvs_window_size)
    elif method=="amplitude" :
        window_values = calculate_moving_mean(csi_data, band, window_size=mvs_window_size)
    else : 
        print("Method name in argument not valid !")
        return  1.0, [], np.inf
    
    if not window_values:
        print("No windows values found for the validating band process !")
        return 1.0, [], np.inf 
    
    adaptive_threshold = calculate_percentile(window_values, validation_percentile, keep_above=False) * validation_factor
    fp_count = sum(1 for window in window_values if window > adaptive_threshold)
    fp_rate = fp_count / len(window_values)
    
    return fp_rate, window_values, adaptive_threshold


def select_best_band(csi_data, candidate_bands_dict, all_subcarriers=False, gain_locked=True, mvs_window_size=100,
                     acceptable_fp_threshold=0.05, factor=1.1, method="turbulence"):
    """
    Evaluates candidate subcarrier bands and selects the optimal one based on empirical false positive rates.
    
    Args:
        csi_data (np.ndarray or list): Full calibration CSI data.
        candidate_bands_dict (dict): Dictionary mapping band names to lists of subcarrier indices.
        all_subcarriers (bool, optional): If True, defaults to assessing a single band with all subcarriers. Defaults to False.
        gain_locked (bool, optional): Passed to band validation. Defaults to True.
        mvs_window_size (int, optional): Window size for moving variance calculation. Defaults to 100.
        acceptable_fp_threshold (float, optional): Maximum acceptable false positive rate. Defaults to 0.05.
        factor (float, optional): Multiplier for the validation threshold. Defaults to 1.1.
        method (str, optional): Metric to use: "turbulence" or "amplitude". Defaults to "turbulence".
    
    Returns:
        dict: Results dictionary containing 'best_band', 'best_fp_rate', 'best_mv_values', 'best_band_name', 'all_results', and 'threshold'.
            - best_band (list) : list of subcarrier index for the selected band.
            - best_fp_rate (float) : rate of false positive regarding the movement detection in the full calibration CSI data for the selected band
            - best_mv_values (list) : list of the calculated metric values (either variance of turbulence or mean of variance) for each moving window
            - all_results (dict) : dictionary containing 'band', 'fp_rate', 'mv_values', 'threshold' for each candidate subcarriers band
            - threshold (float) : threshold value used to decide if a window contains movement or not
    """
    if not all_subcarriers :
        band_names = ['band_entropy', 'band_mad', 'band_classic_spaced', 'band_classic']
    elif all_subcarriers :
        band_names = ['default_band']
    
    best_band = None
    best_fp_rate = 1.0
    best_mv_values = []
    best_band_name = ""
    all_results = {}
    best_threshold = np.inf
    
    for band_name in band_names:
        band = candidate_bands_dict[band_name]
        
        if not band or len(band) == 0:
            all_results[band_name] = {'band': band, 'fp_rate': None, 'mv_values': []}
            continue
        
        fp_rate, mv_values, threshold = validate_band(csi_data, band, gain_locked=gain_locked, mvs_window_size=mvs_window_size, validation_factor=factor, method=method)
        all_results[band_name] = {'band': band, 'fp_rate': fp_rate, 'mv_values': mv_values}
        
        override = False
        
        if best_band is None:
            override = True
        elif fp_rate <= acceptable_fp_threshold:
            if best_fp_rate > acceptable_fp_threshold:
                override = True
            elif fp_rate <= best_fp_rate and threshold < best_threshold :
                override = True
        else:
            if fp_rate < best_fp_rate:
                override = True
        
        if override:
            best_band = band
            best_fp_rate = fp_rate
            best_mv_values = mv_values
            best_band_name = band_name
            best_threshold = threshold
            
    return {
        'best_band': best_band,
        'best_fp_rate': best_fp_rate,
        'best_mv_values': best_mv_values,
        'best_band_name': best_band_name,
        'all_results': all_results,
        'threshold': best_threshold
    }

def complete_nbvi_calibration(csi_data, all_subcarriers=False, gain_locked=True, apply_noise_gate_band=True, window_size=300, step=25, percentile=5, mvs_window_size=100, factor=1.1, method="turbulence"):
    """
    Executes the complete calibration workflow: finds baseline windows, generates band candidates, and selects the optimal band.
    
    Args:
        csi_data (np.ndarray or list): 2D array of calibration CSI data.
        all_subcarriers (bool, optional): If True, skips band optimization and uses all subcarriers. Defaults to False.
        gain_locked (bool, optional): Passed down to turbulence calculations. Defaults to True.
        apply_noise_gate_band (bool, optional): Whether to use noise gating during band generation. Defaults to True.
        window_size (int, optional): Window size for finding candidate windows. Defaults to 300.
        step (int, optional): Step size between candidate windows. Defaults to 25.
        percentile (float, optional): Percentile threshold for filtering quiet windows. Defaults to 5.
        mvs_window_size (int, optional): Window size for moving metrics validation. Defaults to 100.
        factor (float, optional): Threshold scaling factor. Defaults to 1.1.
        method (str, optional): Metric to use: "turbulence" or "amplitude". Defaults to "turbulence".
    
    Returns:
        dict: The final selection results dictionary containing 'best_band', 'best_fp_rate', 'best_mv_values', 'best_band_name', 'all_results', and 'threshold'.
            - best_band (list) : list of subcarrier index for the selected band.
            - best_fp_rate (float) : rate of false positive regarding the movement detection in the full calibration CSI data for the selected band
            - best_mv_values (list) : list of the calculated metric values (either variance of turbulence or mean of variance) for each moving window
            - all_results (dict) : dictionary containing 'band', 'fp_rate', 'mv_values', 'threshold' for each candidate subcarriers band
            - threshold (float) : threshold value used to decide if a window contains movement or not
    """
    if method=="turbulence" or method=="amplitude" :
        candidate_windows, variances, p_threshold = find_candidate_windows(csi_data, all_subcarriers=all_subcarriers, gain_locked=gain_locked, window_size=window_size, step=step, percentile=percentile, quietest=True, method=method, weighted=False)
    else :
        print("Method name in argument not valid !")
        return {}
    
    if not candidate_windows:
        print("ERROR: No quiet windows found!")
        return {}
    
    best_index_window = np.argmin([window[2] for window in candidate_windows])
    best_window_start, best_window_end, best_variance = candidate_windows[best_index_window]
    csi_window = csi_data[best_window_start : best_window_end + 1]
    
    candidates = generate_four_candidate_bands(csi_window, all_subcarriers=all_subcarriers, band_size=12, alpha=0.75, apply_noise_gate_band=apply_noise_gate_band)
    selection_result = select_best_band(csi_data, candidates, all_subcarriers=all_subcarriers, gain_locked=gain_locked, mvs_window_size=mvs_window_size, factor=factor, method=method)
    
    return selection_result

def selected_band_csi(n_packets, band) :
    """
    Extracts only the specified subcarriers in the band from a given CSI signal.
    
    Args:
        n_packets (np.ndarray or list): 2D array of CSI data of shape (num_packets, total_subcarriers).
        band (list): List of indices corresponding to the subcarriers to retain.
    
    Returns:
        np.ndarray: 2D array of shape (num_packets, len(band)) containing only the selected subcarriers.
    """
    print(f"Shape of sample before selected band : ({len(n_packets)}, {len(n_packets[0])})")
    print(f"selected band : {band}")
    new_n_packets = []
    for packet in n_packets :
        new_n_packets.append([packet[subcarrier] for subcarrier in band])
    new_n_packets = np.array(new_n_packets)
    print(f"Shape of sample after selected band : ({len(new_n_packets)}, {len(new_n_packets[0])})")
    return new_n_packets

def reshaping(database, band) :
    """
    Iterates over an entire dataset of CSI samples and filters each one to only contain the chosen subcarrier band.
    
    Args:
        database (list or pd.Series): Series containing 2D arrays of CSI data for each sample.
        band (list): List of subcarrier indices to keep.
    
    Returns:
        list: A new list where each element is a filtered 2D NumPy array for a given sample.
    """
    new_database = []
    print(f"selected band : {band}") 
    for sample in range(len(database)) :
        new_database.append(selected_band_csi(database[sample], band))
    return new_database

def gaussian(distance, sigma=12):
    """
    Compute a Gaussian weight for a given distance to prioritize centralized windows.
    
    Args:
        distance (float): The distance value to be weighted.
        sigma (float, optional): The spread parameter of the Gaussian curve. Defaults to 12.
        
    Returns:
        float: The calculated Gaussian weight.
    """
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))

def mean_time_differences(df, gain_locked=True, all_subcarriers=True, apply_noise_gate_band=False, window_calibration_size=300, window_detection_size=300, mvs_window_size=100, step_calibration=25, step_detection=10, factor=1.1, percentile_calibration=5, percentile_detection=35, method="amplitude", calibration=False, weighted=True, sigma=12) :
    """
    Evaluates the algorithm's detection accuracy by computing the mean time difference between annotated window centers and detected window centers for each CSI sample.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'CSI_processed', 'event_start', and 'event_end'.
        gain_locked (bool, optional): Used to choose how to calculate turbulence (std of subcarriers amplitude for each packet if True, std/mean of subcarriers amplitude for each packet if False). Defaults to True.
        all_subcarriers (bool, optional): If True, skips band optimization and uses all subcarriers. Defaults to True.
        apply_noise_gate_band (bool, optional): Whether to use noise gating during band generation. Defaults to False.
        window_calibration_size (int, optional): Window size for calibration. Defaults to 300.
        window_detection_size (int, optional): Window size for detection. Defaults to 300.
        mvs_window_size (int, optional): Moving window size for validation metric calculation. Defaults to 100.
        step_calibration (int, optional): Step size for calibration. Defaults to 25.
        step_detection (int, optional): Step size for detection. Defaults to 10.
        factor (float, optional): Multiplier for calibration threshold. The higher the factor is, the less movement will be detected. It must be strictly greater than 1. Defaults to 1.1.
        percentile_calibration (float, optional): Percentile used for the choice of the window for calibration filter. Defaults to 5.
        percentile_detection (float, optional): Percentile used to fix the threshold of movement detection when calibration=False. Defaults to 35.
        method (str, optional): Metric to use for the detection of movement for each moving window : "turbulence" or "amplitude". Defaults to "amplitude".
        calibration (bool, optional): If True, used a CSI calibration signal without movement to fix a threshold for the movement detection in real time. Defaults to False.
        weighted (bool, optional): If True, applies Gaussian weighting during detection to take the most centered window. Defaults to True.
        sigma (float, optional): Spread parameter for Gaussian weighting. Defaults to 12.
    
    Returns:
        tuple: (mean_difference, missed_detections_count)
            - mean_difference (float): Average absolute error in packets between true and predicted window centers.
            - missed_detections_count (int): Number of samples where no motion was detected (this metric is usefull only when calibration=True)
    """
    time_distances = []
    nb_no_motion_detected = 0
    for sample_idx in range(len(df["CSI_processed"])) :
        if not df["non_acceptable_signal_quality"][sample_idx] :
            if calibration :
                if df.columns.isin(["event_start","event_end"]).any():
                    joined_csi = join_idle_states(df, index=sample_idx)
                    result = complete_nbvi_calibration(joined_csi, all_subcarriers=all_subcarriers, gain_locked=gain_locked, apply_noise_gate_band=apply_noise_gate_band, window_size=window_calibration_size, step=step_calibration, percentile=percentile_calibration, mvs_window_size=mvs_window_size, factor=factor, method=method)
                    candidate_windows, variances, p_threshold = find_candidate_windows(df['CSI_processed'][sample_idx], all_subcarriers=all_subcarriers, gain_locked=gain_locked, window_size=window_detection_size, band=result['best_band'], step=step_detection, percentile=percentile_detection, threshold=result['threshold'], quietest=False, method=method, weighted=weighted, sigma=sigma)
                else :
                    print("Calibration is not available because no event_start and event_end columns name were found in the dataframe, not enabling the creation of an idle state !")
                    return np.inf, np.inf
            else :
                candidate_windows, variances, p_threshold = find_candidate_windows(df["CSI_processed"][sample_idx], all_subcarriers=all_subcarriers, gain_locked=gain_locked, window_size=window_detection_size, band=None, step=step_detection, percentile=percentile_detection, threshold=None, quietest=False, method=method, weighted=weighted, sigma=sigma)
            if len(candidate_windows) > 0 :
                metrics = [window[2] for window in candidate_windows]
                best_window = candidate_windows[np.argmax(metrics)]
                time_distance = abs((best_window[0]-df["event_start"][sample_idx]+best_window[1]-df["event_end"][sample_idx])//2)
                print(f"TIME DISTANCE FOR SAMPLE {sample_idx}: {time_distance}")
                time_distances.append(time_distance)
            else :
                nb_no_motion_detected += 1
                print("NO MOTION WINDOW HAS BEEN DETECTED")
    return sum(time_distances)/len(time_distances), nb_no_motion_detected

def create_filtered_database(df, gain_locked=True, all_subcarriers=True, apply_noise_gate_band=False, window_calibration_size=300, window_detection_size=300, mvs_window_size=100, step_calibration=25, step_detection=10, factor=1.1, percentile_calibration=5, percentile_detection=35, method="turbulence",calibration=False, weighted=True, sigma=12) :
    """
    Creates a new DataFrame retaining only valid rows and create/overwrites 'event_start' and 'event_end' columns with algorithmically detected event boundaries.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'CSI_processed', 'event_start', and 'event_end'.
        gain_locked (bool, optional): Used to choose hpw to calculate turbulence (std of subcarriers amplitude for each packet if True, std/mean of subcarriers amplitude for each packet if False). Defaults to True.
        all_subcarriers (bool, optional): If True, skips band optimization and uses all subcarriers. Defaults to True.
        apply_noise_gate_band (bool, optional): Whether to use noise gating during band generation. Defaults to False.
        window_calibration_size (int, optional): Window size for calibration. Defaults to 300.
        window_detection_size (int, optional): Window size for detection. Defaults to 300.
        mvs_window_size (int, optional): Moving window size for validation metric calculation. Defaults to 100.
        step_calibration (int, optional): Step size for calibration. Defaults to 25.
        step_detection (int, optional): Step size for detection. Defaults to 10.
        factor (float, optional): Multiplier for calibration threshold. The higher the factor is, the less movement will be detected. It must be strictly greater than 1. Defaults to 1.1.
        percentile_calibration (float, optional): Percentile used for the choice of the window for calibration filter. Defaults to 5.
        percentile_detection (float, optional): Percentile used to fix the threshold of movement detection when calibration=False. Defaults to 35.
        method (str, optional): Metric to use for the detection of movement for each moving window : "turbulence" or "amplitude". Defaults to "amplitude".
        calibration (bool, optional): If True, used a CSI calibration signal without movement to fix a threshold for the movement detection in real time. Defaults to False.
        weighted (bool, optional): If True, applies Gaussian weighting during detection to take the most centered window. Defaults to True.
        sigma (float, optional): Spread parameter for Gaussian weighting. Defaults to 12.
    
    Returns:
        pd.DataFrame: A new, filtered DataFrame featuring updated 'event_start' and 'event_end' boundaries based on the detection phase.
    """
    print("Annotating the samples by finding the best movement window for each of them...")
    new_df = df.copy()
    new_df['type_annotation'] = 'manually'
    for sample_idx in range(len(new_df["CSI_processed"])) :
        if not new_df["non_acceptable_signal_quality"][sample_idx] :
            if calibration :
                if df.columns.isin(["event_start","event_end"]).any():
                    joined_csi = join_idle_states(new_df, index=sample_idx)
                    result = complete_nbvi_calibration(joined_csi, all_subcarriers=all_subcarriers, gain_locked=gain_locked, apply_noise_gate_band=apply_noise_gate_band, window_size=window_calibration_size, step=step_calibration, percentile=percentile_calibration, mvs_window_size=mvs_window_size, factor=factor, method=method)
                    candidate_windows, variances, p_threshold = find_candidate_windows(new_df['CSI_processed'][sample_idx], all_subcarriers=all_subcarriers, gain_locked=gain_locked, window_size=window_detection_size, band=result['best_band'], step=step_detection, percentile=percentile_detection, threshold=result['threshold'], quietest=False, method=method, weighted=weighted, sigma=sigma)
                else :
                    print("Calibration is not available because no event_start and event_end columns name were found in the dataframe, not enabling the creation of an idle state !")
                    return df
            else :
                candidate_windows, variances, p_threshold = find_candidate_windows(new_df["CSI_processed"][sample_idx], all_subcarriers=all_subcarriers, gain_locked=gain_locked, window_size=window_detection_size, band=None, step=step_detection, percentile=percentile_detection, threshold=None, quietest=False, method=method, weighted=weighted, sigma=sigma)
            if len(candidate_windows) > 0 :
                metrics = [window[2] for window in candidate_windows]
                best_window = candidate_windows[np.argmax(metrics)]
                new_df.loc[sample_idx, 'event_start'] = best_window[0]
                new_df.loc[sample_idx, 'event_end'] = best_window[1]
                new_df.loc[sample_idx, 'type_annotation'] = 'automatically'
            else :
                pass
    
    new_df = new_df[new_df["non_acceptable_signal_quality"] == False]
    return new_df

def change_file_name(file_path, remove_null=True, apply_gain=True, denoise=True, bandpass=True, 
                     low_freq=0.5, high_freq=10.0, standardize=False, denoise_method="hampel_filter",
                     all_subcarriers=True, gain_locked=True, apply_noise_gate_band=False, 
                     window_calibration_size=300, window_detection_size=300, mvs_window_size=100, 
                     step_calibration=25, step_detection=10, factor=1.10, percentile_calibration=5, 
                     percentile_detection=35, sigma=12, method="amplitude", calibration=False, weighted=True):
    """
    Generates a descriptive filename string dynamically based on the current pipeline configuration variables.
    
    Args:
        file_path (str): Original input file path containing the Dataset.
        remove_null (bool, optional): If True, deletes the 12 null subcarriers from the 64. Defaults to True.
        apply_gain (bool, optional): If True, gain is applied to the CSI signal to have correct subcarriers amplitude. Defaults to True.
        denoise (bool, optional): If True, apply a denoise method on the CSI signal. This method is specified in the 'denoise_method' argument. Defaults to True.
        bandpass (bool, optional): If True, apply a bandpass filter on the CSI signal. Defaults to True.
        low_freq (float, optional): Low frequence of the bandpass filter. Defaults to 0.5.
        high_freq (float, optional): High frequence of the bandpass filter. Defaults to 10.0.
        standardize (bool, optional): If True, substract the mean from the CSI signal amplitude and divide everything by the standard deviation, for each subcarrier. Defaults to False.
        denoise_method (str, optional): Name of the denoise method between "median_filter", "hampel_filter". Defaults to "hampel_filter".
        gain_locked (bool, optional): Used to choose hpw to calculate turbulence (std of subcarriers amplitude for each packet if True, std/mean of subcarriers amplitude for each packet if False). Defaults to True.
        all_subcarriers (bool, optional): If True, skips band optimization and uses all subcarriers. Defaults to True.
        apply_noise_gate_band (bool, optional): Whether to use noise gating during band generation. Defaults to False.
        window_calibration_size (int, optional): Window size for calibration. Defaults to 300.
        window_detection_size (int, optional): Window size for detection. Defaults to 300.
        mvs_window_size (int, optional): Moving window size for validation metric calculation. Defaults to 100.
        step_calibration (int, optional): Step size for calibration. Defaults to 25.
        step_detection (int, optional): Step size for detection. Defaults to 10.
        factor (float, optional): Multiplier for calibration threshold. The higher the factor is, the less movement will be detected. It must be strictly greater than 1. Defaults to 1.1.
        percentile_calibration (float, optional): Percentile used for the choice of the window for calibration filter. Defaults to 5.
        percentile_detection (float, optional): Percentile used to fix the threshold of movement detection when calibration=False. Defaults to 35.
        method (str, optional): Metric to use for the detection of movement for each moving window : "turbulence" or "amplitude". Defaults to "amplitude".
        calibration (bool, optional): If True, used a CSI calibration signal without movement to fix a threshold for the movement detection in real time. Defaults to False.
        weighted (bool, optional): If True, applies Gaussian weighting during detection to take the most centered window. Defaults to True.
        sigma (float, optional): Spread parameter for Gaussian weighting. Defaults to 12.
        
    Returns:
        str: Generated system file path for the output `.pkl` dataframe.
    """
    print("Saving the new dataset to a new file dynamically named based on the parameters...")
    dir_name = os.path.dirname(file_path)
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

    suffixes = ["automatically_indexed"]

    if not remove_null: suffixes.append("not_remove_null")
    if not apply_gain: suffixes.append("not_apply_gain")
    if not all_subcarriers: suffixes.append("not_all_subcarriers")
    if not gain_locked: suffixes.append("not_gain_locked")
    if not weighted: suffixes.append("not_weighted")
    if not bandpass: suffixes.append("not_bandpass")

    if standardize: suffixes.append("standardize")
    if apply_noise_gate_band: suffixes.append("apply_noise_gate_band")

    if not denoise:
        suffixes.append("not_denoise")
    elif denoise_method != "hampel_filter":
        suffixes.append(denoise_method)

    if bandpass:
        if round(low_freq, 1) != 0.5: suffixes.append(f"low_freq_{low_freq}")
        if round(high_freq, 1) != 10.0: suffixes.append(f"high_freq_{high_freq}")

    if method != "amplitude":
        suffixes.append(f"method_{method}")
        
    if int(window_detection_size) != 300: suffixes.append(f"window_detection_size_{window_detection_size}")
    if int(step_detection) != 10: suffixes.append(f"step_detection_{step_detection}")
    if round(sigma, 2) != 12.0: suffixes.append(f"sigma_{sigma}")

    if calibration:
        suffixes.append("calibration")
        if int(window_calibration_size) != 300: suffixes.append(f"window_calibration_size_{window_calibration_size}")
        if int(mvs_window_size) != 100: suffixes.append(f"mvs_window_size_{mvs_window_size}")
        if int(step_calibration) != 25: suffixes.append(f"step_calibration_{step_calibration}")
        if round(factor, 2) != 1.10: suffixes.append(f"factor_{factor}")
        if int(percentile_calibration) != 5: suffixes.append(f"percentile_calibration_{percentile_calibration}")
    else:
        if int(percentile_detection) != 35: suffixes.append(f"percentile_detection_{percentile_detection}")

    new_file_name = "_".join(suffixes) + ".pkl"
    
    return os.path.join(dir_name, f"{file_name_without_ext}_{new_file_name}")

def complete_dataframe_annotation(file_path: str,
    remove_null: bool = True,
    apply_gain: bool = True,
    denoise: bool = True,
    bandpass: bool = True,
    standardize: bool = False,
    denoise_method: str = "hampel_filter",
    low_freq: float = 0.5,
    high_freq: float = 10.0,
    all_subcarriers: bool = True, 
    gain_locked: bool = True,
    apply_noise_gate_band: bool = False,
    window_calibration_size: int = 300,
    window_detection_size: int = 300,
    mvs_window_size: int = 100,
    step_calibration: int = 25,
    step_detection: int = 10,
    factor: float = 1.10,
    percentile_calibration: int = 5,
    percentile_detection: int = 35,
    sigma: float = 12, 
    method: str = "amplitude",
    calibration: bool = False,
    weighted: bool = True) :
    """
    Main wrapper pipeline that runs dataset loading, parameter filtering, pre-processing, event detection and event window database creation
    
    Args:
        file_path (str): Original input file path containing the Dataset.
        remove_null (bool, optional): If True, deletes the 12 null subcarriers from the 64. Defaults to True.
        apply_gain (bool, optional): If True, gain is applied to the CSI signal to have correct subcarriers amplitude. Defaults to True.
        denoise (bool, optional): If True, apply a denoise method on the CSI signal. This method is specified in the 'denoise_method' argument. Defaults to True.
        bandpass (bool, optional): If True, apply a bandpass filter on the CSI signal. Defaults to True.
        low_freq (float, optional): Low frequence of the bandpass filter. Defaults to 0.5.
        high_freq (float, optional): High frequence of the bandpass filter. Defaults to 10.0.
        standardize (bool, optional): If True, substract the mean from the CSI signal amplitude and divide everything by the standard deviation, for each subcarrier. Defaults to False.
        denoise_method (str, optional): Name of the denoise method between "median_filter", "hampel_filter". Defaults to "hampel_filter".
        gain_locked (bool, optional): Used to choose hpw to calculate turbulence (std of subcarriers amplitude for each packet if True, std/mean of subcarriers amplitude for each packet if False). Defaults to True.
        all_subcarriers (bool, optional): If True, skips band optimization and uses all subcarriers. Defaults to True.
        apply_noise_gate_band (bool, optional): Whether to use noise gating during band generation. Defaults to False.
        window_calibration_size (int, optional): Window size for calibration. Defaults to 300.
        window_detection_size (int, optional): Window size for detection. Defaults to 300.
        mvs_window_size (int, optional): Moving window size for validation metric calculation. Defaults to 100.
        step_calibration (int, optional): Step size for calibration. Defaults to 25.
        step_detection (int, optional): Step size for detection. Defaults to 10.
        factor (float, optional): Multiplier for calibration threshold. The higher the factor is, the less movement will be detected. It must be strictly greater than 1. Defaults to 1.1.
        percentile_calibration (float, optional): Percentile used for the choice of the window for calibration filter. Defaults to 5.
        percentile_detection (float, optional): Percentile used to fix the threshold of movement detection when calibration=False. Defaults to 35.
        method (str, optional): Metric to use for the detection of movement for each moving window : "turbulence" or "amplitude". Defaults to "amplitude".
        calibration (bool, optional): If True, used a CSI calibration signal without movement to fix a threshold for the movement detection in real time. Defaults to False.
        weighted (bool, optional): If True, applies Gaussian weighting during detection to take the most centered window. Defaults to True.
        sigma (float, optional): Spread parameter for Gaussian weighting. Defaults to 12.
        
    Returns:
        pd.DataFrame: A new, filtered DataFrame featuring updated 'event_start' and 'event_end' boundaries based on the detection phase.
    """
    df = load_dataset(file_path)
    
    filters = {
        "segment_type": "action",
        "action": "fall",
        "label": "fall",
        "non_acceptable_signal_quality": False}
    
    for col, target_val in filters.items():
        if col in df.columns and target_val in df[col].unique():
            df = df[df[col] == target_val]      
    df = df.reset_index(drop=True)
    
    df = change_CSI_processed(df, remove_null=remove_null, apply_gain=apply_gain, denoise=denoise, bandpass=bandpass, low_freq=low_freq, high_freq=high_freq, standardize=standardize, denoise_method=denoise_method)
    new_df = create_filtered_database(df, all_subcarriers=all_subcarriers, gain_locked=gain_locked, apply_noise_gate_band=apply_noise_gate_band, window_calibration_size=window_calibration_size, window_detection_size=window_detection_size, mvs_window_size=mvs_window_size, step_calibration=step_calibration, step_detection=step_detection, factor=factor, percentile_calibration=percentile_calibration, percentile_detection=percentile_detection, sigma=sigma, method=method, calibration=calibration, weighted=weighted)
    """
    new_file_path = change_file_name(file_path, remove_null=remove_null, apply_gain=apply_gain, denoise=denoise, bandpass=bandpass, low_freq=low_freq, high_freq=high_freq, standardize=standardize, denoise_method=denoise_method,
                                     all_subcarriers=all_subcarriers, gain_locked=gain_locked, apply_noise_gate_band=apply_noise_gate_band, window_calibration_size=window_calibration_size, window_detection_size=window_detection_size, mvs_window_size=mvs_window_size, 
                                     step_calibration=step_calibration, step_detection=step_detection, factor=factor, percentile_calibration=percentile_calibration, percentile_detection=percentile_detection, sigma=sigma, method=method, calibration=calibration, weighted=weighted)
    new_df.to_pickle(new_file_path)
    print(f"New database saved as: {new_file_path}")
    """
    return new_df

def plot_csi_processed_sample(df,index=0) :
    """
    Plots the processed CSI signal for a given DataFrame row alongside overlayed event boundaries.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'CSI_processed', 'event_start', and 'event_end'.
        index (int, optional): Target row to analyze and map. Defaults to 0.
        
    Returns:
        None: Renders a matplotlib visualization of the extracted indices.
    """
    print("Affichage du premier exemple")
    csi_processed = df["CSI_processed"]
    if df.columns.isin(["event_start", "event_end"]).any():
        event_start = df["event_start"]
        event_end = df["event_end"]
    plt.figure(figsize=(16,8))
    plt.plot(csi_processed[index])
    if df.columns.isin(["event_start", "event_end"]).any():
        plt.axvline(x=event_start[index], color='g', linestyle='--', label='Event Start')
        plt.axvline(x=event_end[index], color='r', linestyle='--', label='Event End')
    plt.legend()
    plt.show()
    pass

def plot_variance(window, variances, p_threshold, step) :
    """
    Renders a line plot comparing continuous variance output streams relative to an evaluated target threshold limit line.
    
    Args:
        window (list or tuple): The predicted window bounds, formatted as `(start_idx, end_idx, ...)`.
        variances (list): Continuous stream vector of output metric evaluations.
        p_threshold (float): Detected line cutoff limit value.
        step (int): Index length jump required for matching visual length outputs based on configuration parameters.
        
    Returns:
        None: Triggers a localized matplotlib mapping sequence visualizing metric peaks bounding target domains.
    """
    x_axis = np.arange(0, len(variances) * step, step)
    plt.figure(figsize=(16,8))
    plt.plot(x_axis, variances)
    plt.axvline(x=window[0], color='g', linestyle='--', label='Event Start')
    plt.axvline(x=window[1], color='r', linestyle='--', label='Event End')
    plt.axhline(y=p_threshold, color="black", linestyle='--', label='Threshold')
    plt.show()
    pass

if __name__ == "__main__":
    file_path = "data\\dataframe_evening.pkl_annotated.pkl"
    new_df = complete_dataframe_annotation(file_path)