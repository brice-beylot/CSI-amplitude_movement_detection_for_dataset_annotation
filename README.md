# Movement Detection: Wi-Fi CSI-Based Movement Detection
---

## **📌 Description**
An advanced Python package for detecting human movement—specifically falls—using Wi-Fi Channel State Information (CSI). 
This pipeline processes raw CSI data, applies signal filtering (Low-pass, Hampel), selects the optimal subcarrier bands using Narrow Band Variance Index (NBVI) scoring, and uses dynamic thresholding to automatically annotate start and end times for movement events.

**Key Features**:
✅ **Preprocessing**: Gain compensation, denoising, bandpass filtering, and null subcarrier removal.
✅ **Calibration**: Automatically selects optimal subcarrier bands and thresholds using idle-state CSI data.
✅ **Detection**: Sliding-window analysis with configurable sensitivity for real-time or annotated datasets.
✅ **Weighted Selection**: Gaussian-weighted window selection for improved centrality of detected events.
✅ **Visualization**: Built-in plotting for CSI data and variance metrics.

---

## **📥 Installation**

### **Prerequisites**
- Python **3.8 or higher**
- `pip` (Python package manager)

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/movement_detection.git
cd movement_detection
```

### **2. Create a Virtual Environment (Recommended)**
```bash
python -m venv .venv          # Create a virtual environment
source .venv/bin/activate     # Activate on Linux/Mac
.venv\Scripts\activate        # Activate on Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt  # Install all dependencies
pip install -e .                # Install the package in editable mode
```

Note: The -e . flag installs the package in "editable" mode, so changes to the code are reflected immediately without reinstalling.


## **🚀 Quick Start**

### **Basic Usage**

```python
from movement_detection import load_dataset, complete_dataframe_annotation

# Annotate a dataset with detected movement events
annotated_df = complete_dataframe_annotation(
    file_path="data/your_dataset.pkl",
    calibration=True,          # Use calibration mode (requires idle states)
    method="turbulence",       # or "amplitude"
    weighted=True,            # Use Gaussian-weighted window selection
    sigma=12,                 # Spread parameter for Gaussian weighting
)
```

### **Example: Full Pipeline**

```python
from movement_detection import (
    load_dataset,
    change_CSI_processed,
    create_filtered_database,
    plot_csi_processed_sample,
)

# 1. Load the dataset
df = load_dataset("data/your_dataset.pkl")

# 2. Preprocess CSI data
df = change_CSI_processed(
    df,
    remove_null=True,      # Remove null subcarriers
    apply_gain=True,       # Apply gain compensation
    denoise=True,          # Apply denoising (hampel_filter by default)
    bandpass=True,         # Apply bandpass filter
    low_freq=0.5,          # Low cutoff frequency (Hz)
    high_freq=10.0,       # High cutoff frequency (Hz)
)

# 3. Detect movement events and filter the database
filtered_df = create_filtered_database(
    df,
    calibration=True,      # Use calibration mode
    all_subcarriers=True,  # Use all subcarriers (skip band optimization)
    method="turbulence",    # Detection method: "turbulence" or "amplitude"
    weighted=True,         # Use Gaussian-weighted window selection
    sigma=12,              # Spread parameter for Gaussian weighting
    factor=1.1,           # Threshold multiplier (higher = fewer detections)
)

# 4. Visualize a sample
plot_csi_processed_sample(filtered_df, index=0)
```

## **🔧 Configuration Parameters**
The package supports many tunable parameters for preprocessing, calibration, and detection. Below are the most important ones:
| Category | Parameter | Default Value | Description |
| --- | --- | --- | --- |
| Preprocessing | remove_null | True | Remove null subcarriers from the CSI data. |
|  | apply_gain | True | Apply AGC/FFT gain compensation. |
|  | denoise | True | Apply denoising (e.g., Hampel or median filter). |
|  | denoise_method | "hampel_filter" | Denoising method ("hampel_filter" or "median_filter"). |
|  | bandpass | True | Apply a bandpass filter to the CSI data. |
|  | low_freq | 0.5 | Low cutoff frequency for the bandpass filter (Hz). |
|  | high_freq | 10.0 | High cutoff frequency for the bandpass filter (Hz). |
|  | standardize | False | Standardize the CSI data (subtract mean, divide by std). |
| Detection | method | "amplitude" | Detection method ("turbulence" or "amplitude"). |
|  | calibration | False | Use calibration mode (requires event_start and event_end columns). |
|  | all_subcarriers | True | Skip band optimization and use all subcarriers. |
|  | gain_locked | True | Use standard deviation (if True) or coefficient of variation (if False) for turbulence. |
|  | apply_noise_gate_band | False | Apply noise gating to exclude weak/dead subcarriers. |
| Windowing | window_calibration_size | 300 | Window size for calibration (in packets). |
|  | window_detection_size | 300 | Window size for detection (in packets). |
|  | mvs_window_size | 100 | Moving window size for validation metrics (in packets). |
|  | step_calibration | 25 | Step size for calibration windows (in packets). |
|  | step_detection | 10 | Step size for detection windows (in packets). |
| Thresholding | factor | 1.1 | Multiplier for calibration threshold (higher = fewer detections). |
|  | percentile_calibration | 5 | Percentile for calibration threshold calculation. |
|  | percentile_detection | 35 | Percentile for detection threshold (if calibration=False). |
|  | validation_factor | 1.1 | Multiplier for validation threshold. |
| Weighted Selection | weighted | True | Use Gaussian weighting to select centralized windows. |
|  | sigma | 12 | Spread parameter for the Gaussian weighting function. |


## **API Reference**

### **Running function**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| complete_dataframe_annotation | file_path | Main wrapper pipeline that runs dataset loading, parameter filtering, preprocessing, event detection, and event window database creation. |

### **Core Functions**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| load_dataset | file_path | Load the dataset from a .pkl file and return a DataFrame. |
| calculate_nbvi_scores | subcarrier_amplitudes | Calculate three complementary Narrow Band Variance Index (NBVI) scores for a single subcarrier. |
| generate_four_candidate_bands | csi_window | Generate four candidate subcarrier bands using different NBVI scoring and spacing strategies. |
| validate_band | csi_data, band | Validate a subcarrier band by calculating its false positive rate using a baseline moving variance or mean metric. |
| select_best_band | csi_data, candidate_bands_dict | Evaluates candidate subcarrier bands and selects the optimal one based on empirical false positive rates. |
| complete_nbvi_calibration | csi_data | Executes the complete calibration workflow: finds baseline windows, generates band candidates, and selects the optimal band. |
| find_candidate_windows | csi_packets | Partitions the recording into overlapping windows and identifies the quietest or noisiest windows based on variance metrics and a calculated/given threshold. |
| create_filtered_database | df | Creates a new DataFrame retaining only valid rows and create/overwrites 'event_start' and 'event_end' columns with algorithmically detected event boundaries. |

### **Utility Functions**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| calculate_spatial_turbulence | amplitudes, band | Calculate spatial turbulence (standard deviation or coefficient of variation) from subcarrier amplitudes for a single packet. |
| calculate_moving_variance | csi_data, band | Calculate the moving variance of spatial turbulence values over a sliding window. |
| calculate_moving_mean | csi_data, band | Calculate the moving mean of the variance of amplitude values for the specified subcarriers. |
| calculate_percentile | values, percentile | Calculate percentile value from a list using linear interpolation. |
| calculate_variance | values | Calculate variance using a numerically stable two-pass algorithm. |
| calculate_mean | values | Calculate the arithmetic mean of a list of values. |
| apply_noise_gate | subcarrier_metrics | Apply a noise gate to exclude weak/dead subcarriers and those with infinite NBVI scores. |
| select_with_spacing_strict | sorted_metrics | Select subcarriers while strictly maximizing spectral diversity by enforcing spacing constraints. |
| select_with_spacing_clustered | sorted_metrics | Select subcarriers using a clustered strategy (best 5 unrestricted, remaining spaced). |
| gaussian | distance | Compute a Gaussian weight for a given distance to prioritize centralized windows. |

### **Data Preprocessing**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| apply_lowpass_filter_row | df | Apply a 2nd-order low-pass Butterworth filter to the CSI data contained in the "CSI_processed" column of the DataFrame. |
| change_CSI_processed | df | Applies preprocessing steps to the CSI signals present in the "CSI" column of the dataset and regenerates the 'CSI_processed' column. |
| selected_band_csi | n_packets, band | Extracts only the specified subcarriers in the band from a given CSI signal. |
| reshaping | database, band | Iterates over an entire dataset of CSI samples and filters each one to only contain the chosen subcarrier band. |
| join_idle_states | df, index | Concatenates the CSI data before an event starts and after an event ends to create a continuous idle state baseline. |

### **Visualization**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| plot_csi_processed_sample | df, index | Plots the processed CSI signal for a given DataFrame row alongside overlayed event boundaries. |
| plot_variance | window, variances, p_threshold, step | Renders a line plot comparing continuous variance output streams relative to an evaluated target threshold limit line. |

### **Test function**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| mean_time_differences | df | Evaluates the algorithm's detection accuracy by computing the mean time difference between annotated window centers and detected window centers for each CSI sample. |

### **Saving function**
| Function | Non-Optional Parameters | Description |
| --- | --- | --- |
| change_file_name | file_path | Generates a descriptive filename string dynamically based on the current pipeline configuration variables. |

## **📁 Project Structure**
```text
movement_detection/
├── pyproject.toml          # Build system and project metadata
├── requirements.txt        # Dependencies for development
├── README.md               # Project documentation
├── LICENSE                 # License file (MIT)
├── src/
│   └── movement_detection/
│       ├── __init__.py     # Package initialization
│       └── movement_detection.py  # Main module
└── data/                  # Example datasets (not committed to Git)
```
## **Tested results**
The following results were obtained using certain fixed parameters, as set out below : 
 - remove_null = True
 - apply_gain = True
 - denoise = True
 - standardize = False,
 - denoise_method: str = "hampel_filter",
 - low_freq: float = 0.5,
 - high_freq: float = 10.0,
 - window_calibration_size: int = 300,
 - window_detection_size: int = 300,
 - mvs_window_size: int = 100,
 - step_calibration: int = 25,
 - step_detection: int = 10,
 - percentile_calibration: int = 5,

| dataset | bandpass | gain_locked | all_subcarriers | apply_noise_gate_band | method | calibration | weighted | factor | percentile_detection | sigma | difference in packets | samples without movement detected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 8 | 40.06 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 10 | 39.02 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 12 | 38.95 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 15 | 38.87 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 20 | 38.72 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.05 | _ | 25 | 38.72 | 18 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | True | True | 1.1 | _ | 20 | 36.43 | 26 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 10 | 20 | 54.41 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 20 | 20 | 46.95 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 30 | 20 | 40.75 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 40 | 20 | 37.22 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 50 | 20 | 34.74 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 60 | 20 | 32.17 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 70 | 20 | 31.34 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 80 | 20 | 29.95 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 90 | 20 | 29.63 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 100 | 20 | 30.10 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 90 | 15 | 29.24 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 90 | 10 | 33.31 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | turbulence | False | True | _ | 90 | 12 | 31.34 | 0 |
| 26_05_21.pkl_annotated | False | True | False | True | amplitude | True | True | 1.05 | _ | 10 | 36.70 | 17 |
| 26_05_21.pkl_annotated | False | True | False | True | amplitude | True | True | 1.05 | _ | 15 | 35.22 | 17 |
| 26_05_21.pkl_annotated | False | True | False | True | amplitude | True | True | 1.05 | _ | 20 | 34.58 | 17 |