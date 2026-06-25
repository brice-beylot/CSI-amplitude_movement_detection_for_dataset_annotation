"""
Movement Detection Package

This package provides tools for Wi-Fi CSI-based movement detection,
including data loading, preprocessing, filtering, and event detection.
"""

# Import ALL functions from movement_detection.py
from .movement_detection import (
    # Running function
    complete_dataframe_annotation,

    # Core functions
    load_dataset,
    calculate_nbvi_scores,
    generate_four_candidate_bands,
    validate_band,
    select_best_band,
    complete_nbvi_calibration,
    find_candidate_windows,
    create_filtered_database,

    # Utility functions
    calculate_spatial_turbulence,
    calculate_moving_variance,
    calculate_moving_mean,
    calculate_percentile,
    calculate_variance,
    calculate_mean,
    apply_noise_gate,
    select_with_spacing_strict,
    select_with_spacing_clustered,
    gaussian,

    # Data processing
    apply_lowpass_filter_row,
    change_CSI_processed,
    selected_band_csi,
    reshaping,
    join_idle_states,

    # Visualization
    plot_csi_processed_sample,
    plot_variance,

    # Test function
    mean_time_differences,
    
    # Saving function
    change_file_name
)

# Define what is exported when users run `from movement_detection import *`
__all__ = [
    # Running function
    "complete_dataframe_annotation",

    # Core functions
    "load_dataset",
    "calculate_nbvi_scores",
    "generate_four_candidate_bands",
    "validate_band",
    "select_best_band",
    "complete_nbvi_calibration",
    "find_candidate_windows",
    "create_filtered_database",

    # Utility functions
    "calculate_spatial_turbulence",
    "calculate_moving_variance",
    "calculate_moving_mean",
    "calculate_percentile",
    "calculate_variance",
    "calculate_mean",
    "apply_noise_gate",
    "select_with_spacing_strict",
    "select_with_spacing_clustered",
    "gaussian",

    # Data processing
    "apply_lowpass_filter_row",
    "change_CSI_processed",
    "selected_band_csi",
    "reshaping",
    "join_idle_states",

    # Visualization
    "plot_csi_processed_sample",
    "plot_variance",

    # Test function
    "mean_time_differences",
    
    # Saving function
    "change_file_name"
]