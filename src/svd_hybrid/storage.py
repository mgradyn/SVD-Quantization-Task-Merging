"""
Artifact storage and loading utilities.

=== TUTORIAL: Saving and Loading Artifacts ===

SVD-Hybrid produces several artifacts that can be saved for later use:

1. **Bases**: SVD basis matrices (U_high, U_low) for each parameter
2. **Coefficients**: Compressed coefficients for each task
3. **Diagnostics**: Reconstruction errors and compression metrics
4. **Config**: The configuration used for the merge

Saving artifacts allows you to:
- Reconstruct the merged model later without original checkpoints
- Share compression results without full model files
- Analyze reconstruction quality and compression ratios

=== DIRECTORY STRUCTURE ===

After saving artifacts:
    artifact_dir/
    ├── basis/                    # SVD bases per parameter
    │   ├── layer1.weight.pt
    │   └── layer2.weight.pt
    ├── coeffs/                   # Compressed coefficients
    │   ├── layer1.weight.pt
    │   └── layer2.weight.pt
    ├── diagnostics.json          # Error metrics and statistics
    └── config.json               # Configuration used

=== EXAMPLE ===

    >>> from storage import save_all_artifacts, load_all_artifacts
    >>> 
    >>> # Save everything after merging
    >>> save_all_artifacts(bases, compressed, diagnostics, config, "./artifacts")
    >>> 
    >>> # Later, reload everything
    >>> artifacts = load_all_artifacts("./artifacts")
    >>> bases = artifacts["bases"]
    >>> compressed = artifacts["compressed"]
    >>> config = artifacts["config"]
"""

import torch
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


def save_basis(
    basis: Dict,
    param_name: str,
    output_dir: str
):
    """
    Save basis for a single parameter.
    
    Saves the SVD basis matrices and metadata for one parameter to a .pt file.
    
    Args:
        basis: Basis dictionary containing U_high, U_low, singular_values, etc.
        param_name: Parameter name (e.g., "layer1.weight")
        output_dir: Output directory (will create "basis/" subdirectory)
    """
    # Create basis subdirectory
    basis_dir = os.path.join(output_dir, "basis")
    os.makedirs(basis_dir, exist_ok=True)
    
    # Sanitize parameter name for filename (replace slashes)
    safe_name = param_name.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(basis_dir, f"{safe_name}.pt")
    
    # Prepare data to save
    data = {}
    
    # Save masked (signal) basis if present
    if basis.get("masked") is not None:
        masked = basis["masked"]
        data["masked"] = {
            "U_high": masked["U_high"].cpu(),
            "U_low": masked["U_low"].cpu(),
            "singular_values": masked["singular_values"].cpu(),
            "k": masked["k"],
            "mean": masked["mean"].cpu() if masked["mean"] is not None else None,
            "energy_retained": masked["energy_retained"],
            "D": masked["D"],
            "N": masked["N"]
        }
    
    # Save noise basis if present
    if basis.get("noise") is not None:
        noise = basis["noise"]
        data["noise"] = {
            "U_high": noise["U_high"].cpu(),
            "U_low": noise["U_low"].cpu(),
            "singular_values": noise["singular_values"].cpu(),
            "k": noise["k"],
            "mean": noise["mean"].cpu() if noise["mean"] is not None else None,
            "energy_retained": noise["energy_retained"],
            "D": noise["D"],
            "N": noise["N"]
        }
    
    torch.save(data, filepath)


def load_basis(
    param_name: str,
    artifact_dir: str,
    device: str = "cpu"
) -> Dict:
    """
    Load basis for a single parameter.
    
    Args:
        param_name: Parameter name
        artifact_dir: Artifact directory
        device: Device to load to
        
    Returns:
        Basis dictionary
    """
    basis_dir = os.path.join(artifact_dir, "basis")
    safe_name = param_name.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(basis_dir, f"{safe_name}.pt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Basis file not found: {filepath}")
    
    data = torch.load(filepath, map_location=device)
    
    return data


def save_compressed_coefficients(
    compressed: Dict[str, Dict[str, Dict]],
    output_dir: str
):
    """
    Save compressed coefficients for all parameters and tasks.
    
    Args:
        compressed: Dictionary mapping param_name -> task_name -> artifacts
        output_dir: Output directory
    """
    coeffs_dir = os.path.join(output_dir, "coeffs")
    os.makedirs(coeffs_dir, exist_ok=True)
    
    for param_name, task_artifacts in compressed.items():
        safe_name = param_name.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(coeffs_dir, f"{safe_name}.pt")
        
        # Move all tensors to CPU for storage
        cpu_artifacts = {}
        for task_name, artifact in task_artifacts.items():
            cpu_artifact = {}
            
            if artifact.get("masked") is not None:
                cpu_artifact["masked"] = {
                    "c_high_fp16": artifact["masked"]["c_high_fp16"].cpu(),
                    "c_low_quant": artifact["masked"]["c_low_quant"]
                }
            
            if artifact.get("unmasked") is not None:
                cpu_artifact["unmasked"] = {
                    "c_high_fp16": artifact["unmasked"]["c_high_fp16"].cpu(),
                    "c_low_quant": artifact["unmasked"]["c_low_quant"]
                }
            
            cpu_artifacts[task_name] = cpu_artifact
        
        torch.save(cpu_artifacts, filepath)


def load_compressed_coefficients(
    param_name: str,
    artifact_dir: str,
    device: str = "cpu"
) -> Dict[str, Dict]:
    """
    Load compressed coefficients for a single parameter.
    
    Args:
        param_name: Parameter name
        artifact_dir: Artifact directory
        device: Device to load to
        
    Returns:
        Dictionary mapping task_name -> artifacts
    """
    coeffs_dir = os.path.join(artifact_dir, "coeffs")
    safe_name = param_name.replace("/", "_").replace("\\", "_")
    filepath = os.path.join(coeffs_dir, f"{safe_name}.pt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Coefficients file not found: {filepath}")
    
    data = torch.load(filepath, map_location=device)
    
    return data


def save_diagnostics(
    diagnostics: Dict[str, Any],
    output_dir: str
):
    """
    Save diagnostics to JSON.
    
    Args:
        diagnostics: Diagnostics dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "diagnostics.json")
    
    # Convert any tensors or numpy arrays to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, torch.Size)):
            if isinstance(obj, torch.Size):
                return list(obj)
            return obj.cpu().tolist() if obj.numel() > 1 else obj.item()
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    serializable = convert_to_serializable(diagnostics)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_diagnostics(artifact_dir: str) -> Dict[str, Any]:
    """
    Load diagnostics from JSON.
    
    Args:
        artifact_dir: Artifact directory
        
    Returns:
        Diagnostics dictionary
    """
    filepath = os.path.join(artifact_dir, "diagnostics.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Diagnostics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_config(
    config,
    output_dir: str
):
    """
    Save configuration to JSON.
    
    Args:
        config: SVDHybridConfig object
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "config.json")
    
    # Convert dataclass to dict
    from dataclasses import asdict
    config_dict = asdict(config)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(artifact_dir: str):
    """
    Load configuration from JSON.
    
    Args:
        artifact_dir: Artifact directory
        
    Returns:
        SVDHybridConfig object
    """
    from .config import SVDHybridConfig
    
    filepath = os.path.join(artifact_dir, "config.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return SVDHybridConfig(**config_dict)


def save_all_artifacts(
    bases: Dict[str, Dict],
    compressed: Dict[str, Dict[str, Dict]],
    diagnostics: Dict[str, Any],
    config,
    output_dir: str
):
    """
    Save all artifacts to disk.
    
    Args:
        bases: All parameter bases
        compressed: All compressed coefficients
        diagnostics: Diagnostics data
        config: SVDHybridConfig
        output_dir: Output directory
    """
    print(f"Saving artifacts to {output_dir}")
    
    # Save bases
    for param_name, basis in bases.items():
        save_basis(basis, param_name, output_dir)
    
    # Save compressed coefficients
    save_compressed_coefficients(compressed, output_dir)
    
    # Save diagnostics
    save_diagnostics(diagnostics, output_dir)
    
    # Save config
    save_config(config, output_dir)
    
    print(f"Artifacts saved successfully")


def load_all_artifacts(
    artifact_dir: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load all artifacts from disk.
    
    Args:
        artifact_dir: Artifact directory
        device: Device to load to
        
    Returns:
        Dictionary with keys: bases, compressed, diagnostics, config
    """
    print(f"Loading artifacts from {artifact_dir}")
    
    # Load config
    config = load_config(artifact_dir)
    
    # Load diagnostics
    diagnostics = load_diagnostics(artifact_dir)
    
    # Get parameter names from diagnostics
    param_names = list(diagnostics.get("per_parameter", {}).keys())
    
    # Load bases
    bases = {}
    for param_name in param_names:
        try:
            bases[param_name] = load_basis(param_name, artifact_dir, device)
        except FileNotFoundError:
            print(f"Warning: Basis not found for {param_name}")
    
    # Load compressed coefficients
    compressed = {}
    for param_name in param_names:
        try:
            compressed[param_name] = load_compressed_coefficients(param_name, artifact_dir, device)
        except FileNotFoundError:
            print(f"Warning: Coefficients not found for {param_name}")
    
    print(f"Artifacts loaded successfully")
    
    return {
        "bases": bases,
        "compressed": compressed,
        "diagnostics": diagnostics,
        "config": config
    }


def save_merged_model(
    merged_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    filename: str = "merged_state_dict.pt"
):
    """
    Save merged model state dict.
    
    Args:
        merged_state_dict: Merged model state dict
        output_dir: Output directory
        filename: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    torch.save(merged_state_dict, filepath)
    print(f"Merged model saved to {filepath}")
