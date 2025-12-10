"""
Classification head utilities for loading task-specific classification heads.

This module provides functions to load pre-trained classification heads
for different datasets. Classification heads are typically stored as
PyTorch checkpoint files in a heads directory.
"""

import os
import torch


def get_classification_head(args, dataset_name):
    """
    Load a classification head for the specified dataset.
    
    The classification head is loaded from a file at:
    {args.save}/{dataset_name}Head.pt
    
    Args:
        args: Arguments object containing:
            - save: Directory where classification heads are stored
            - device: Device to load the head onto (default: 'cuda')
        dataset_name: Name of the dataset (e.g., 'Cars', 'DTD', 'EuroSAT')
        
    Returns:
        ClassificationHead: The loaded classification head
        
    Raises:
        FileNotFoundError: If the classification head file doesn't exist
    """
    # Get the save directory from args
    save_dir = getattr(args, 'save', getattr(args, 'checkpoint_dir', './checkpoints'))
    device = getattr(args, 'device', 'cuda')
    
    # Construct the path to the classification head
    head_path = os.path.join(save_dir, f'{dataset_name}Head.pt')
    
    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"Classification head not found at {head_path}. "
            f"Please ensure the head file exists for dataset '{dataset_name}'."
        )
    
    # Load the classification head
    print(f'Loading classification head from {head_path}')
    classification_head = torch.load(head_path, map_location=device, weights_only=False)
    
    # Move to device if needed
    if hasattr(classification_head, 'to'):
        classification_head = classification_head.to(device)
    
    return classification_head


def save_classification_head(head, args, dataset_name):
    """
    Save a classification head for the specified dataset.
    
    The classification head is saved to a file at:
    {args.save}/{dataset_name}Head.pt
    
    Args:
        head: The classification head to save
        args: Arguments object containing:
            - save: Directory where classification heads are stored
        dataset_name: Name of the dataset (e.g., 'Cars', 'DTD', 'EuroSAT')
    """
    # Get the save directory from args
    save_dir = getattr(args, 'save', getattr(args, 'checkpoint_dir', './checkpoints'))
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the path to save the classification head
    head_path = os.path.join(save_dir, f'{dataset_name}Head.pt')
    
    # Save the classification head
    print(f'Saving classification head to {head_path}')
    torch.save(head, head_path)
