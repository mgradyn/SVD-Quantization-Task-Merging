"""
Dataset constants and task lists.

Centralizes the 8 evaluation tasks used throughout the codebase.

=== TUTORIAL: Standard 8-Task Evaluation ===

This module defines the standard set of 8 benchmark tasks commonly used for
evaluating multi-task model merging methods. These tasks are diverse image
classification benchmarks.

=== THE 8 STANDARD TASKS ===

1. **EuroSAT**: Satellite imagery land use classification (10 classes)
2. **Cars**: Stanford Cars fine-grained car model classification (196 classes)
3. **DTD**: Describable Textures Dataset texture classification (47 classes)
4. **SUN397**: Scene Understanding (397 scene categories)
5. **RESISC45**: Remote Sensing Image Scene Classification (45 classes)
6. **SVHN**: Street View House Numbers digit recognition (10 classes)
7. **GTSRB**: German Traffic Sign Recognition Benchmark (43 classes)
8. **MNIST**: Handwritten digit recognition (10 classes)

=== WHY THESE TASKS? ===

- Diverse domains: satellite imagery, cars, textures, scenes, signs, digits
- Different difficulty levels and class counts
- Commonly used in task vector and model merging research
- Standardized for reproducibility across papers

=== USAGE ===

    >>> from dataset_constants import STANDARD_8_TASKS, normalize_task_name
    >>> 
    >>> # Get the standard task list
    >>> tasks = STANDARD_8_TASKS  # ['EuroSAT', 'Cars', 'DTD', ...]
    >>> 
    >>> # Normalize user input to standard format
    >>> normalize_task_name('eurosat')  # Returns 'EuroSAT'
    >>> normalize_task_name('CARS')     # Returns 'Cars'
"""

# Standard 8 evaluation tasks list
# These task names are used to:
# - Name checkpoint files (e.g., "Cars.pt")
# - Look up task-specific masks (e.g., "Cars_mask.pt")
# - Report per-task evaluation metrics
STANDARD_8_TASKS = [
    'EuroSAT',    # Satellite land use: urban, forest, river, etc.
    'Cars',       # Fine-grained car models
    'DTD',        # Textures: banded, bubbly, checkered, etc.
    'SUN397',     # Scenes: airport, bedroom, church, etc.
    'RESISC45',   # Remote sensing scenes: airport, beach, bridge, etc.
    'SVHN',       # Street view house numbers (digits 0-9)
    'GTSRB',      # German traffic signs: stop, yield, speed limit, etc.
    'MNIST'       # Handwritten digits (0-9)
]

# Task name variations mapping (for compatibility with different naming conventions)
# Users might specify tasks in various formats (lowercase, uppercase, etc.)
# This map normalizes them to the standard format
TASK_NAME_MAP = {
    # Lowercase variants
    'eurosat': 'EuroSAT',
    'cars': 'Cars',
    'dtd': 'DTD',
    'sun397': 'SUN397',
    'resisc45': 'RESISC45',
    'svhn': 'SVHN',
    'gtsrb': 'GTSRB',
    'mnist': 'MNIST',
    # Uppercase variants (already normalized forms)
    'EUROSAT': 'EuroSAT',
    'CARS': 'Cars',
    'DTD': 'DTD',
    'SUN397': 'SUN397',
    'RESISC45': 'RESISC45',
    'SVHN': 'SVHN',
    'GTSRB': 'GTSRB',
    'MNIST': 'MNIST'
}


def normalize_task_name(task_name: str) -> str:
    """
    Normalize task name to standard format.
    
    Converts various task name formats to the standard capitalization used
    throughout this codebase. If the task name is not recognized, returns
    the original input unchanged.
    
    Args:
        task_name: Task name in any format (e.g., 'eurosat', 'EUROSAT', 'EuroSAT')
        
    Returns:
        Standardized task name (e.g., 'EuroSAT')
        
    Example:
        >>> normalize_task_name('eurosat')
        'EuroSAT'
        >>> normalize_task_name('CARS')
        'Cars'
        >>> normalize_task_name('CustomTask')
        'CustomTask'  # Unknown names returned unchanged
    """
    return TASK_NAME_MAP.get(task_name, task_name)


def get_standard_tasks():
    """
    Get list of standard 8 evaluation tasks.
    
    Returns a copy of the task list to prevent accidental modification
    of the global list.
    
    Returns:
        List of 8 standard task names
        
    Example:
        >>> tasks = get_standard_tasks()
        >>> len(tasks)
        8
        >>> 'EuroSAT' in tasks
        True
    """
    return STANDARD_8_TASKS.copy()


def is_standard_task(task_name: str) -> bool:
    """
    Check if task is one of the standard 8 tasks.
    
    Normalizes the task name first, so 'eurosat', 'EUROSAT', and 'EuroSAT'
    all return True.
    
    Args:
        task_name: Task name to check (any format)
        
    Returns:
        True if task is in standard list, False otherwise
        
    Example:
        >>> is_standard_task('eurosat')
        True
        >>> is_standard_task('ImageNet')
        False
    """
    normalized = normalize_task_name(task_name)
    return normalized in STANDARD_8_TASKS
