"""
Dataset constants and task lists.

Centralizes the 8 evaluation tasks used throughout the codebase.
"""

# Standard 8 evaluation tasks list
STANDARD_8_TASKS = [
    'EuroSAT',
    'Cars',
    'DTD',
    'SUN397',
    'RESISC45',
    'SVHN',
    'GTSRB',
    'MNIST'
]

# Task name variations (for compatibility with different naming conventions)
TASK_NAME_MAP = {
    'eurosat': 'EuroSAT',
    'cars': 'Cars',
    'dtd': 'DTD',
    'sun397': 'SUN397',
    'resisc45': 'RESISC45',
    'svhn': 'SVHN',
    'gtsrb': 'GTSRB',
    'mnist': 'MNIST',
    # Uppercase variants
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
    
    Args:
        task_name: Task name in any format
        
    Returns:
        Standardized task name
    """
    return TASK_NAME_MAP.get(task_name, task_name)


def get_standard_tasks():
    """
    Get list of standard 8 evaluation tasks.
    
    Returns:
        List of task names
    """
    return STANDARD_8_TASKS.copy()


def is_standard_task(task_name: str) -> bool:
    """
    Check if task is one of the standard 8 tasks.
    
    Args:
        task_name: Task name to check
        
    Returns:
        True if task is in standard list
    """
    normalized = normalize_task_name(task_name)
    return normalized in STANDARD_8_TASKS
