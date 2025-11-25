"""
Main dispatcher for different merging methods.

=== TUTORIAL: Using the Dispatcher ===

This module provides a unified entry point for different model merging methods.
Currently supports SVD-Hybrid, with the architecture ready for additional methods.

=== BASIC USAGE ===

    python src/main.py --method svd_hybrid [options]

=== AVAILABLE METHODS ===

- **svd_hybrid**: SVD-based compression with Tall Masks and RTVQ quantization
- **task_arithmetic**: Simple task vector averaging (not yet implemented)
- **ties**: TIES merging method (not yet implemented)
- **dare**: DARE merging method (not yet implemented)

=== HOW IT WORKS ===

The dispatcher:
1. Parses the --method argument
2. Delegates to the appropriate method's CLI
3. Passes remaining arguments to the method

=== EXAMPLE ===

    # Run SVD-Hybrid merging
    python src/main.py --method svd_hybrid \\
        --tasks Cars DTD EuroSAT SUN397 \\
        --checkpoint-dir ./checkpoints \\
        --base-model-path ./base_model.pt \\
        --output-dir ./output
"""

import argparse
import sys


def main():
    """
    Main entry point for all merging methods.
    
    Parses the --method argument to determine which merging method to use,
    then delegates to that method's CLI with the remaining arguments.
    
    Returns:
        Results from the selected merging method, or None if method not found
    """
    parser = argparse.ArgumentParser(
        description="Task merging methods for multi-task models"
    )
    
    parser.add_argument("--method", type=str, default="svd_hybrid",
                       choices=["svd_hybrid", "task_arithmetic", "ties", "dare"],
                       help="Merging method to use (default: svd_hybrid)")
    
    # Parse known args to get method, then delegate to method-specific CLI
    # remaining contains all arguments not consumed by this parser
    args, remaining = parser.parse_known_args()
    
    if args.method == "svd_hybrid":
        # Import and run SVD-Hybrid
        from svd_hybrid.run import main as svd_main
        
        # Reconstruct sys.argv for the method's parser
        # This allows the method to parse its own arguments
        sys.argv = [sys.argv[0]] + remaining
        
        return svd_main(args)
    
    elif args.method == "task_arithmetic":
        print(f"Method '{args.method}' not yet implemented")
        print("Use svd_hybrid for now, or implement task_arithmetic in a new module")
        return None
    
    elif args.method == "ties":
        print(f"Method '{args.method}' not yet implemented")
        return None
    
    elif args.method == "dare":
        print(f"Method '{args.method}' not yet implemented")
        return None
    
    else:
        print(f"Unknown method: {args.method}")
        parser.print_help()
        return None


if __name__ == "__main__":
    main()
