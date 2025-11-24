"""
Main dispatcher for different merging methods.
"""
import argparse
import sys


def main():
    """Main entry point for all merging methods."""
    parser = argparse.ArgumentParser(
        description="Task merging methods for multi-task models"
    )
    
    parser.add_argument("--method", type=str, default="svd_hybrid",
                       choices=["svd_hybrid", "task_arithmetic", "ties", "dare"],
                       help="Merging method to use")
    
    # Parse known args to get method, then delegate to method-specific CLI
    args, remaining = parser.parse_known_args()
    
    if args.method == "svd_hybrid":
        # Import and run SVD-Hybrid
        from svd_hybrid.run import main as svd_main
        
        # Reconstruct sys.argv for the method's parser
        sys.argv = [sys.argv[0]] + remaining
        
        return svd_main(args)
    
    elif args.method == "task_arithmetic":
        print(f"Method '{args.method}' not yet implemented")
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
