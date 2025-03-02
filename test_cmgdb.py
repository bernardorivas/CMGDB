import sys
import traceback

try:
    import CMGDB
    print("Successfully imported CMGDB")
    print(f"CMGDB version: {CMGDB.__version__ if hasattr(CMGDB, '__version__') else 'unknown'}")
    print(f"CMGDB path: {CMGDB.__file__}")
    
    # Try to use some basic functionality
    try:
        # Create a simple box map
        lower_bounds = [0.0, 0.0]
        upper_bounds = [1.0, 1.0]
        print("Creating a simple BoxMap...")
        box_map = CMGDB.BoxMap(lower_bounds, upper_bounds)
        print("BoxMap created successfully")
    except Exception as e:
        print(f"Error creating BoxMap: {e}")
        traceback.print_exc()
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc() 