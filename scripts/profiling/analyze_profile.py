import pstats
from pstats import SortKey

try:
    # Load the profiling data
    stats = pstats.Stats('output.prof')

    # Remove path information
    stats.strip_dirs()

    print("--- Top 30 functions by CUMULATIVE time ---")
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)

    print("\n--- Top 30 functions by TOTAL (internal) time ---")
    stats.sort_stats(SortKey.TIME) # Or 'tottime'
    stats.print_stats(30)

except FileNotFoundError:
    print("Error: output.prof not found. Please ensure profile_runner.py has been run successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
