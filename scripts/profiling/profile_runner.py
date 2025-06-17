import asyncio
import cProfile
import os
import sys

# Add src to sys.path to allow importing from adaptive_graph_of_thoughts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from adaptive_graph_of_thoughts.config import (
    settings,  # Use the global settings instance
)
from adaptive_graph_of_thoughts.domain.services.got_processor import GoTProcessor


async def main():  # Renamed back to main as per instruction
    """
    Runs a profiling session by processing a sample query with GoTProcessor asynchronously.

    Initializes the GoTProcessor with global settings, submits a predefined query about climate change and ocean acidification, and prints a truncated summary of the final answer from the session data.
    """
    print("Initializing GoTProcessor...")
    processor = GoTProcessor(settings=settings)

    query = "What is the relationship between climate change and ocean acidification?"
    operational_params = {
        "include_reasoning_trace": False,
        "include_graph_state": False,
    }
    initial_context = {}
    session_id = "profile-session-001"

    print(f"Processing query: '{query}' with session ID: {session_id}")
    session_data = await processor.process_query(
        query=query,
        session_id=session_id,
        operational_params=operational_params,
        initial_context=initial_context,
    )
    print(
        f"Query processing complete. Final answer (summary): {session_data.final_answer[:200]}..."
    )


if __name__ == "__main__":
    profiler_output_file = "output.prof"

    print(f"Starting profiling, output will be saved to {profiler_output_file}")

    profiler = cProfile.Profile()
    profiler.enable()

    asyncio.run(main())  # main() is now the workload

    profiler.disable()
    print(f"Profiling complete. Saving stats to {profiler_output_file}")
    profiler.dump_stats(profiler_output_file)

    print("\nTo analyze the profile data, use pstats. Example:")
    print("import pstats; from pstats import SortKey")
    print(f"stats = pstats.Stats('{profiler_output_file}')")
    print("stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)")
