"""
test_png.py
-----------
Minimal test runner that only generates PNG visualizations.

This script runs a short simulation on a simple graph
and outputs a visualization PNG using the integrated
visualization code in CompleteSimulation.
"""

from final_simulation import CompleteSimulation
from graph_basic import Graph
import os, webbrowser


def build_test_graph():
    """Create a small 3x3 grid graph with unit edge weights."""
    graph = Graph()
    for x in range(3):
        for y in range(3):
            if x < 2:
                graph.add_edge((x, y), (x + 1, y), 1)
            if y < 2:
                graph.add_edge((x, y), (x, y + 1), 1)
    return graph


def main():
    print("=== PNG TEST START ===")

    # Build graph
    graph = build_test_graph()

    # Run a short simulation
    sim = CompleteSimulation(graph, num_cars=3, duration=10)
    sim.run()

    # Generate visualization PNG
    png_file = sim.generate_visualizations("test_visual.png")

    if os.path.exists(png_file):
        print(f"PNG generated: {png_file}")
        try:
            webbrowser.open(os.path.abspath(png_file))
        except Exception as e:
            print(f"Could not auto-open PNG: {e}")
    else:
        print("ERROR: PNG file not created.")

    print("=== PNG TEST END ===")


if __name__ == "__main__":
    main()
