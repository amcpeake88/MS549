# Ride-Sharing Simulator

A discrete-event ride-sharing simulation using Dijkstra's shortest path, Quadtree driver matching, and a dynamic event engine. Produces analytics and a visual summary of results.

## Overview

This project simulates ride requests in a city using advanced algorithms and data structures to model real-world ride-sharing dynamics. The simulation tracks cars, riders, and events in real-time, providing comprehensive analytics on system performance.

### Key Technologies

- **Event Engine** – heapq-based priority queue for managing requests, pickups, and dropoffs
- **Quadtree** – efficient nearest-driver lookup for optimal matching
- **Dijkstra's Algorithm** – shortest-path routing between nodes
- **Analytics Engine** – comprehensive metrics on trip counts, wait times, and utilization
- **Visualization** – generates analytical summary PNG with maps and performance charts

## Repository Structure

```
ms549 final/
├── car.py                    # Car class with Dijkstra integration
├── rider.py                  # Rider class
├── graph_basic.py           # Graph loader (supports 3-col & 7-col CSVs)
├── enhanced_quadtree.py     # k-nearest search implementation
├── final_simulation.py      # Main simulation engine
├── visualization_engine.py  # PNG summary generator
├── test_complete.py         # Full console demonstration
├── test_png.py             # Quick PNG generation test
└── map.csv                 # Sample map data file
```

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--max-time` | INT | Total simulated time in minutes (e.g., 100) |
| `--num-cars` | INT | Number of cars to deploy (e.g., 5) |
| `--arrival-rate` | FLOAT | Average rider arrival rate per minute (Poisson/exponential) |
| `--map-file` | PATH | CSV file path for the road graph (e.g., map.csv) |
| `--png` | PATH | Save analytical summary image to this path |
| `--realtime` | FLAG | Run event loop in real-time (ideal for live demos) |
| `--surge` | FLAG | Enable surge-based rider generation or dispatch logic |

## Usage Examples

### Full Simulation with PNG Output
```bash
python final_simulation.py --max-time 100 --num-cars 5 --arrival-rate 2.0 --map-file map.csv --png simulation_summary.png
```

### Console Demo with Real-time Pacing
```bash
python test_complete.py --max-time 100 --num-cars 5 --arrival-rate 2.0 --map-file map.csv --realtime --surge
```

### Quick PNG Test
```bash
python test_png.py --png simulation_summary.png
```

## Analytical Visualization

The simulation generates `simulation_summary.png` - a comprehensive performance dashboard that analyzes ride-sharing system efficiency. This visualization includes:

### 1. Trip Map
Shows routes taken by cars from pickups to drop-offs, illustrating geographic coverage and dispatch efficiency across the simulated city.

### 2. Per-Car Trip Counts
Bar chart displaying trip completion by each vehicle, highlighting workload distribution and identifying potential fleet imbalances.

### 3. Wait-Time Histogram
Distribution of rider wait times, measuring service quality and system responsiveness to ride requests.

### 4. Performance Metrics Header
Summary statistics including:
- Total riders and completed trips
- Average and median rider wait time
- Average trip duration
- Overall driver utilization percentage

### Performance Analysis

The visualization enables analysis of key trade-offs:

**Service Quality vs. Operational Efficiency**
- **Low wait times** → High rider satisfaction, potentially low driver utilization
- **High utilization** → Better efficiency, potentially longer wait times

By examining the PNG output, you can evaluate how system parameters (fleet size, arrival rate, surge mode) impact both rider experience and operational efficiency.

## Dependencies

- **Python 3.9+**
- **matplotlib**

### Installation
```bash
pip install matplotlib
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install matplotlib`
3. Run a basic simulation: `python final_simulation.py --max-time 50 --num-cars 3 --arrival-rate 1.5 --map-file map.csv --png results.png`
4. View the generated `results.png` for performance analysis

## Technical Details

The simulation uses discrete-event modeling to accurately represent the temporal dynamics of ride-sharing operations. The Quadtree data structure enables efficient spatial queries for driver-rider matching, while Dijkstra's algorithm ensures optimal routing between city nodes.
