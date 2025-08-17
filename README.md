Ride-Sharing Simulator — Final Project

A discrete-event ride-sharing simulation using Dijkstra’s shortest path, Quadtree driver matching, and a dynamic event engine. Produces analytics and visualization of results.

Purpose & Design
This project simulates ride requests in a city using:
Event Engine – heapq-based priority queue for requests, pickups, and dropoffs.
Quadtree – efficient nearest-driver lookup.
Dijkstra – shortest-path routing between nodes.
Analytics – trip counts, wait times, utilization.
Visualization – generates simulation_summary.png with maps and charts.

Repository Layout
car.py – Car class + Dijkstra integration
rider.py – Rider class
graph_basic.py – Graph loader (3-col & 7-col CSV support)
enhanced_quadtree.py – k-nearest search
final_simulation.py – Main simulation run()
visualization_engine.py – Creates PNG summary
map.csv – Sample map data
test_complete.py – Full console demo
test_png.py – Quick PNG test

How to Run

Full Simulation + PNG
python final_simulation.py --max-time 100 --num-cars 5 --arrival-rate 2.0 --map-file map.csv

Console Demo
python test_complete.py --max-time 100 --num-cars 5

PNG Quick Test
python test_png.py

Key Features
Dynamic Rider Generation – exponential inter-arrival for realistic requests.
Car Dispatch – Quadtree finds nearest k cars, Dijkstra selects fastest route.
Arrival Handling – updates car location at pickup and dropoff, re-inserts into Quadtree.
Visualization – trip map, per-car trip counts, and wait time histogram.

Dependencies
Python 3.9+
matplotlib

Install:
pip install matplotlib

Demo Checklist

Show event engine loop with heapq.

Explain rider generation and car assignment.

Highlight Quadtree and Dijkstra integration.

Show pickup/dropoff handlers updating car location.

Run simulation and display simulation_summary.png.
