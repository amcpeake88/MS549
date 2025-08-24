# Ride-Sharing Simulator — Final Project

A discrete-event ride-sharing simulation using **Dijkstra’s shortest path**, **Quadtree driver matching**, and a **dynamic event engine**.  
Produces analytics and a visual summary of results.

---

## Purpose & Design

This project simulates ride requests in a city using:

- **Event Engine** – heapq-based priority queue for requests, pickups, and dropoffs  
- **Quadtree** – efficient nearest-driver lookup  
- **Dijkstra** – shortest-path routing between nodes  
- **Analytics** – trip counts, wait times, utilization  
- **Visualization** – generates an analytical summary PNG (`simulation_summary.png`) with maps and charts  

---

## Repository Layout

- `ms549 final/` — contains all Python source files:
  - `car.py` – Car class + Dijkstra integration  
  - `rider.py` – Rider class  
  - `graph_basic.py` – Graph loader (supports 3-col & 7-col CSVs)  
  - `enhanced_quadtree.py` – k-nearest search  
  - `final_simulation.py` – Main simulation `run()`  
  - `visualization_engine.py` – Creates PNG summary  
  - `test_complete.py` – Full console demo  
  - `test_png.py` – Quick PNG test  
- `map.csv` – Sample map data file  

---

## Command-Line Arguments

- `--max-time INT` — total simulated time in minutes (e.g., `100`)  
- `--num-cars INT` — number of cars to deploy (e.g., `5`)  
- `--arrival-rate FLOAT` — average rider arrival rate per minute (Poisson/exponential)  
- `--map-file PATH` — CSV file path for the road graph (e.g., `map.csv`)  
- `--png PATH` — save an analytical summary image to this path (e.g., `simulation_summary.png`)  
- `--realtime` — run the event loop in real-time (ideal for live demos)  
- `--surge` — enable surge-based rider generation or dispatch logic  

### Examples

**Full simulation with PNG output**
```bash
python final_simulation.py --max-time 100 --num-cars 5 --arrival-rate 2.0 --map-file map.csv --png simulation_summary.png
