#!/usr/bin/env python3
"""
test_complete.py — Self-contained, production-ready simulation that satisfies 2a–2d.

Features
--------
• Clean OOP design: Graph, Car, Rider, EnhancedQuadtree, Simulation (2a: Class Structure)
• Weighted adjacency-list Graph + Dijkstra pathfinding (2a: Graph, Pathfinding)
• Event-driven engine w/ min-heap, event types, dynamic rider generation (2a: Event Engine)
• Quadtree spatial index + k-nearest matching (2b: Efficient Driver Matching)
• Instrumentation & analytics (2c: wait time, trip duration, utilization)
• Visualization:
    - PNG summary report saved to disk
    - Optional real-time Matplotlib animation (--realtime) (2c: Visualization)
• Extra credit: Surge pricing by zones (enable with --surge) (2d)

Usage
-----
python test_complete.py --max-time 100 --num-cars 25 --arrival-rate 2.0 --map-file map.csv --png out.png --realtime --surge
"""

from __future__ import annotations
import argparse, csv, heapq, math, os, random, sys, time, webbrowser
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_MEAN_ARRIVAL = 2.0
K_NEAREST = 5
SURGE_BETA = 0.5       # sensitivity of surge multiplier
SURGE_GRID = 3         # divide map into SURGE_GRID x SURGE_GRID zones

# -----------------------------
# Graph (weighted adjacency list) + Dijkstra (2a)
# -----------------------------
class Graph:
    def __init__(self) -> None:
        self.adj: Dict[str, List[Tuple[str, float]]] = {}
        self.node_coordinates: Dict[str, Tuple[float, float]] = {}

    def add_edge(self, u: str, v: str, w: float) -> None:
        self.adj.setdefault(u, []).append((v, float(w)))

    def get_neighbors(self, u: str) -> List[Tuple[str, float]]:
        return self.adj.get(u, [])

    def get_all_nodes(self) -> List[str]:
        return list(self.adj.keys())

    def load_map(self, path: str) -> None:
        """
        CSV columns (minimum): start_node,end_node,travel_time
        Optional: start_x,start_y,end_x,end_y
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing map file: {path}")
        with open(path, newline='') as f:
            rdr = csv.DictReader(f)
            has_xy = {'start_x','start_y','end_x','end_y'}.issubset(set(rdr.fieldnames or []))
            for row in rdr:
                u, v, w = row['start_node'].strip(), row['end_node'].strip(), float(row['travel_time'])
                self.add_edge(u, v, w)
                if has_xy:
                    sx, sy = float(row['start_x']), float(row['start_y'])
                    ex, ey = float(row['end_x']), float(row['end_y'])
                    self.node_coordinates.setdefault(u, (sx, sy))
                    self.node_coordinates.setdefault(v, (ex, ey))
        if not self.node_coordinates:
            # fallback: generate a compact grid for the nodes that appear as starts
            nodes = sorted(set(self.get_all_nodes()))
            # Distribute in a 3x3 repeating grid
            grid = [(1,1),(3,1),(5,1),(1,3),(3,3),(5,3),(1,5),(3,5),(5,5)]
            for i, n in enumerate(nodes):
                self.node_coordinates[n] = grid[i % len(grid)]

    def find_nearest_vertex(self, xy: Tuple[float,float]) -> str:
        x,y = xy
        best = None
        bestd = float('inf')
        for node, (nx,ny) in self.node_coordinates.items():
            d = (nx-x)*(nx-x) + (ny-y)*(ny-y)
            if d < bestd:
                bestd, best = d, node
        return best or next(iter(self.node_coordinates))


def dijkstra(graph: Graph, start: str, goal: str, weight_scale: float=1.0) -> Tuple[float, List[str]]:
    """Return (distance, path) from start to goal using Dijkstra. Scale all weights by weight_scale."""
    if start not in graph.adj or goal not in graph.adj:
        # allow goal with no outgoing edges
        if start not in graph.adj or (goal not in graph.adj and goal not in graph.node_coordinates):
            return float('inf'), []
    dist: Dict[str, float] = defaultdict(lambda: float('inf'))
    prev: Dict[str, Optional[str]] = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = set()
    while pq:
        d,u = heapq.heappop(pq)
        if u in visited: 
            continue
        visited.add(u)
        if u == goal:
            break
        for v,w in graph.get_neighbors(u):
            nd = d + w*weight_scale
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[goal] == float('inf'):
        return float('inf'), []
    # reconstruct
    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return dist[goal], path

# -----------------------------
# Quadtree (2b)
# -----------------------------
@dataclass
class Rectangle:
    x: float
    y: float
    w: float
    h: float
    def contains(self, pt: Tuple[float,float]) -> bool:
        px,py = pt
        return (self.x <= px < self.x+self.w) and (self.y <= py < self.y+self.h)
    def min_dist2(self, pt: Tuple[float,float]) -> float:
        px,py = pt
        dx = 0.0
        if px < self.x: dx = self.x - px
        elif px > self.x+self.w: dx = px - (self.x+self.w)
        dy = 0.0
        if py < self.y: dy = self.y - py
        elif py > self.y+self.h: dy = py - (self.y+self.h)
        return dx*dx + dy*dy

class QuadtreeNode:
    __slots__ = ("boundary","points","cars","divided","nw","ne","sw","se","capacity")
    def __init__(self, boundary: Rectangle, capacity: int = 6):
        self.boundary = boundary
        self.points: List[Tuple[float,float]] = []
        self.cars: List["Car"] = []
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None
        self.capacity = capacity

    def subdivide(self):
        if self.divided: return
        x,y,w,h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
        hw, hh = w/2.0, h/2.0
        self.nw = QuadtreeNode(Rectangle(x, y, hw, hh), self.capacity)
        self.ne = QuadtreeNode(Rectangle(x+hw, y, hw, hh), self.capacity)
        self.sw = QuadtreeNode(Rectangle(x, y+hh, hw, hh), self.capacity)
        self.se = QuadtreeNode(Rectangle(x+hw, y+hh, hw, hh), self.capacity)
        self.divided = True

    def insert(self, car: "Car") -> bool:
        if not self.boundary.contains(car.coordinates):
            return False
        if len(self.points) < self.capacity:
            self.points.append(car.coordinates)
            self.cars.append(car)
            return True
        if not self.divided:
            self.subdivide()
            # reinsert existing
            old_points, old_cars = self.points, self.cars
            self.points, self.cars = [], []
            for p, c in zip(old_points, old_cars):
                (self.nw.insert(c) or self.ne.insert(c) or self.sw.insert(c) or self.se.insert(c))
        return (self.nw.insert(car) or self.ne.insert(car) or self.sw.insert(car) or self.se.insert(car))

    def remove(self, car: "Car") -> bool:
        # attempt removal in this node
        for i,c in enumerate(self.cars):
            if c is car:
                del self.cars[i]
                del self.points[i]
                return True
        if self.divided:
            return self.nw.remove(car) or self.ne.remove(car) or self.sw.remove(car) or self.se.remove(car)
        return False

    def k_nearest(self, pt: Tuple[float,float], k: int, out: List[Tuple[float,"Car"]]):
        # prune if current best k has better distance than this node bbox
        if len(out) >= k:
            out.sort(key=lambda x:x[0])
            worst = out[-1][0]
            if self.boundary.min_dist2(pt) > worst:
                return
        # test local points
        for c in self.cars:
            dx = c.coordinates[0]-pt[0]; dy = c.coordinates[1]-pt[1]
            d2 = dx*dx + dy*dy
            out.append((d2, c))
        if self.divided:
            # search child likely containing pt first for speed
            children = [self.nw, self.ne, self.sw, self.se]
            children.sort(key=lambda child: child.boundary.min_dist2(pt))
            for child in children:
                child.k_nearest(pt, k, out)

class EnhancedQuadtree:
    def __init__(self, boundary: Rectangle):
        self.root = QuadtreeNode(boundary)
    def insert_car(self, car: "Car") -> bool:
        return self.root.insert(car)
    def remove_car(self, car: "Car") -> bool:
        return self.root.remove(car)
    def update_car(self, car: "Car", new_xy: Tuple[float,float]) -> bool:
        self.remove_car(car)
        car.coordinates = new_xy
        return self.insert_car(car)
    def find_k_nearest_cars(self, pt: Tuple[float,float], k: int = K_NEAREST, status_filter: Optional[str]=None) -> List["Car"]:
        out: List[Tuple[float,"Car"]] = []
        self.root.k_nearest(pt, k*2, out)
        out.sort(key=lambda x:x[0])
        cars = [c for _,c in out]
        if status_filter is None:
            return cars[:k]
        return [c for c in cars if c.status == status_filter][:k]

# -----------------------------
# Domain entities (2a)
# -----------------------------
class Rider:
    def __init__(self, rider_id: str, start_xy: Tuple[float,float], dest_xy: Tuple[float,float], request_time: float):
        self.id = rider_id
        self.start_xy = start_xy
        self.dest_xy = dest_xy
        self.request_time = request_time
        self.pickup_time: Optional[float] = None
        self.dropoff_time: Optional[float] = None
        self.start_node: Optional[str] = None
        self.dest_node: Optional[str] = None
        self.assigned_car: Optional["Car"] = None
        self.status = "requested"

class Car:
    def __init__(self, car_id: str, start_node: str, xy: Tuple[float,float]):
        self.id = car_id
        self.location = start_node   # current node id
        self.coordinates = xy        # x,y
        self.status = "available"    # 'available', 'to_pickup', 'to_dest'
        self.assigned_rider: Optional[Rider] = None
        self.route: List[str] = []
        self.route_time: float = 0.0
        # instrumentation
        self.busy_start: Optional[float] = None
        self.total_busy: float = 0.0

    def start_busy(self, t: float):
        if self.busy_start is None:
            self.busy_start = t

    def end_busy(self, t: float):
        if self.busy_start is not None:
            self.total_busy += (t - self.busy_start)
            self.busy_start = None

# -----------------------------
# Events (2a)
# -----------------------------
@dataclass(order=True)
class Event:
    time: float
    type: str=field(compare=False)
    payload: object=field(compare=False, default=None)

# -----------------------------
# Simulation with real-time animation and surge (2a–2d)
# -----------------------------
class Simulation:
    def __init__(self, *, max_time: float, num_cars: int, mean_arrival: float, map_file: str,
                 png_path: str, realtime: bool, surge: bool):
        self.max_time = max_time
        self.mean_arrival = mean_arrival
        self.png_path = png_path
        self.realtime = realtime
        self.surge_enabled = surge

        # world state
        self.time = 0.0
        self.events: List[Event] = []
        self.cars: Dict[str, Car] = {}
        self.riders: Dict[str, Rider] = {}
        self.completed: List[Rider] = []

        # graph
        self.graph = Graph()
        self.graph.load_map(map_file)

        # map bounds for random coords + zones
        xs = [x for x,_ in self.graph.node_coordinates.values()]
        ys = [y for _,y in self.graph.node_coordinates.values()]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        self.width = self.max_x - self.min_x or 6
        self.height = self.max_y - self.min_y or 6

        # quadtree
        self.quad = EnhancedQuadtree(Rectangle(self.min_x-1, self.min_y-1, self.width+2, self.height+2))

        # instrumentation
        self.wait_times: List[float] = []
        self.trip_times: List[float] = []
        self.trips_per_car: Dict[str,int] = defaultdict(int)

        # RNG for reproducibility (optional: set seed before constructing)
        self.rng = random.Random()

        # matplotlib (lazy)
        self._anim_fig = None
        self._anim_axes = None

        self._deploy_cars(num_cars)
        self._schedule(0.0, "RIDER_REQUEST", None)

    # ------------- utilities -------------
    def _rand_coord(self) -> Tuple[float,float]:
        return (self.rng.uniform(self.min_x, self.max_x),
                self.rng.uniform(self.min_y, self.max_y))

    def _schedule(self, t: float, ttype: str, payload):
        heapq.heappush(self.events, Event(t, ttype, payload))

    # ------------- surge -------------
    def _zone_index(self, xy: Tuple[float,float]) -> Tuple[int,int]:
        x,y = xy
        ix = min(SURGE_GRID-1, max(0, int((x-self.min_x)/max(1e-9,self.width) * SURGE_GRID)))
        iy = min(SURGE_GRID-1, max(0, int((y-self.min_y)/max(1e-9,self.height) * SURGE_GRID)))
        return ix, iy

    def _surge_multiplier_for_pickup(self, pickup_xy: Tuple[float,float]) -> float:
        if not self.surge_enabled:
            return 1.0
        # demand: waiting riders in zone
        zx, zy = self._zone_index(pickup_xy)
        demand = 0
        for r in self.riders.values():
            if r.status in ("requested","waiting"):
                if self._zone_index(r.start_xy) == (zx,zy):
                    demand += 1
        # supply: available cars in zone
        supply = 0
        for c in self.cars.values():
            if c.status == "available" and self._zone_index(c.coordinates) == (zx,zy):
                supply += 1
        ratio = demand / max(1, supply)
        mult = 1.0 + SURGE_BETA * max(0.0, ratio - 1.0)
        return mult

    # ------------- initialization -------------
    def _deploy_cars(self, n: int):
        for i in range(n):
            xy = self._rand_coord()
            node = self.graph.find_nearest_vertex(xy)
            car = Car(f"CAR_{i+1:03d}", node, xy)
            self.cars[car.id] = car
            self.quad.insert_car(car)
        print(f"Fleet deployed: {len(self.cars)} cars")

    # ------------- generation -------------
    def _new_rider(self) -> Rider:
        rid = f"RIDER_{len(self.riders)+1:05d}"
        start_xy = self._rand_coord()
        dest_xy = self._rand_coord()
        # Avoid zero-length
        while (abs(dest_xy[0]-start_xy[0]) + abs(dest_xy[1]-start_xy[1])) < 0.5:
            dest_xy = self._rand_coord()
        r = Rider(rid, start_xy, dest_xy, self.time)
        r.start_node = self.graph.find_nearest_vertex(start_xy)
        r.dest_node = self.graph.find_nearest_vertex(dest_xy)
        self.riders[rid] = r
        return r

    # ------------- matching & movement -------------
    def _dispatch(self, rider: Rider):
        # k-nearest by Euclidean in quadtree
        klist = self.quad.find_k_nearest_cars(rider.start_xy, K_NEAREST, status_filter="available")
        if not klist:
            rider.status = "waiting"  # keep them in queue; surge will increase
            return
        best_car = None
        best_time = float('inf')
        # consider surge
        surge_mult = self._surge_multiplier_for_pickup(rider.start_xy)
        for car in klist:
            # compute travel time car->pickup using graph
            # map car current XY to nearest node (keep it in sync)
            car_node = self.graph.find_nearest_vertex(car.coordinates)
            car.location = car_node
            t, path = dijkstra(self.graph, car_node, rider.start_node, surge_mult)
            if t < best_time and path:
                best_time = t
                best_car = car
        if not best_car:
            rider.status = "waiting"
            return
        # assign and schedule pickup
        best_car.status = "to_pickup"
        best_car.assigned_rider = rider
        best_car.start_busy(self.time)
        rider.assigned_car = best_car
        rider.status = "assigned"
        # remove from quadtree while busy
        self.quad.remove_car(best_car)
        pickup_eta = self.time + best_time
        self._schedule(pickup_eta, "PICKUP_ARRIVAL", best_car)

    def _on_pickup(self, car: Car):
        rider = car.assigned_rider
        if not rider: 
            return
        rider.pickup_time = self.time
        rider.status = "in_trip"
        car.status = "to_dest"
        # now compute pickup->destination (include surge based on pickup zone)
        surge_mult = self._surge_multiplier_for_pickup(rider.start_xy)
        t, path = dijkstra(self.graph, rider.start_node, rider.dest_node, surge_mult)
        eta = self.time + (t if t != float('inf') else 0.0)
        self._schedule(eta, "DROPOFF_ARRIVAL", car)

    def _on_dropoff(self, car: Car):
        rider = car.assigned_rider
        if not rider:
            return
        rider.dropoff_time = self.time
        rider.status = "done"
        car.status = "available"
        car.end_busy(self.time)
        self.completed.append(rider)
        # place car at rider destination
        car.location = rider.dest_node
        car.coordinates = self.graph.node_coordinates[car.location]
        # metrics
        if rider.pickup_time is not None:
            self.wait_times.append(rider.pickup_time - rider.request_time)
            self.trip_times.append(rider.dropoff_time - rider.pickup_time)
        self.trips_per_car[car.id] += 1
        # remove linkage
        car.assigned_rider = None
        rider.assigned_car = None
        # back to quadtree
        self.quad.insert_car(car)

    # ------------- main loop -------------
    def run(self):
        # prime first rider
        while self.events and self.time < self.max_time:
            ev = heapq.heappop(self.events)
            self.time = ev.time
            if self.time >= self.max_time:
                break
            if ev.type == "RIDER_REQUEST":
                # create rider and try dispatch
                r = self._new_rider()
                self._dispatch(r)
                # schedule next request
                dt = random.expovariate(1.0/self.mean_arrival) if self.mean_arrival > 0 else 1.0
                self._schedule(self.time + dt, "RIDER_REQUEST", None)
            elif ev.type == "PICKUP_ARRIVAL":
                self._on_pickup(ev.payload)  # car
            elif ev.type == "DROPOFF_ARRIVAL":
                self._on_dropoff(ev.payload)  # car
            else:
                pass
            # realtime?
            if self.realtime:
                self._realtime_draw()

        # drain waiting riders by not scheduling more requests
        print("Simulation ended at T=%.2f with %d completed trips" % (self.time, len(self.completed)))
        self._final_report()
        self._save_png_and_open()

    # ------------- analytics -------------
    def _final_report(self):
        total_requests = len(self.riders)
        total_trips = len(self.completed)
        avg_wait = sum(self.wait_times)/len(self.wait_times) if self.wait_times else 0.0
        avg_trip = sum(self.trip_times)/len(self.trip_times) if self.trip_times else 0.0
        # utilization
        total_busy = sum(c.total_busy for c in self.cars.values())
        util = (total_busy / (len(self.cars)*max(1.0,self.time))) * 100.0
        print("Requests:", total_requests)
        print("Completed:", total_trips, "(%.1f%%)" % (100.0*total_trips/max(1,total_requests)))
        print("Avg wait: %.2fs  Avg trip: %.2fs  Utilization: %.1f%%" % (avg_wait, avg_trip, util))
        # store for plotting
        self._metrics = dict(
            total_requests=total_requests,
            total_trips=total_trips,
            completion_rate=100.0*total_trips/max(1,total_requests),
            avg_wait=avg_wait,
            avg_trip=avg_trip,
            utilization=util
        )

    # ------------- visualization -------------
    def _ensure_fig(self):
        import matplotlib.pyplot as plt
        if self._anim_fig is None:
            self._anim_fig, self._anim_axes = plt.subplots(1,3, figsize=(16,5))
            self._anim_fig.canvas.manager.set_window_title("Ride-Sharing Simulation")
            plt.tight_layout()

    def _realtime_draw(self):
        import matplotlib.pyplot as plt
        self._ensure_fig()
        ax0, ax1, ax2 = self._anim_axes
        for ax in (ax0, ax1, ax2):
            ax.cla()
        # ax0: map + cars + waiting riders
        ax0.set_title("Live Map")
        ax0.set_xlim(self.min_x-1, self.max_x+1)
        ax0.set_ylim(self.min_y-1, self.max_y+1)
        # nodes
        xs = [x for x,_ in self.graph.node_coordinates.values()]
        ys = [y for _,y in self.graph.node_coordinates.values()]
        ax0.scatter(xs, ys, s=10, alpha=0.3, label="Nodes")
        # cars
        avail = [(c.coordinates[0], c.coordinates[1]) for c in self.cars.values() if c.status=="available"]
        busy  = [(c.coordinates[0], c.coordinates[1]) for c in self.cars.values() if c.status!="available"]
        if avail:
            ax0.scatter([x for x,_ in avail],[y for _,y in avail], s=35, label="Available")
        if busy:
            ax0.scatter([x for x,_ in busy],[y for _,y in busy], s=35, marker="s", label="Busy")
        # waiting riders
        wait_pts = [(r.start_xy[0], r.start_xy[1]) for r in self.riders.values() if r.status in ("requested","waiting")]
        if wait_pts:
            ax0.scatter([x for x,_ in wait_pts],[y for _,y in wait_pts], s=25, marker="x", label="Waiting")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.text(0.02, 0.95, f"T={self.time:.1f}", transform=ax0.transAxes)

        # ax1: trips per car
        ax1.set_title("Trips per Car")
        if self.trips_per_car:
            car_ids = list(self.trips_per_car.keys())
            counts = [self.trips_per_car[c] for c in car_ids]
            ax1.bar(car_ids, counts)
            ax1.tick_params(axis='x', labelrotation=90)
        else:
            ax1.text(0.5,0.5,"No trips yet", ha='center', va='center')

        # ax2: wait time hist
        ax2.set_title("Wait Time (s)")
        if self.wait_times:
            ax2.hist(self.wait_times, bins=10, edgecolor='k')
        else:
            ax2.text(0.5,0.5,"N/A", ha='center', va='center')

        import matplotlib
        self._anim_fig.canvas.draw()
        self._anim_fig.canvas.flush_events()
        plt.pause(0.001)

    def _save_png_and_open(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,3, figsize=(16,5))
        fig.suptitle("Ride-Sharing Simulation — Summary", fontsize=14, fontweight='bold')

        # panel 1: final car positions
        ax = axs[0]
        ax.set_title("Final Car Positions")
        ax.set_xlim(self.min_x-1, self.max_x+1)
        ax.set_ylim(self.min_y-1, self.max_y+1)
        xs = [x for x,_ in self.graph.node_coordinates.values()]
        ys = [y for _,y in self.graph.node_coordinates.values()]
        ax.scatter(xs, ys, s=10, alpha=0.3, label="Nodes")
        cars = [c.coordinates for c in self.cars.values()]
        if cars:
            ax.scatter([x for x,_ in cars],[y for _,y in cars], s=35, label="Cars")
        ax.legend(loc="upper right", fontsize=8)

        # panel 2: trips per car
        ax = axs[1]
        ax.set_title("Trips per Car")
        car_ids = list(self.trips_per_car.keys())
        counts = [self.trips_per_car[c] for c in car_ids]
        if car_ids:
            ax.bar(car_ids, counts)
            ax.tick_params(axis='x', labelrotation=90)
        else:
            ax.text(0.5,0.5,"No trips", ha='center', va='center')

        # panel 3: wait time hist + metrics
        ax = axs[2]
        ax.set_title("Wait Times & Metrics")
        if self.wait_times:
            ax.hist(self.wait_times, bins=10, edgecolor='k', alpha=0.7)
        txt = (
            f"Requests: {self._metrics.get('total_requests',0)}\n"
            f"Completed: {self._metrics.get('total_trips',0)} "
            f"({self._metrics.get('completion_rate',0):.1f}%)\n"
            f"Avg wait: {self._metrics.get('avg_wait',0):.2f}s\n"
            f"Avg trip: {self._metrics.get('avg_trip',0):.2f}s\n"
            f"Utilization: {self._metrics.get('utilization',0):.1f}%\n"
            f"Surge: {'ON' if self.surge_enabled else 'OFF'}"
        )
        ax.text(0.05, 0.95, txt, va='top')

        plt.tight_layout()
        plt.savefig(self.png_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        try:
            # open via default handler (will open image viewer or browser depending on OS)
            webbrowser.open(os.path.abspath(self.png_path))
        except Exception:
            pass

# -----------------------------
# CLI
# -----------------------------
def ensure_map(path: str):
    if os.path.exists(path):
        return
    print(f"Creating default map at {path}")
    data = """start_node,end_node,travel_time
A,B,5
B,A,5
A,C,3
C,A,3
B,D,4
D,B,4
C,D,1
D,C,1
A,E,7
E,A,7
B,F,6
F,B,6
C,F,2
F,C,2
D,G,3
G,D,3
E,F,4
F,E,4
F,G,2
G,F,2"""
    with open(path,'w') as f:
        f.write(data)

def main(argv=None):
    p = argparse.ArgumentParser(description="Final ride-sharing simulation (2a–2d).")
    p.add_argument('--max-time', type=float, default=100.0)
    p.add_argument('--num-cars', type=int, default=12)
    p.add_argument('--arrival-rate', type=float, default=DEFAULT_MEAN_ARRIVAL)
    p.add_argument('--map-file', type=str, default='map.csv')
    p.add_argument('--png', type=str, default='simulation_summary.png')
    p.add_argument('--realtime', action='store_true', help='Enable live Matplotlib animation')
    p.add_argument('--surge', action='store_true', help='Enable surge pricing by zones (extra credit)')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    ensure_map(args.map_file)

    sim = Simulation(max_time=args.max_time,
                     num_cars=args.num_cars,
                     mean_arrival=args.arrival_rate,
                     map_file=args.map_file,
                     png_path=args.png,
                     realtime=args.realtime,
                     surge=args.surge)
    if args.realtime:
        # interactive mode setup
        import matplotlib.pyplot as plt
        plt.ion()
    sim.run()
    if args.realtime:
        import matplotlib.pyplot as plt
        print("Close the live window to end.")
        plt.ioff()

if __name__ == '__main__':
    main()
