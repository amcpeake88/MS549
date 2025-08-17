# final_simulation.py
import argparse
import heapq
import random
import math
from collections import defaultdict

# Core modules from your tree
from graph_basic import Graph
from car import Car
from rider import Rider
from enhanced_quadtree import EnhancedQuadtree, Rectangle

# Optional visualization – used only if available and method exists
try:
    from visualization_engine import VisualizationEngine
except Exception:
    VisualizationEngine = None

DEFAULT_MEAN_ARRIVAL = 2.0


class CompleteSimulation:
    """
    Integrated ride-sharing simulation:
    - Event-driven engine (min-heap)
    - Rider generation with exponential inter-arrivals
    - EnhancedQuadtree for nearby-car lookup
    - Dijkstra via Car.calculate_route()
    - Optional PNG viz via VisualizationEngine.create_complete_visualization()
    """

    def __init__(self,
                 max_time: float = 100.0,
                 num_cars: int = 5,
                 mean_arrival: float = DEFAULT_MEAN_ARRIVAL,
                 map_file: str = "map.csv",
                 output_png: str = "simulation_summary.png",
                 enable_viz: bool = False):
        self.max_time = max_time
        self.num_cars = num_cars
        self.mean_arrival = mean_arrival
        self.map_file = map_file
        self.output_png = output_png
        self.enable_viz = enable_viz

        # Simulation clock and queue
        self.current_time = 0.0
        self.event_queue = []  # (timestamp, type, payload)

        # Entities
        self.graph: Graph = Graph()
        self.cars: dict[str, Car] = {}
        self.riders: dict[str, Rider] = {}

        # Spatial index
        self.qt = None  # EnhancedQuadtree

        # Metrics
        self.completed_trips = []
        self.wait_times = []
        self.trip_durations = []
        self.car_trip_counts = defaultdict(int)

        # ID counters
        self._rider_seq = 0

    # ---------- setup ----------
    def _load_graph(self):
        self.graph.load_map_data(self.map_file)
        if not getattr(self.graph, "node_coordinates", None):
            raise RuntimeError("Graph must provide node_coordinates. Check graph_basic.load_map_data().")

    def _map_bounds(self):
        coords = list(self.graph.node_coordinates.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, (max_x - min_x), (max_y - min_y)

    def _init_quadtree(self):
        min_x, min_y, w, h = self._map_bounds()
        self.qt = EnhancedQuadtree(Rectangle(min_x, min_y, w, h))

    def _deploy_cars(self):
        for i in range(self.num_cars):
            cid = f"CAR_{i+1:03d}"
            # random coordinate in bounds, then snap to nearest vertex for routing
            min_x, min_y, w, h = self._map_bounds()
            cx = random.uniform(min_x, min_x + w)
            cy = random.uniform(min_y, min_y + h)
            car_coords = (cx, cy)
            vtx = self.graph.find_nearest_vertex(car_coords)
            car = Car(cid, vtx)
            car.coordinates = car_coords
            car.status = "available"
            self.cars[cid] = car
            self.qt.insert_car(car)

    # ---------- event engine ----------
    def _push_event(self, t, etype, payload):
        heapq.heappush(self.event_queue, (t, etype, payload))

    def _schedule_first_request(self):
        rider = self._new_rider()
        self._push_event(self.current_time, "RIDER_REQUEST", rider)

    def _new_rider(self) -> Rider:
        self._rider_seq += 1
        rid = f"RIDER_{self._rider_seq:04d}"

        # random start/end by sampling coordinates then snapping to nearest vertex
        min_x, min_y, w, h = self._map_bounds()
        sx, sy = random.uniform(min_x, min_x + w), random.uniform(min_y, min_y + h)
        ex, ey = random.uniform(min_x, min_x + w), random.uniform(min_y, min_y + h)

        s_v = self.graph.find_nearest_vertex((sx, sy))
        e_v = self.graph.find_nearest_vertex((ex, ey))
        while e_v == s_v:
            ex, ey = random.uniform(min_x, min_x + w), random.uniform(min_y, min_y + h)
            e_v = self.graph.find_nearest_vertex((ex, ey))

        r = Rider(rid, s_v, e_v)
        r.request_time = self.current_time
        r.coordinates = (sx, sy)  # for spatial search
        self.riders[rid] = r
        return r

    # ---------- handlers ----------
    def _handle_request(self, rider: Rider):
        # k nearest available cars by EnhancedQuadtree
        candidates = self.qt.find_k_nearest_cars(rider.coordinates, k=5, status_filter="available")
        if not candidates:
            rider.status = "no_car_available"
        else:
            best_car = None
            best_time = float("inf")
            for car in candidates:
                car.location = self.graph.find_nearest_vertex(car.coordinates)
                if car.calculate_route(rider.start_location, self.graph):
                    if car.route_time < best_time:
                        best_time = car.route_time
                        best_car = car
            if best_car:
                self.qt.remove_car(best_car)
                best_car.assigned_rider = rider
                best_car.status = "en_route_to_pickup"
                rider.assigned_car = best_car
                rider.status = "waiting_for_pickup"
                self._push_event(self.current_time + best_time, "PICKUP", best_car)
            else:
                rider.status = "no_route_available"

        # daisy-chain next rider
        nxt = self.current_time + random.expovariate(1.0 / self.mean_arrival)
        if nxt < self.max_time:
            self._push_event(nxt, "RIDER_REQUEST", self._new_rider())

    def _handle_pickup(self, car: Car):
        r = car.assigned_rider
        if not r:
            return
        car.location = r.start_location
        car.coordinates = self.graph.node_coordinates.get(r.start_location, car.coordinates)
        car.status = "en_route_to_destination"
        r.status = "in_transit"
        if car.calculate_route(r.destination, self.graph):
            self._push_event(self.current_time + car.route_time, "DROPOFF", car)

    def _handle_dropoff(self, car: Car):
        r = car.assigned_rider
        if not r:
            return
        car.location = r.destination
        car.coordinates = self.graph.node_coordinates.get(r.destination, car.coordinates)
        car.status = "available"
        r.status = "completed"
        self.qt.insert_car(car)

        # metrics
        wait_time = self.current_time - (r.request_time or 0)
        trip_duration = self.current_time - (r.request_time or 0)
        self.wait_times.append(wait_time)
        self.trip_durations.append(trip_duration)
        self.car_trip_counts[car.id] += 1
        self.completed_trips.append(
            dict(rider_id=r.id, car_id=car.id,
                 start=r.start_location, end=r.destination,
                 request_time=r.request_time, completion_time=self.current_time,
                 wait_time=wait_time, trip_duration=trip_duration)
        )
        car.assigned_rider = None
        r.assigned_car = None

    # ---------- run ----------
    def run(self):
        # setup
        self._load_graph()
        self._init_quadtree()
        self._deploy_cars()
        self._schedule_first_request()

        # event loop
        while self.event_queue and self.current_time < self.max_time:
            t, etype, payload = heapq.heappop(self.event_queue)
            self.current_time = t
            if self.current_time >= self.max_time:
                break
            if etype == "RIDER_REQUEST":
                self._handle_request(payload)
            elif etype == "PICKUP":
                self._handle_pickup(payload)
            elif etype == "DROPOFF":
                self._handle_dropoff(payload)

        return self.metrics()

    # ---------- metrics & optional viz ----------
    def metrics(self):
        total_requests = len(self.riders)
        total_trips = len(self.completed_trips)
        completion_rate = (100.0 * total_trips / total_requests) if total_requests else 0.0
        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0.0
        avg_trip = sum(self.trip_durations) / len(self.trip_durations) if self.trip_durations else 0.0

        return dict(
            simulation_time=self.current_time,
            fleet_size=len(self.cars),
            total_requests=total_requests,
            completed_trips=total_trips,
            completion_rate=completion_rate,
            avg_wait_time=avg_wait,
            avg_trip_duration=avg_trip
        )

    def maybe_generate_png(self):
        """Call only if you want a PNG and VisualizationEngine is present with the right API."""
        if not self.enable_viz:
            return None
        if VisualizationEngine is None:
            print("Visualization disabled: visualization_engine.py not importable.")
            return None

        sim_data = dict(
            cars=self.cars,
            riders=self.riders,
            completed_trips=self.completed_trips,
            metrics=self.metrics()
        )

        engine = VisualizationEngine(simulation_data=sim_data, graph=self.graph, output_filename=self.output_png)
        # Only call the public method that we standardized on
        if hasattr(engine, "create_complete_visualization"):
            return engine.create_complete_visualization()
        print("Visualization disabled: VisualizationEngine.create_complete_visualization() not found.")
        return None


# ------------ optional CLI for direct running ------------
def _parse_args():
    p = argparse.ArgumentParser(description="Integrated ride-sharing simulation")
    p.add_argument("--max-time", type=float, default=100.0)
    p.add_argument("--num-cars", type=int, default=5)
    p.add_argument("--arrival-rate", type=float, default=DEFAULT_MEAN_ARRIVAL)
    p.add_argument("--map-file", type=str, default="map.csv")
    p.add_argument("--output", type=str, default="simulation_summary.png")
    p.add_argument("--viz", action="store_true", help="Generate PNG if VisualizationEngine is available")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    sim = CompleteSimulation(
        max_time=args.max_time,
        num_cars=args.num_cars,
        mean_arrival=args.arrival_rate,
        map_file=args.map_file,
        output_png=args.output,
        enable_viz=args.viz,
    )
    metrics = sim.run()
    print("\nFINAL METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    sim.maybe_generate_png()


if __name__ == "__main__":
    main()
