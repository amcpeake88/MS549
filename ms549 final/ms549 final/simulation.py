# simulation.py - Event-Driven Simulation System
import heapq
import math
from graph_basic import Graph

# Constants for simulation
TRAVEL_SPEED_FACTOR = 1.0  # Time per unit distance

class Simulation:
    """Enhanced Simulation class with event-driven architecture."""
    
    def __init__(self, map_file=None):
        """Initialize the simulation with optional map loading."""
        self.cars = {}           # Dictionary of Car objects keyed by ID
        self.riders = {}         # Dictionary of Rider objects keyed by ID
        self.event_queue = []    # Min-heap for event scheduling
        self.current_time = 0.0  # Simulation clock
        self.completed_rides = []  # Track completed rides for statistics
        
        # Load map if provided
        if map_file:
            try:
                self.graph = Graph()
                self.graph.load_from_file(map_file)
                print(f"Map loaded successfully from {map_file}")
            except Exception as e:
                print(f"Failed to load map: {e}")
                self.graph = None
        else:
            self.graph = None
    
    def add_event(self, timestamp, event_type, data):
        """Add an event to the priority queue."""
        event = (timestamp, event_type, data)
        heapq.heappush(self.event_queue, event)
        print(f"EVENT SCHEDULED: {event}")
    
    def find_closest_car_brute_force(self, rider_location):
        """Find the closest available car using brute force search."""
        available_cars = [car for car in self.cars.values() 
                         if car.status == "available"]
        
        if not available_cars:
            return None
        
        closest_car = None
        min_distance = float('inf')
        
        for car in available_cars:
            distance = self.calculate_manhattan_distance(car.location, rider_location)
            if distance < min_distance:
                min_distance = distance
                closest_car = car
        
        return closest_car
    
    def calculate_manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        # Handle both coordinate tuples and node names
        if isinstance(loc1, tuple) and isinstance(loc2, tuple):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
        else:
            # For node names, use simple heuristic (could be enhanced with actual coordinates)
            # For now, treat different nodes as distance 1
            return 0 if loc1 == loc2 else 1
    
    def calculate_travel_time(self, start_location, end_location):
        """Calculate travel time using Manhattan distance heuristic."""
        distance = self.calculate_manhattan_distance(start_location, end_location)
        travel_time = distance * TRAVEL_SPEED_FACTOR
        return max(travel_time, 0.1)  # Minimum travel time to avoid zero-time events
    
    def handle_rider_request(self, rider):
        """Handle a new rider request event."""
        print(f"TIME {self.current_time:.1f}: Processing request from RIDER {rider.id}")
        
        # Find closest available car
        car = self.find_closest_car_brute_force(rider.start_location)
        
        if car is None:
            print(f"TIME {self.current_time:.1f}: No available cars for RIDER {rider.id}")
            return
        
        # Link rider and car
        car.assigned_rider = rider
        rider.assigned_car = car  # Back-reference for easier tracking
        
        # Update car status
        car.status = "en_route_to_pickup"
        rider.status = "waiting_for_pickup"
        
        # Calculate travel time to pickup location
        pickup_duration = self.calculate_travel_time(car.location, rider.start_location)
        
        # Schedule pickup arrival event
        pickup_time = self.current_time + pickup_duration
        self.add_event(pickup_time, "ARRIVAL", car)
        
        print(f"TIME {self.current_time:.1f}: CAR {car.id} dispatched to RIDER {rider.id}")
        print(f"   Pickup scheduled for time {pickup_time:.1f}")
    
    def handle_arrival(self, car):
        """Handle car arrival event (either pickup or dropoff)."""
        if car.status == "en_route_to_pickup":
            self._handle_pickup_arrival(car)
        elif car.status == "en_route_to_destination":
            self._handle_dropoff_arrival(car)
        else:
            print(f"ERROR: Unexpected car status '{car.status}' for ARRIVAL event")
    
    def _handle_pickup_arrival(self, car):
        """Handle car arrival at pickup location."""
        rider = car.assigned_rider
        
        print(f"TIME {self.current_time:.1f}: CAR {car.id} picked up RIDER {rider.id}")
        
        # Update car location to pickup location
        car.location = rider.start_location
        print(f"   CAR {car.id} location updated to {car.location}")
        
        # Update statuses
        car.status = "en_route_to_destination"
        rider.status = "in_car"
        
        # Calculate travel time to destination
        dropoff_duration = self.calculate_travel_time(rider.start_location, rider.destination)
        
        # Schedule dropoff arrival event
        dropoff_time = self.current_time + dropoff_duration
        self.add_event(dropoff_time, "ARRIVAL", car)
        
        print(f"   Dropoff scheduled for time {dropoff_time:.1f}")
    
    def _handle_dropoff_arrival(self, car):
        """Handle car arrival at dropoff location."""
        rider = car.assigned_rider
        
        print(f"TIME {self.current_time:.1f}: CAR {car.id} dropped off RIDER {rider.id}")
        
        # Update car location to dropoff location
        car.location = rider.destination
        print(f"   CAR {car.id} location updated to {car.location}")
        
        # Update statuses
        car.status = "available"
        rider.status = "completed"
        
        # Record completed ride
        ride_info = {
            'rider_id': rider.id,
            'car_id': car.id,
            'start_time': rider.request_time if hasattr(rider, 'request_time') else 0,
            'end_time': self.current_time,
            'start_location': rider.start_location,
            'end_location': rider.destination
        }
        self.completed_rides.append(ride_info)
        
        # Unlink rider and car
        car.assigned_rider = None
        rider.assigned_car = None
        
        print(f"   Ride completed. CAR {car.id} now available.")
    
    def run(self, max_time=100.0):
        """Main event loop - the heart of the simulation."""
        print("=" * 60)
        print("STARTING EVENT-DRIVEN SIMULATION")
        print("=" * 60)
        print(f"Initial cars: {len(self.cars)}")
        print(f"Initial riders: {len(self.riders)}")
        print(f"Max simulation time: {max_time}")
        print()
        
        # Schedule initial rider requests
        for rider in self.riders.values():
            # Schedule rider requests at time 0 (or could be staggered)
            request_time = 0.0
            rider.request_time = request_time
            self.add_event(request_time, "RIDER_REQUEST", rider)
        
        print("\nBEGINNING EVENT PROCESSING:")
        print("-" * 40)
        
        # Main event loop using min-heap priority queue
        while self.event_queue and self.current_time < max_time:
            # Pop next event from priority queue
            timestamp, event_type, data = heapq.heappop(self.event_queue)
            
            # Advance simulation clock
            self.current_time = timestamp
            
            # Handle the event based on type
            if event_type == "RIDER_REQUEST":
                self.handle_rider_request(data)
            elif event_type == "ARRIVAL":
                self.handle_arrival(data)
            else:
                print(f"WARNING: Unknown event type '{event_type}'")
            
            print()  # Blank line between events for readability
        
        # Simulation complete
        print("-" * 40)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print simulation statistics."""
        print(f"Final simulation time: {self.current_time:.1f}")
        print(f"Total completed rides: {len(self.completed_rides)}")
        print(f"Events remaining in queue: {len(self.event_queue)}")
        
        # Car status summary
        car_statuses = {}
        for car in self.cars.values():
            status = car.status
            car_statuses[status] = car_statuses.get(status, 0) + 1
        
        print("\\nFinal car statuses:")
        for status, count in car_statuses.items():
            print(f"  {status}: {count} cars")
        
        # Rider status summary
        rider_statuses = {}
        for rider in self.riders.values():
            status = rider.status
            rider_statuses[status] = rider_statuses.get(status, 0) + 1
        
        print("\\nFinal rider statuses:")
        for status, count in rider_statuses.items():
            print(f"  {status}: {count} riders")
    
    def display_map(self):
        """Display loaded map information."""
        if self.graph:
            print("Loaded map nodes:", sorted(self.graph.get_all_nodes()))
        else:
            print("No map loaded")


# Enhanced Car class with assigned_rider attribute
class Car:
    """Car class enhanced for event-driven simulation."""
    
    def __init__(self, car_id, initial_location):
        self.id = car_id
        self.location = initial_location
        self.status = 'available'  # available, en_route_to_pickup, en_route_to_destination
        self.assigned_rider = None  # Link to assigned Rider object
        self.route = []
        self.route_time = 0
        print(f"Car {self.id} created at location {self.location}.")
    
    def __str__(self):
        rider_info = f" (assigned to {self.assigned_rider.id})" if self.assigned_rider else ""
        return f"Car {self.id} at {self.location}, status: {self.status}{rider_info}"


# Enhanced Rider class with assigned_car attribute
class Rider:
    """Rider class enhanced for event-driven simulation."""
    
    def __init__(self, rider_id, start_location, destination):
        self.id = rider_id
        self.start_location = start_location
        self.destination = destination
        self.status = 'waiting_for_car'  # waiting_for_car, waiting_for_pickup, in_car, completed
        self.assigned_car = None  # Link to assigned Car object
        self.request_time = None  # Will be set when request is made
        print(f"Rider {self.id} requesting ride from {self.start_location} to {self.destination}.")
    
    def __str__(self):
        car_info = f" (assigned car {self.assigned_car.id})" if self.assigned_car else ""
        return f"Rider {self.id}: {self.start_location} → {self.destination}, status: {self.status}{car_info}"