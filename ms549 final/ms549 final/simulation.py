
from graph_basic import Graph

class Simulation:
    def __init__(self, map_filename="map.csv"):
        """
        Initialize the simulation state.
        
        Args:
            map_filename: Path to the CSV file containing the map data
        """
        self.cars = {}      # Dictionary to store Car objects (key: car_id, value: Car object)
        self.riders = {}    # Dictionary to store Rider objects (key: rider_id, value: Rider object)
        
        # Create and load the map
        self.map = Graph()
        self.map.load_from_file(map_filename)
        
        print("Simulation initialized with map data")
    
    def display_map(self):
        """Display the loaded map for debugging."""
        print(self.map)