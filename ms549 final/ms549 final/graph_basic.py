import csv
import math
import random

class Graph:
    def __init__(self):
        """Initialize an empty adjacency list and coordinate system."""
        self.adjacency_list = {}
        self.node_coordinates = {}  # NEW: Store (x,y) coordinates for each node
    
    def add_edge(self, start_node, end_node, weight):
        """
        Add a directed edge to the graph.
        
        Args:
            start_node: The starting node ID (e.g., 'A')
            end_node: The ending node ID (e.g., 'B') 
            weight: The travel time/weight of the edge
        """
        # Check if start_node is already a key in the dictionary
        if start_node not in self.adjacency_list:
            self.adjacency_list[start_node] = []
        
        # Append the (end_node, weight) tuple to the list for start_node
        self.adjacency_list[start_node].append((end_node, weight))
    
    def load_from_file(self, filename):
        """
        Load graph data from a CSV file (legacy 3-column format).
        
        Args:
            filename: Path to the CSV file containing map data
        """
        try:
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    start_node = row['start_node']
                    end_node = row['end_node']
                    travel_time = float(row['travel_time'])
                    
                    # Use add_edge method to populate the graph
                    self.add_edge(start_node, end_node, travel_time)
                    
            print(f"Successfully loaded map from {filename}")
            
        except FileNotFoundError:
            print(f"Error: Could not find file {filename}")
        except Exception as e:
            print(f"Error loading map: {e}")
    
    def load_map_data(self, filename):
        """
        Load both edge data and node coordinates from CSV file.
        Supports both legacy 3-column and new 7-column formats.
        
        Args:
            filename: Path to the CSV file containing map data
        """
        try:
            with open(filename, 'r', newline='') as csvfile:
                # First, detect the format by checking column headers
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames
                
                # Check if this is the new 7-column format
                is_new_format = ('start_x' in fieldnames and 'start_y' in fieldnames and 
                               'end_x' in fieldnames and 'end_y' in fieldnames)
                
                if is_new_format:
                    print("Detected new 7-column format with coordinates")
                    self._load_new_format(filename)
                else:
                    print("Detected legacy 3-column format, generating coordinates")
                    self._load_legacy_format(filename)
                    
        except Exception as e:
            print(f"Error loading map data: {e}")
            # Fallback to legacy method
            self.load_from_file(filename)
            self._generate_coordinates_for_legacy()
    
    def _load_new_format(self, filename):
        """Load from new 7-column format: start_node_id,start_x,start_y,end_node_id,end_x,end_y,weight"""
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Skip comment lines
                if any(row[key].startswith('#') for key in row.keys() if row[key]):
                    continue
                
                start_node = row['start_node_id']
                start_x = float(row['start_x'])
                start_y = float(row['start_y'])
                end_node = row['end_node_id']
                end_x = float(row['end_x'])
                end_y = float(row['end_y'])
                weight = float(row['weight'])
                
                # Store coordinates for both nodes
                self.node_coordinates[start_node] = (start_x, start_y)
                self.node_coordinates[end_node] = (end_x, end_y)
                
                # Add edges for undirected graph
                self.add_edge(start_node, end_node, weight)
                self.add_edge(end_node, start_node, weight)
        
        print(f"Loaded {len(self.node_coordinates)} node coordinates from new format")
    
    def _load_legacy_format(self, filename):
        """Load from legacy 3-column format and generate coordinates"""
        # First load the edges
        self.load_from_file(filename)
        
        # Then generate appropriate coordinates
        nodes = list(self.adjacency_list.keys())
        
        if len(nodes) <= 10:
            # Small debug map (7x7 grid)
            self._generate_debug_coordinates()
        else:
            # Large production map (1000x1000)
            self._generate_production_coordinates()
    
    def _generate_coordinates_for_legacy(self):
        """Generate coordinates for legacy maps that don't have them"""
        nodes = list(self.adjacency_list.keys())
        
        if len(nodes) <= 10:
            self._generate_debug_coordinates()
        else:
            self._generate_production_coordinates()
    
    def _generate_debug_coordinates(self):
        """Generate coordinates for debug map (7x7 grid)."""
        # Simple grid layout for nodes A, B, C, D, E, F, G, etc.
        debug_coords = {
            'A': (1, 1), 'B': (3, 1), 'C': (5, 1),
            'D': (1, 3), 'E': (3, 3), 'F': (5, 3),
            'G': (1, 5), 'H': (3, 5), 'I': (5, 5)
        }
        
        # Assign coordinates to actual nodes
        nodes = sorted(list(self.adjacency_list.keys()))
        coord_values = list(debug_coords.values())
        
        for i, node in enumerate(nodes):
            if i < len(coord_values):
                self.node_coordinates[node] = coord_values[i]
            else:
                # Random coordinates for additional nodes within 7x7 grid
                self.node_coordinates[node] = (
                    random.uniform(0, 7), 
                    random.uniform(0, 7)
                )
        
        print(f"Generated debug coordinates for {len(self.node_coordinates)} nodes (7x7 grid)")
    
    def _generate_production_coordinates(self):
        """Generate coordinates for production map (1000x1000)."""
        # Random coordinates across the 1000x1000 space
        for node in self.adjacency_list.keys():
            self.node_coordinates[node] = (
                random.uniform(0, 1000),
                random.uniform(0, 1000)
            )
        
        print(f"Generated production coordinates for {len(self.node_coordinates)} nodes (1000x1000 grid)")
    
    def find_nearest_vertex(self, point):
        """
        Find the graph vertex closest to a given (x,y) point.
        This is the critical "snapping" function for the simulation.
        
        Args:
            point: (x, y) tuple representing a coordinate
            
        Returns:
            str: The node ID of the closest vertex
        """
        if not self.node_coordinates:
            # Fallback if no coordinates available
            return list(self.adjacency_list.keys())[0] if self.adjacency_list else None
        
        min_distance = float('inf')
        nearest_vertex = None
        
        # Extract coordinates from point
        if isinstance(point, tuple) and len(point) == 2:
            px, py = point
        elif hasattr(point, 'coordinates'):
            px, py = point.coordinates
        else:
            raise ValueError(f"Invalid point format: {point}")
        
        # Find closest vertex
        for vertex_id, (vx, vy) in self.node_coordinates.items():
            distance = math.sqrt((px - vx)**2 + (py - vy)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = vertex_id
        
        return nearest_vertex
    
    def get_map_bounds(self):
        """
        Get the bounding box of the map based on node coordinates.
        
        Returns:
            tuple: (min_x, min_y, max_x, max_y) or None if no coordinates
        """
        if not self.node_coordinates:
            return None
        
        x_coords = [coord[0] for coord in self.node_coordinates.values()]
        y_coords = [coord[1] for coord in self.node_coordinates.values()]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_coordinate_range(self):
        """
        Get the coordinate range for the map.
        
        Returns:
            tuple: (width, height) of the map
        """
        bounds = self.get_map_bounds()
        if bounds is None:
            return (7, 7)  # Default debug size
        
        min_x, min_y, max_x, max_y = bounds
        return (max_x - min_x, max_y - min_y)
    
    def __str__(self):
        """
        Return a nicely formatted string representation of the adjacency list.
        Essential for debugging.
        """
        if not self.adjacency_list:
            return "Graph is empty"
        
        result = "Graph Adjacency List:\n"
        result += "=" * 25 + "\n"
        
        for node, neighbors in sorted(self.adjacency_list.items()):
            coord_info = ""
            if node in self.node_coordinates:
                x, y = self.node_coordinates[node]
                coord_info = f" at ({x:.1f}, {y:.1f})"
            
            result += f"Node {node}{coord_info}: "
            neighbor_strings = [f"{neighbor} (weight: {weight})" for neighbor, weight in neighbors]
            result += " -> ".join(neighbor_strings)
            result += "\n"
        
        return result
    
    def get_neighbors(self, node):
        """
        Get all neighbors of a given node.
        
        Args:
            node: The node ID to get neighbors for
            
        Returns:
            List of (neighbor, weight) tuples, or empty list if node doesn't exist
        """
        return self.adjacency_list.get(node, [])
    
    def get_all_nodes(self):
        """Get all node IDs in the graph."""
        return list(self.adjacency_list.keys())
    
    def get_node_coordinate(self, node):
        """
        Get the (x, y) coordinate of a specific node.
        
        Args:
            node: The node ID
            
        Returns:
            tuple: (x, y) coordinate or None if node doesn't exist
        """
        return self.node_coordinates.get(node, None)
    
    def validate_coordinates(self):
        """
        Validate that all nodes in the adjacency list have coordinates.
        
        Returns:
            bool: True if all nodes have coordinates, False otherwise
        """
        for node in self.adjacency_list.keys():
            if node not in self.node_coordinates:
                print(f"Warning: Node {node} missing coordinates")
                return False
        return True

# Test function to demonstrate the new functionality
def test_graph_functionality():
    """Test the enhanced Graph class functionality."""
    print("Testing Enhanced Graph Class")
    print("=" * 30)
    
    # Test with legacy format
    graph = Graph()
    
    # Create a simple test CSV content
    test_csv_content = """start_node,end_node,travel_time
A,B,5
B,A,5
A,C,3
C,A,3
B,D,4
D,B,4
C,D,1
D,C,1"""
    
    # Write test file
    with open('test_map.csv', 'w') as f:
        f.write(test_csv_content)
    
    # Load and test
    graph.load_map_data('test_map.csv')
    
    print(f"Nodes: {graph.get_all_nodes()}")
    print(f"Coordinates loaded: {len(graph.node_coordinates)}")
    
    # Test coordinate snapping
    test_point = (2.5, 2.5)
    nearest = graph.find_nearest_vertex(test_point)
    print(f"Point {test_point} snaps to vertex: {nearest}")
    
    # Test bounds
    bounds = graph.get_map_bounds()
    print(f"Map bounds: {bounds}")
    
    print("\nGraph structure:")
    print(graph)
    
    # Clean up
    import os
    if os.path.exists('test_map.csv'):
        os.remove('test_map.csv')

if __name__ == "__main__":
    test_graph_functionality()