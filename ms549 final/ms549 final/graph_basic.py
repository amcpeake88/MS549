import csv

class Graph:
    def __init__(self):
        """Initialize an empty adjacency list."""
        self.adjacency_list = {}
    
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
        Load graph data from a CSV file.
        
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
            result += f"Node {node}: "
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