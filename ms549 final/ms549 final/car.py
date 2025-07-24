import heapq

class Car:
    def __init__(self, car_id, location):
        """
        Initialize a Car object.
        
        Args:
            car_id (str): A unique identifier for the car (e.g., "CAR001")
            location: The car's initial location (node ID for pathfinding)
        """
        self.id = car_id
        self.location = location
        self.status = "available"
        self.destination = None
        
        # New attributes for route planning
        self.route = None
        self.route_time = None
    
    def calculate_route(self, destination, graph):
        """
        Calculate the shortest route from current location to destination using Dijkstra's algorithm.
        
        Args:
            destination: The destination node ID
            graph: Graph object containing the map data
            
        Returns:
            bool: True if route found, False if no route exists
        """
        # Use car's current location as start node
        start_node = self.location
        end_node = destination
        
        # Check if start and end nodes exist in the graph
        if start_node not in graph.adjacency_list or end_node not in graph.adjacency_list:
            self.route = None
            self.route_time = float('inf')
            return False
        
        # Initialize distances and predecessors
        distances = {}
        predecessors = {}
        visited = set()
        
        # Set all distances to infinity initially
        for node in graph.adjacency_list:
            distances[node] = float('inf')
            predecessors[node] = None
        
        # Distance to start node is 0
        distances[start_node] = 0
        
        # Priority queue (min-heap): (distance, node)
        priority_queue = [(0, start_node)]
        
        while priority_queue:
            # Get the node with minimum distance
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # Skip if we've already visited this node
            if current_node in visited:
                continue
                
            # Mark current node as visited
            visited.add(current_node)
            
            # If we reached the destination, we can stop
            if current_node == end_node:
                break
            
            # Skip if current distance is greater than recorded distance
            if current_distance > distances[current_node]:
                continue
            
            # Check all neighbors of current node
            neighbors = graph.get_neighbors(current_node)
            for neighbor, weight in neighbors:
                # Skip if neighbor already visited
                if neighbor in visited:
                    continue
                    
                # Calculate new distance through current node
                new_distance = distances[current_node] + weight
                
                # If we found a shorter path to neighbor
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    # Add to priority queue
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        # Reconstruct path from predecessors
        if distances[end_node] == float('inf'):
            # No path found
            self.route = None
            self.route_time = float('inf')
            return False
        
        # Build path by following predecessors backwards
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        # Reverse path to get start -> end order
        path.reverse()
        
        # Store results in car attributes
        self.route = path
        self.route_time = distances[end_node]
        self.destination = destination
        
        return True
    
    def __str__(self):
        """
        Return a formatted string summarizing the car's status.
        
        Returns:
            str: Formatted string with car ID, location, and status
        """
        route_info = ""
        if self.route is not None:
            route_str = " -> ".join(self.route)
            route_info = f" | Route: {route_str} (time: {self.route_time})"
        
        return f"Car {self.id} at {self.location} - Status: {self.status}{route_info}"