

import math
from collections import defaultdict

class Rectangle:
    """Helper class to represent a rectangular boundary."""
    def __init__(self, x, y, width, height):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
    
    def contains(self, point):
        """Check if a point (x, y) is inside this rectangle."""
        try:
            if hasattr(point, 'location'):
                px, py = point.location
            elif isinstance(point, tuple) and len(point) == 2:
                if isinstance(point[0], (int, float)):
                    px, py = float(point[0]), float(point[1])
                elif isinstance(point[0], tuple) and len(point[0]) == 2:
                    px, py = float(point[0][0]), float(point[0][1])
                else:
                    return False
            elif hasattr(point, 'coordinates'):
                px, py = point.coordinates
                px, py = float(px), float(py)
            else:
                return False
            
            return (self.x <= px < self.x + self.width and 
                    self.y <= py < self.y + self.height)
        except (ValueError, TypeError, AttributeError):
            return False
    
    def distance_to_point(self, point):
        """Calculate minimum distance from point to this rectangle."""
        try:
            if hasattr(point, 'location'):
                px, py = point.location
            elif isinstance(point, tuple) and len(point) == 2:
                if isinstance(point[0], (int, float)):
                    px, py = float(point[0]), float(point[1])
                elif isinstance(point[0], tuple) and len(point[0]) == 2:
                    px, py = float(point[0][0]), float(point[0][1])
                else:
                    return float('inf')
            elif hasattr(point, 'coordinates'):
                px, py = point.coordinates
                px, py = float(px), float(py)
            else:
                return float('inf')
            
            dx = max(0, max(self.x - px, px - (self.x + self.width)))
            dy = max(0, max(self.y - py, py - (self.y + self.height)))
            distance = math.sqrt(dx*dx + dy*dy)
            return distance if not math.isnan(distance) else float('inf')
        except (ValueError, TypeError, AttributeError):
            return float('inf')

def distance_between_points(point1, point2):
    """Calculate Euclidean distance between two points."""
    
    def extract_coords_safe(point):
        try:
            if hasattr(point, 'location'):
                x, y = point.location
                return float(x), float(y)
            elif isinstance(point, tuple) and len(point) == 2:
                if isinstance(point[0], (int, float)):
                    return float(point[0]), float(point[1])
                elif isinstance(point[0], tuple) and len(point[0]) == 2:
                    return float(point[0][0]), float(point[0][1])
            elif hasattr(point, 'coordinates'):
                x, y = point.coordinates
                return float(x), float(y)
            return None, None
        except (ValueError, TypeError, AttributeError):
            return None, None
    
    x1, y1 = extract_coords_safe(point1)
    x2, y2 = extract_coords_safe(point2)
    
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return float('inf')
    
    try:
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance if not math.isnan(distance) else float('inf')
    except (ValueError, OverflowError):
        return float('inf')

class EnhancedQuadtreeNode:
    """Enhanced quadtree node with car management capabilities."""
    
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.points = []           # Store point coordinates
        self.point_objects = {}    # Map coordinates to Car objects
        self.capacity = capacity
        self.divided = False
        
        # Four child quadrants
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
    
    def subdivide(self):
        """Create four child quadrants when capacity is exceeded."""
        if self.divided:
            return
        
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.width / 2
        h = self.boundary.height / 2
        
        nw_boundary = Rectangle(x, y, w, h)
        ne_boundary = Rectangle(x + w, y, w, h)
        sw_boundary = Rectangle(x, y + h, w, h)
        se_boundary = Rectangle(x + w, y + h, w, h)
        
        self.northwest = EnhancedQuadtreeNode(nw_boundary, self.capacity)
        self.northeast = EnhancedQuadtreeNode(ne_boundary, self.capacity)
        self.southwest = EnhancedQuadtreeNode(sw_boundary, self.capacity)
        self.southeast = EnhancedQuadtreeNode(se_boundary, self.capacity)
        
        self.divided = True
    
    def insert(self, point, car_object=None):
        """Insert a point with optional associated car object."""
        if not self.boundary.contains(point):
            return False
        
        if len(self.points) < self.capacity:
            self.points.append(point)
            if car_object:
                self.point_objects[point] = car_object
            return True
        
        if not self.divided:
            self.subdivide()
            
            # Redistribute existing points
            for existing_point in self.points:
                existing_object = self.point_objects.get(existing_point)
                (self.northwest.insert(existing_point, existing_object) or 
                 self.northeast.insert(existing_point, existing_object) or 
                 self.southwest.insert(existing_point, existing_object) or 
                 self.southeast.insert(existing_point, existing_object))
            
            self.points = []
            self.point_objects = {}
        
        return (self.northwest.insert(point, car_object) or 
                self.northeast.insert(point, car_object) or 
                self.southwest.insert(point, car_object) or 
                self.southeast.insert(point, car_object))
    
    def remove(self, point):
        """Remove a specific point from the quadtree."""
        # Check if point is in this node
        if point in self.points:
            self.points.remove(point)
            if point in self.point_objects:
                del self.point_objects[point]
            return True
        
        # If subdivided, check children
        if self.divided:
            return (self.northwest.remove(point) or 
                   self.northeast.remove(point) or 
                   self.southwest.remove(point) or 
                   self.southeast.remove(point))
        
        return False
    
    def find_k_nearest(self, query_point, k=5, candidates=None):
        """Find k nearest points to query point."""
        if candidates is None:
            candidates = []
        
        # Add points from this node to candidates
        for point in self.points:
            distance = distance_between_points(query_point, point)
            if distance != float('inf') and not math.isnan(distance):
                car_object = self.point_objects.get(point)
                candidates.append((distance, point, car_object))
        
        # If subdivided, search children
        if self.divided:
            # Determine which quadrant contains query point for priority search
            priority_quadrant = None
            if self.boundary.contains(query_point):
                if self.northwest.boundary.contains(query_point):
                    priority_quadrant = self.northwest
                elif self.northeast.boundary.contains(query_point):
                    priority_quadrant = self.northeast
                elif self.southwest.boundary.contains(query_point):
                    priority_quadrant = self.southwest
                elif self.southeast.boundary.contains(query_point):
                    priority_quadrant = self.southeast
            
            # Search priority quadrant first
            if priority_quadrant:
                priority_quadrant.find_k_nearest(query_point, k, candidates)
            
            # Search other quadrants if needed
            for quadrant in [self.northwest, self.northeast, self.southwest, self.southeast]:
                if quadrant != priority_quadrant:
                    # Prune if quadrant is too far
                    if len(candidates) >= k:
                        candidates.sort(key=lambda x: x[0])
                        kth_distance = candidates[k-1][0]
                        if quadrant.boundary.distance_to_point(query_point) > kth_distance:
                            continue
                    
                    quadrant.find_k_nearest(query_point, k, candidates)
        
        return candidates

class EnhancedQuadtree:
    """Enhanced Quadtree with car management and k-nearest search."""
    
    def __init__(self, boundary):
        self.boundary = boundary
        self.root = EnhancedQuadtreeNode(boundary)
        self.car_locations = {}  # Map car_id to coordinates for tracking
    
    def insert_car(self, car):
        """Insert a car into the quadtree using its coordinates."""
        if hasattr(car, 'coordinates') and car.coordinates:
            success = self.root.insert(car.coordinates, car)
            if success:
                self.car_locations[car.id] = car.coordinates
                print(f"SPATIAL INDEX: {car.id} inserted at {car.coordinates}")
            return success
        return False
    
    def remove_car(self, car):
        """Remove a car from the quadtree."""
        if car.id in self.car_locations:
            old_coords = self.car_locations[car.id]
            success = self.root.remove(old_coords)
            if success:
                del self.car_locations[car.id]
                print(f"SPATIAL INDEX: {car.id} removed from {old_coords}")
            return success
        return False
    
    def update_car_location(self, car, new_coordinates):
        """Update a car's location in the quadtree."""
        # Remove from old location
        self.remove_car(car)
        
        # Update car coordinates
        car.coordinates = new_coordinates
        
        # Insert at new location
        return self.insert_car(car)
    
    def find_k_nearest_cars(self, query_point, k=5, status_filter="available"):
        """
        Find k nearest cars to a query point.
        This is the enhanced method required for Step 2.
        """
        candidates = self.root.find_k_nearest(query_point, k * 2)  # Get extra candidates
        
        # Filter by car status if specified
        if status_filter:
            filtered_candidates = []
            for distance, point, car_object in candidates:
                if car_object and hasattr(car_object, 'status') and car_object.status == status_filter:
                    filtered_candidates.append((distance, point, car_object))
            candidates = filtered_candidates
        
        # Sort by distance and return top k cars
        candidates.sort(key=lambda x: x[0])
        result_cars = []
        
        for distance, point, car_object in candidates[:k]:
            if car_object:
                result_cars.append(car_object)
        
        print(f"TACTICAL SCAN: Found {len(result_cars)} available units within search radius")
        return result_cars
    
    def insert(self, point):
        """Legacy insert method for backward compatibility."""
        return self.root.insert(point)
    
    def find_nearest(self, query_point):
        """Legacy find_nearest method for backward compatibility."""
        candidates = self.root.find_k_nearest(query_point, 1)
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]  # Return the point
        return None
    
    def get_all_cars(self):
        """Get all cars currently in the quadtree."""
        all_cars = []
        
        def collect_cars(node):
            if node:
                for point in node.points:
                    car = node.point_objects.get(point)
                    if car:
                        all_cars.append(car)
                
                if node.divided:
                    collect_cars(node.northwest)
                    collect_cars(node.northeast)
                    collect_cars(node.southwest)
                    collect_cars(node.southeast)
        
        collect_cars(self.root)
        return all_cars
    
    def get_statistics(self):
        """Get quadtree statistics for debugging."""
        def count_nodes(node):
            if not node:
                return 0, 0, 0
            
            node_count = 1
            point_count = len(node.points)
            max_depth = 0
            
            if node.divided:
                for child in [node.northwest, node.northeast, node.southwest, node.southeast]:
                    child_nodes, child_points, child_depth = count_nodes(child)
                    node_count += child_nodes
                    point_count += child_points
                    max_depth = max(max_depth, child_depth + 1)
            
            return node_count, point_count, max_depth
        
        nodes, points, depth = count_nodes(self.root)
        return {
            'total_nodes': nodes,
            'total_points': points,
            'max_depth': depth,
            'tracked_cars': len(self.car_locations)
        }

# Test functions for enhanced quadtree
def test_enhanced_quadtree_car_management():
    """Test enhanced quadtree car management features."""
    print("BATTLE DRILL: Testing Enhanced Quadtree Car Management")
    print("=" * 55)
    
    # Create enhanced quadtree
    boundary = Rectangle(0, 0, 100, 100)
    quadtree = EnhancedQuadtree(boundary)
    
    # Create mock car class for testing
    class MockCar:
        def __init__(self, car_id, x, y, status="available"):
            self.id = car_id
            self.coordinates = (x, y)
            self.status = status
    
    # Test 1: Insert cars
    print("\nTEST 1: Car insertion")
    cars = [
        MockCar("ALPHA", 10, 10),
        MockCar("BRAVO", 30, 20),
        MockCar("CHARLIE", 50, 50),
        MockCar("DELTA", 80, 80, "busy"),
        MockCar("ECHO", 15, 25)
    ]
    
    for car in cars:
        success = quadtree.insert_car(car)
        print(f"  {car.id}: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 2: K-nearest search
    print("\nTEST 2: K-nearest car search")
    query_point = (20, 20)
    nearest_cars = quadtree.find_k_nearest_cars(query_point, k=3, status_filter="available")
    
    print(f"Query point: {query_point}")
    for i, car in enumerate(nearest_cars):
        distance = math.sqrt((car.coordinates[0] - query_point[0])**2 + 
                           (car.coordinates[1] - query_point[1])**2)
        print(f"  {i+1}. {car.id} at {car.coordinates} (distance: {distance:.2f})")
    
    # Test 3: Car removal
    print("\nTEST 3: Car removal")
    car_to_remove = cars[1]  # BRAVO
    success = quadtree.remove_car(car_to_remove)
    print(f"Removed {car_to_remove.id}: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 4: Location update
    print("\nTEST 4: Car location update")
    car_to_update = cars[0]  # ALPHA
    new_location = (90, 90)
    success = quadtree.update_car_location(car_to_update, new_location)
    print(f"Updated {car_to_update.id} to {new_location}: {'SUCCESS' if success else 'FAILED'}")
    
    # Test 5: Statistics
    print("\nTEST 5: Quadtree statistics")
    stats = quadtree.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nBATTLE DRILL COMPLETE: Enhanced Quadtree operational")
    return True

if __name__ == "__main__":
    test_enhanced_quadtree_car_management()