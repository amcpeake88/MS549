# quadtree.py
import math

class Rectangle:
    """Helper class to represent a rectangular boundary."""
    def __init__(self, x, y, width, height):
        self.x = x          # x-coordinate of top-left corner
        self.y = y          # y-coordinate of top-left corner
        self.width = width  # width of rectangle
        self.height = height # height of rectangle
    
    def contains(self, point):
        """Check if a point (x, y) is inside this rectangle."""
        # Handle both tuple points (x, y) and objects with .location
        if hasattr(point, 'location'):
            px, py = point.location
        else:
            px, py = point
            
        # Exclude points on right/bottom edges to avoid double-counting
        return (self.x <= px < self.x + self.width and 
                self.y <= py < self.y + self.height)
    
    def distance_to_point(self, point):
        """Calculate minimum distance from point to this rectangle."""
        # Handle both tuple points and objects with .location
        if hasattr(point, 'location'):
            px, py = point.location
        else:
            px, py = point
        
        # Calculate distance to closest edge of rectangle
        dx = max(0, max(self.x - px, px - (self.x + self.width)))
        dy = max(0, max(self.y - py, py - (self.y + self.height)))
        return math.sqrt(dx*dx + dy*dy)

def distance_between_points(point1, point2):
    """Calculate Euclidean distance between two points."""
    # Handle both tuple points and objects with .location
    if hasattr(point1, 'location'):
        x1, y1 = point1.location
    else:
        x1, y1 = point1
        
    if hasattr(point2, 'location'):
        x2, y2 = point2.location
    else:
        x2, y2 = point2
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class QuadtreeNode:
    """Represents a rectangular region of the map."""
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary    # Rectangle object
        self.points = []           # List of points in this node
        self.capacity = capacity   # Max points before subdivision
        self.divided = False       # Has this node been subdivided?
        
        # Four child quadrants (initially None)
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
    
    def subdivide(self):
        """Create four child quadrants when node capacity is exceeded."""
        if self.divided:
            return  # Already subdivided
        
        # Calculate dimensions for quadrants
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.width / 2
        h = self.boundary.height / 2
        
        # Create four child quadrants
        nw_boundary = Rectangle(x, y, w, h)
        ne_boundary = Rectangle(x + w, y, w, h)
        sw_boundary = Rectangle(x, y + h, w, h)
        se_boundary = Rectangle(x + w, y + h, w, h)
        
        self.northwest = QuadtreeNode(nw_boundary, self.capacity)
        self.northeast = QuadtreeNode(ne_boundary, self.capacity)
        self.southwest = QuadtreeNode(sw_boundary, self.capacity)
        self.southeast = QuadtreeNode(se_boundary, self.capacity)
        
        self.divided = True
    
    def insert(self, point):
        """Insert a point into this node or its children."""
        # Step 1: If point is outside this node's boundary, ignore it
        if not self.boundary.contains(point):
            return False
        
        # Step 2: If node has capacity, add the point to its points list
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        
        # Step 3: Node is full - subdivide if not already divided
        if not self.divided:
            self.subdivide()
            
            # Distribute all existing points to appropriate child quadrants
            # Keep points in parent AND put them in children
            for existing_point in self.points:
                self.northwest.insert(existing_point)
                self.northeast.insert(existing_point)
                self.southwest.insert(existing_point)
                self.southeast.insert(existing_point)
        
        # Step 4: Try to insert the new point into appropriate child quadrant
        return (self.northwest.insert(point) or 
                self.northeast.insert(point) or 
                self.southwest.insert(point) or 
                self.southeast.insert(point))
    
    def _which_quadrant_contains(self, point):
        """Determine which quadrant contains the query point."""
        if not self.divided:
            return None
            
        if self.northwest.boundary.contains(point):
            return self.northwest
        elif self.northeast.boundary.contains(point):
            return self.northeast
        elif self.southwest.boundary.contains(point):
            return self.southwest
        elif self.southeast.boundary.contains(point):
            return self.southeast
        else:
            return None
    
    def find_nearest(self, query_point, best_point=None, min_distance=float('inf')):
        """Recursively find the nearest point to query_point."""
        # Pruning: if this node's boundary is farther than current best, skip it
        boundary_distance = self.boundary.distance_to_point(query_point)
        if boundary_distance >= min_distance:
            return best_point, min_distance
        
        # Check all points in this node
        for point in self.points:
            distance = distance_between_points(query_point, point)
            if distance < min_distance:
                min_distance = distance
                best_point = point
        
        # If this node has children, search them with priority
        if self.divided:
            # Priority 1: Search the quadrant that contains the query point first
            priority_quadrant = self._which_quadrant_contains(query_point)
            other_quadrants = []
            
            for quadrant in [self.northwest, self.northeast, self.southwest, self.southeast]:
                if quadrant == priority_quadrant:
                    # Search priority quadrant first
                    best_point, min_distance = quadrant.find_nearest(query_point, best_point, min_distance)
                else:
                    other_quadrants.append(quadrant)
            
            # Priority 2: Search other quadrants if they cannot be pruned
            for quadrant in other_quadrants:
                best_point, min_distance = quadrant.find_nearest(query_point, best_point, min_distance)
        
        return best_point, min_distance

class Quadtree:
    """Main Quadtree class that users interact with."""
    def __init__(self, boundary):
        self.boundary = boundary              # Rectangle for entire map
        self.root = QuadtreeNode(boundary)    # Root node
    
    def insert(self, point):
        """Insert a point into the quadtree."""
        return self.root.insert(point)
    
    def find_nearest(self, query_point):
        """Find the nearest point to the query point."""
        best_point, min_distance = self.root.find_nearest(query_point)
        return best_point