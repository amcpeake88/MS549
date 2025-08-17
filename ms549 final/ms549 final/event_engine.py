

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