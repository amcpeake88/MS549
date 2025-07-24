Car Class (car.py)
Project Title: Vehicle Management System with Integrated Pathfinding
Purpose/Design: This module implements the Car class representing individual vehicles in the ride-sharing fleet. The class integrates Dijkstra's algorithm for optimal route calculation and maintains vehicle status, location, and route information. The design approach uses object-oriented principles with methods for route planning and status management.
How to Run: Import the Car class in your Python script or test file. Create a Car instance with car_id and initial location, then use the calculate_route method with a destination and graph object. Example: car = Car("TestCar", 'A') followed by car.calculate_route('G', graph).
Dependencies: heapq (Python standard library)

Rider Class (rider.py)
Project Title: Passenger Request Management System
Purpose/Design: This module implements the Rider class representing customers requesting ride services. The class tracks passenger information including pickup location, destination, and current service status throughout the ride lifecycle. The design approach emphasizes simple attribute management with clear status tracking capabilities.
How to Run: Import the Rider class and create instances with rider_id, start_location, and destination parameters. Example: rider = Rider("PASSENGER_001", 'A', 'D'). Use the string representation to display current rider status.
Dependencies: None - This module uses only Python's built-in functionality.

Graph Class (graph_basic.py)
Project Title: City Infrastructure Network Representation
Purpose/Design: This module implements the Graph class for representing city transportation networks using adjacency list data structure. The class supports CSV file loading for external city data and provides methods for network traversal and neighbor lookup. The design approach optimizes for sparse networks typical in urban environments.
How to Run: Create a Graph instance and load city data using load_from_file method with a CSV filename. Example: graph = Graph() followed by graph.load_from_file('map.csv'). Use get_neighbors and get_all_nodes methods for network queries.
Dependencies: csv (Python standard library)

Pathfinding Module (pathfinding.py)
Project Title: Dijkstra's Shortest Path Algorithm Implementation
Purpose/Design: This module implements the standalone find_shortest_path function using Dijkstra's algorithm with min-heap optimization. The function calculates optimal routes between any two nodes in a weighted graph with predecessor tracking for path reconstruction. The design approach prioritizes algorithmic efficiency and clear path representation.
How to Run: Import the find_shortest_path function and call it with graph, start_node, and end_node parameters. Example: path, distance = find_shortest_path(graph, 'A', 'D'). The function returns a tuple containing the path as a list and total travel time.
Dependencies: heapq (Python standard library)

Simulation Control System (simulation.py)
Project Title: Central Ride-Sharing Coordination System
Purpose/Design: This module implements the Simulation class serving as the central controller for the entire ride-sharing operation. The class manages collections of vehicles and passengers while integrating with city infrastructure mapping. The design approach provides centralized coordination with automated map loading capabilities.
How to Run: Create a Simulation instance with optional map filename parameter. Example: sim = Simulation('map.csv'). Add vehicles and passengers to the cars and riders dictionaries using their respective IDs as keys. Use display_map method to view loaded infrastructure.
Dependencies: graph_basic module (local import)

Map Data File (map.csv)
Project Title: City Transportation Network Data
Purpose/Design: This CSV file contains the city transportation network data with nodes representing intersections and edges representing roads with travel times. The data format uses three columns: start_node, end_node, and travel_time for directed graph representation. The design approach supports bidirectional roads through separate entries for each direction.
How to Run: Place the map.csv file in the same directory as other project files. The file is automatically loaded by the Graph class load_from_file method. No direct execution required as this is a data file.
Dependencies: None - Standard CSV format readable by any CSV parser.

Comprehensive Test Suite (test_complete.py)
Project Title: Advanced System Validation Protocol
Purpose/Design: This module implements a comprehensive testing suite that validates all system components through a four-phase protocol. The test suite covers core entity validation, infrastructure mapping, pathfinding algorithms, and full system integration. The design approach ensures thorough verification of all functionality with clear progress reporting.
How to Run: Set test_complete.py as the startup file in Visual Studio by right-clicking in Solution Explorer and selecting "Set as Startup File". Run using F5 or Ctrl+F5, or click the green Start button. Alternatively, execute in Python Interactive by right-clicking in the code window.
Dependencies: csv, car, rider, simulation, graph_basic modules (standard library and local imports)
