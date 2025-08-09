Advanced Ride-Sharing Simulation System with Spatial Optimization
Purpose/Design
This project implements a comprehensive ride-sharing simulation system featuring advanced pathfinding algorithms, efficient spatial data structures, and optimized vehicle dispatch capabilities. The system integrates Dijkstra's algorithm for route optimization with a Quadtree spatial indexing structure that achieves O(log N) nearest-neighbor search performance compared to O(N) brute force approaches, enabling efficient vehicle-passenger matching in large-scale transportation networks.

How to Run
Complete System Validation: Set test_complete.py as the startup file in Visual Studio by right-clicking in Solution Explorer and selecting "Set as Startup File". Run using F5 or Ctrl+F5, or click the green Start button. Alternatively, execute python test_complete.py in command line or Python Interactive.

Standalone Quadtree Testing: Execute python test_quadtree.py in command line, or set as startup file in Visual Studio and run using F5 or the green Start button. This runs dedicated Quadtree verification with 5,000 random points and brute force comparison.

Dependencies
Python Standard Library Only: heapq, csv, random, time, math
No External Libraries Required: The system uses only Python's built-in functionality for maximum compatibility and ease of deployment.
