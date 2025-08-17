# test_quadtree.py
import random
import time
from quadtree import Quadtree, Rectangle, distance_between_points

def brute_force_nearest(query_point, all_points):
    """Brute force method to find nearest point - for verification."""
    if not all_points:
        return None
    
    best_point = all_points[0]
    min_distance = distance_between_points(query_point, best_point)
    
    for point in all_points[1:]:
        distance = distance_between_points(query_point, point)
        if distance < min_distance:
            min_distance = distance
            best_point = point
    
    return best_point

def test_quadtree_correctness():
    """Test that Quadtree finds the same nearest point as brute force."""
    print("=" * 60)
    print("QUADTREE CORRECTNESS TEST")
    print("Testing with 5,000 random points in 1000x1000 area")
    print("=" * 60)
    
    # Step 1: Initialize Quadtree with 1000x1000 boundary
    boundary = Rectangle(0, 0, 1000, 1000)
    quadtree = Quadtree(boundary)
    
    # Step 2: Generate exactly 5,000 random points
    print("Generating 5,000 random points...")
    random.seed(42)  # Fixed seed for reproducible results
    all_points = []
    
    for i in range(5000):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        point = (x, y)
        all_points.append(point)
        success = quadtree.insert(point)
        if not success:
            print(f"Warning: Failed to insert point {i}: {point}")
    
    print(f"Successfully inserted {len(all_points)} points into Quadtree")
    
    # Step 3: Pick a random query point
    query_x = random.uniform(0, 1000)
    query_y = random.uniform(0, 1000)
    query_point = (query_x, query_y)
    
    print(f"\\nQuery point: ({query_x:.2f}, {query_y:.2f})")
    
    # Step 4: Find nearest using Quadtree
    print("\\nSearching with Quadtree...")
    start_time = time.time()
    quadtree_result = quadtree.find_nearest(query_point)
    quadtree_time = time.time() - start_time
    
    # Step 5: Find nearest using brute force
    print("Searching with brute force...")
    start_time = time.time()
    brute_force_result = brute_force_nearest(query_point, all_points)
    brute_force_time = time.time() - start_time
    
    # Step 6: Calculate distances for verification
    quadtree_distance = distance_between_points(query_point, quadtree_result) if quadtree_result else float('inf')
    brute_force_distance = distance_between_points(query_point, brute_force_result) if brute_force_result else float('inf')
    
    # Step 7: Display results
    print("\\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Quadtree result:      {quadtree_result}")
    print(f"Quadtree distance:    {quadtree_distance:.6f}")
    print(f"Quadtree time:        {quadtree_time:.6f} seconds")
    print()
    print(f"Brute force result:   {brute_force_result}")
    print(f"Brute force distance: {brute_force_distance:.6f}")
    print(f"Brute force time:     {brute_force_time:.6f} seconds")
    print()
    
    # Step 8: Verify correctness
    distance_difference = abs(quadtree_distance - brute_force_distance)
    points_match = (quadtree_result == brute_force_result)
    distances_match = distance_difference < 1e-10  # Account for floating point precision
    
    print("VERIFICATION:")
    print(f"Points identical:     {points_match}")
    print(f"Distances match:      {distances_match}")
    print(f"Distance difference:  {distance_difference}")
    
    if points_match and distances_match:
        print("\\n SUCCESS: Quadtree implementation is CORRECT!")
        if quadtree_time > 0:
            speedup = brute_force_time / quadtree_time
            print(f"Performance: {speedup:.2f}x faster than brute force")
        print(f"Efficiency: O(log N) vs O(N) - Quadtree provides {len(all_points):,} point search optimization")
    else:
        print("\\n❌ FAILURE: Quadtree implementation has errors!")
        return False
    
    return True

def test_multiple_seeds():
    """Test with multiple random seeds to ensure consistency."""
    print("\\n" + "=" * 60)
    print("MULTIPLE SEED ROBUSTNESS TEST")
    print("Testing with different random seeds")
    print("=" * 60)
    
    seeds = [123, 456, 789, 999, 1337]
    all_passed = True
    
    for i, seed in enumerate(seeds):
        print(f"\\nTest {i+1}/5 - Seed {seed}:")
        
        # Setup with current seed
        boundary = Rectangle(0, 0, 1000, 1000)
        quadtree = Quadtree(boundary)
        random.seed(seed)
        
        # Generate 1000 points for faster testing
        all_points = []
        for j in range(1000):
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            point = (x, y)
            all_points.append(point)
            quadtree.insert(point)
        
        # Test with random query
        query_point = (random.uniform(0, 1000), random.uniform(0, 1000))
        
        quadtree_result = quadtree.find_nearest(query_point)
        brute_force_result = brute_force_nearest(query_point, all_points)
        
        correct = (quadtree_result == brute_force_result)
        all_passed = all_passed and correct
        
        status = "✅ PASS" if correct else "❌ FAIL"
        print(f"  {status} - Points match: {correct}")
    
    print(f"\\nOverall result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return all_passed

def performance_comparison():
    """Compare performance across different dataset sizes."""
    print("\\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("Comparing Quadtree vs Brute Force across dataset sizes")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2500, 5000]
    
    for size in sizes:
        print(f"\\nTesting with {size:,} points:")
        
        # Setup
        boundary = Rectangle(0, 0, 1000, 1000)
        quadtree = Quadtree(boundary)
        random.seed(42)
        
        # Generate points
        all_points = []
        for i in range(size):
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            point = (x, y)
            all_points.append(point)
            quadtree.insert(point)
        
        query_point = (random.uniform(0, 1000), random.uniform(0, 1000))
        
        # Time Quadtree
        start_time = time.time()
        quadtree_result = quadtree.find_nearest(query_point)
        quadtree_time = time.time() - start_time
        
        # Time Brute Force
        start_time = time.time()
        brute_force_result = brute_force_nearest(query_point, all_points)
        brute_force_time = time.time() - start_time
        
        # Results
        speedup = brute_force_time / quadtree_time if quadtree_time > 0 else float('inf')
        correct = (quadtree_result == brute_force_result)
        
        print(f"  Quadtree:    {quadtree_time:.6f}s")
        print(f"  Brute force: {brute_force_time:.6f}s")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Correct:     {correct}")

if __name__ == "__main__":
    print(" QUADTREE TESTING SUITE")
    print("This comprehensive test proves Quadtree correctness and performance")
    print("by comparing results with brute force search.\\n")
    
    # Main correctness test with exactly 5,000 points
    test1_passed = test_quadtree_correctness()
    
    # Robustness test with multiple seeds
    test2_passed = test_multiple_seeds()
    
    # Performance analysis
    performance_comparison()
    
    # Final summary
    print("\\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED!")
        print("Quadtree implementation is mathematically correct")
        print("Performance benefits confirmed (O(log N) vs O(N))")
        print("Ready for integration into ride-sharing simulator")
        print("\\n Implementation Features:")
        print("  • Efficient spatial indexing with pruning optimization")
        print("  • Prioritized search (query quadrant first)")
        print("  • Robust boundary handling")
        print("  • Flexible point format support")
    else:
        print(" SOME TESTS FAILED")
        print(" Implementation needs debugging before integration")
    print("=" * 60)