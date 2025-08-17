

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
import os, webbrowser

class VisualizationEngine:
    """
    Complete visualization engine implementing Step 3 requirements:
    - PNG file generation instead of text output
    - Scatter plot of final car locations on city map
    - Metrics display using matplotlib.pyplot.text()
    - Optional charts (bar chart, histogram)
    """
    
    def __init__(self, simulation_data, graph, output_filename='simulation_summary.png'):
        """Initialize visualization engine with simulation results."""
        self.simulation_data = simulation_data
        self.graph = graph
        self.output_filename = output_filename
        
        # Extract data components
        self.cars = simulation_data.get('cars', {})
        self.riders = simulation_data.get('riders', {})
        self.completed_trips = simulation_data.get('completed_trips', [])
        self.metrics = simulation_data.get('metrics', {})
        
        print("INTELLIGENCE ANALYSIS: Visualization engine initialized")
        print(f"TARGET ACQUIRED: Output file: {self.output_filename}")
    
    def create_complete_visualization(self):
        """
        Create the complete analytical visualization implementing all Step 3 requirements.
        """
        print("BATTLE STATIONS: Generating complete analytical visualization")
        
        # Create figure with specified layout (Step 3 requirement)
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('RIDE-SHARING SIMULATION - TACTICAL ANALYSIS REPORT', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create main layout: large plot on left, metrics/charts on right
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])
        
        # Main scatter plot (left side - Step 3 requirement)
        main_ax = fig.add_subplot(gs[:, 0])
        self.create_city_map_visualization(main_ax)
        
        # Metrics display (top right - Step 3 requirement)
        metrics_ax = fig.add_subplot(gs[0, 1:])
        self._display_performance_metrics(metrics_ax)
        
        # Additional chart (bottom right - Step 3 enhancement)
        chart_ax = fig.add_subplot(gs[1, 1])
        self._create_trips_per_car_chart(chart_ax)
        
        # Wait time histogram (bottom right)
        hist_ax = fig.add_subplot(gs[1, 2])
        self._create_wait_time_histogram(hist_ax)
        
        # Add timestamp and simulation info
        self._add_simulation_info(fig)
        
        # Save as PNG (Step 3 requirement)
        plt.tight_layout()
        plt.savefig(self.output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MISSION ACCOMPLISHED: Visualization saved to {self.output_filename}")
        print("INTELLIGENCE PACKAGE: Complete analytical report generated")

        # ⬇️ Auto-open the saved PNG
        try:
            webbrowser.open(os.path.abspath(self.output_filename))
        except Exception as e:
            print(f"Could not open image automatically: {e}")
        
        return self.output_filename
    
   
    
    def create_simple_visualization(self):
      
        print("RECON MISSION: Generating simplified tactical overview")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('RIDE-SHARING SIMULATION - QUICK ANALYSIS', fontweight='bold')
        
        # Simple car location plot
        self._create_city_map_visualization(ax1)
        
        # Simple metrics display
        ax2.axis('off')
        ax2.set_title('KEY METRICS', fontweight='bold')
        
        # Display basic metrics
        y_pos = 0.9
        metrics_lines = [
            f"Duration: {self.metrics.get('simulation_time', 0):.1f}s",
            f"Fleet: {self.metrics.get('fleet_size', 0)} vehicles",
            f"Requests: {self.metrics.get('total_requests', 0)}",
            f"Completed: {self.metrics.get('completed_trips', 0)}",
            f"Success Rate: {self.metrics.get('completion_rate', 0):.1f}%",
            f"Avg Wait: {self.metrics.get('avg_wait_time', 0):.2f}s"
        ]
        
        for line in metrics_lines:
            ax2.text(0.1, y_pos, line, transform=ax2.transAxes, fontsize=12)
            y_pos -= 0.12
        
        plt.tight_layout()
        simple_filename = self.output_filename.replace('.png', '_simple.png')
        plt.savefig(simple_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"RECON COMPLETE: Simple visualization saved to {simple_filename}")

        # ⬇️ Auto-open the saved simple PNG
        try:
            webbrowser.open(os.path.abspath(simple_filename))
        except Exception as e:
            print(f"Could not open simple image automatically: {e}")
        
        return simple_filename

# (rest of the file with test_visualization_engine stays the same)
