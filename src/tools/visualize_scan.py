#!/usr/bin/env python3

# --- 1. CRITICAL: Set the backend BEFORE importing anything else ---
import matplotlib
matplotlib.use('WebAgg') 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants from your environment logic
LIDAR_SAMPLES = 180
MAX_RANGE = 3.5 
MIN_RANGE = 0.12

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        self.subscription_raw = self.create_subscription(
            LaserScan, '/scan', self.raw_scan_callback, 10)
        self.subscription_obs = self.create_subscription(
            Float32MultiArray, '/rl/processed_obs', self.obs_callback, 10)
        
        self.raw_ranges = np.zeros(LIDAR_SAMPLES)
        self.processed_ranges = np.zeros(LIDAR_SAMPLES)
        self.noisy_ranges = np.zeros(LIDAR_SAMPLES)
        self.angles = np.linspace(0, 2 * np.pi, LIDAR_SAMPLES)

        # Setup Plot
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
        self.fig.canvas.manager.set_window_title('Lidar Data Verification')

    def process_logic(self, msg):
        # This logic is for the "before" visualization
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=MIN_RANGE)
        
        indices = np.linspace(0, len(ranges)-1, LIDAR_SAMPLES, dtype=int)
        sampled_ranges = ranges[indices]
        
        # Raw data in meters
        raw_viz = np.clip(sampled_ranges, MIN_RANGE, MAX_RANGE)
        
        # Normalized data (what the env starts with before adding noise)
        normalized = np.clip(sampled_ranges, MIN_RANGE, MAX_RANGE) / MAX_RANGE
        
        return raw_viz, normalized

    def raw_scan_callback(self, msg):
        self.raw_ranges, self.processed_ranges = self.process_logic(msg)

    def obs_callback(self, msg):
        # This new callback gets the noisy data from the environment
        # The first LIDAR_SAMPLES elements are the lidar data
        self.noisy_ranges = np.array(msg.data[:LIDAR_SAMPLES])

    def update_plot(self, frame):
        # Plot 1: Raw Scan Plot (in meters)
        self.ax1.clear()
        self.ax1.set_title(f"Raw Scan (Meters)\nFrom /scan topic")
        self.ax1.plot(self.angles, self.raw_ranges, color='blue')
        self.ax1.set_ylim(0, MAX_RANGE)
        self.ax1.set_theta_zero_location('N')
        self.ax1.set_theta_direction(-1)

        # Plot 2: Normalized Scan (before noise)
        self.ax2.clear()
        self.ax2.set_title("Normalized Scan (0-1)\nEnv Input (Pre-Noise)")
        self.ax2.fill(self.angles, self.processed_ranges, color='green', alpha=0.6)
        self.ax2.set_ylim(0, 1.1)
        self.ax2.set_theta_zero_location('N')
        self.ax2.set_theta_direction(-1)

        # Plot 3: Noisy Scan (from environment)
        self.ax3.clear()
        self.ax3.set_title("Noisy Scan (0-1)\nAgent Input (Post-Noise)")
        self.ax3.fill(self.angles, self.noisy_ranges, color='red', alpha=0.6)
        self.ax3.set_ylim(0, 1.1)
        self.ax3.set_theta_zero_location('N')
        self.ax3.set_theta_direction(-1)

def main():
    rclpy.init()
    node = LidarVisualizer()

    # --- 2. CRITICAL: Assign to a variable (anim) so it isn't deleted ---
    anim = FuncAnimation(node.fig, node.update_plot, interval=100, cache_frame_data=False)
    
    # Run ROS 2 in a separate thread or use a timer to allow matplotlib to loop
    import threading
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    print("\n" + "="*50)
    print("Starting Matplotlib Web Server...")
    print("Look for the URL below (usually http://127.0.0.1:8988).")
    print("Ctrl+Click the link to open the live plot in your browser!")
    print("="*50 + "\n")
    
    # --- 3. Run plt.show() which will now launch the web server ---
    plt.show()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()