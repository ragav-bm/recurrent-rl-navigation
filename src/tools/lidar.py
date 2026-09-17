import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarVerifier(Node):
    def __init__(self):
        super().__init__('lidar_verifier')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        print("--- LIDAR Index Verifier Started ---")
        print("Note: Index 0 is assumed to be FRONT based on your metadata (angle_min=0).")
        print("Move your hand around the robot to see which values change.\n")

    def listener_callback(self, msg):
        ranges = np.array(msg.ranges)
        num_beams = len(ranges)
        
        # Based on your metadata (0 to 6.28 radians):
        idx_front = 0
        idx_left  = int(num_beams * 0.25)  # 90 degrees
        idx_back  = int(num_beams * 0.50)  # 180 degrees
        idx_right = int(num_beams * 0.75)  # 270 degrees

        # Get values (using a small window of 3 beams for stability)
        def get_dist(idx):
            # Use modulo to handle wrap around for index 0
            indices = [(idx + i) % num_beams for i in range(-1, 2)]
            vals = [ranges[i] for i in indices if not np.isinf(ranges[i]) and not np.isnan(ranges[i])]
            return np.mean(vals) if vals else float('inf')

        f_dist = get_dist(idx_front)
        l_dist = get_dist(idx_left)
        b_dist = get_dist(idx_back)
        r_dist = get_dist(idx_right)

        # Print output to terminal
        print(f"FRONT (idx {idx_front}): {f_dist:.2f}m | "
              f"LEFT (idx {idx_left}): {l_dist:.2f}m | "
              f"BACK (idx {idx_back}): {b_dist:.2f}m | "
              f"RIGHT (idx {idx_right}): {r_dist:.2f}m", end='\r')

def main(args=None):
    rclpy.init(args=args)
    node = LidarVerifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
