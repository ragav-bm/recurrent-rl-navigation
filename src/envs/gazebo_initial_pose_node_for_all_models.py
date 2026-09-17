#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Twist
# Assuming these services are in a package named 'rl_training'
from rl_training.srv import ResetRobotPose, SetGoalAndObstaclePose 
import math
import tf_transformations

class GazeboInitialPose(Node):
    def __init__(self):
        super().__init__('gazebo_initialpose_service')

        # --- PARAMETERS ---
        # NOTE: These names must exactly match the model names in your Gazebo world.
        self.declare_parameter("robot_name", "burger")
        self.declare_parameter("obstacle1_name", "turtlebot3_dqn_obstacle1") 
        self.declare_parameter("obstacle2_name", "turtlebot3_dqn_obstacle2") 
        self.declare_parameter("goal_name", "goal_marker")    
        
        self.robot_name = self.get_parameter("robot_name").value
        self.obstacle1_name = self.get_parameter("obstacle1_name").value
        self.obstacle2_name = self.get_parameter("obstacle2_name").value
        self.goal_name = self.get_parameter("goal_name").value

        # --- PUBLISHER (AMCL Initial Pose) ---
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # --- CLIENTS for Gazebo services ---
        self.get_client = self.create_client(GetEntityState, '/get_entity_state')
        while not self.get_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/get_entity_state service not available, waiting...')

        self.set_client = self.create_client(SetEntityState, '/set_entity_state')
        while not self.set_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/set_entity_state service not available, waiting...')

        # --- SERVICES (Provided by this node) ---
        # 1. Service to trigger a robot reset
        self.reset_srv = self.create_service(ResetRobotPose, 'reset_robot_pose', self.reset_callback)
        
        # 2. Service to trigger setting the goal and obstacle poses
        self.set_entities_srv = self.create_service(
            SetGoalAndObstaclePose, 
            'set_goal_and_obstacle_pose', 
            self.set_entities_callback
        )

        # Timer to call initial pose once after startup
        self.timer = self.create_timer(2.0, self.call_service_once)

    def call_service_once(self):
        """Requests the robot's current pose from Gazebo."""
        req = GetEntityState.Request()
        req.name = self.robot_name
        req.reference_frame = "world"

        future = self.get_client.call_async(req)
        future.add_done_callback(self.initial_pose_callback)
        self.timer.cancel()

    def initial_pose_callback(self, future):
        """Receives robot pose and publishes it to /initialpose for AMCL."""
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed {e}")
            return

        if not resp.success:
            self.get_logger().error(f"Could not get state for {self.robot_name}")
            return

        self.publish_initial_pose(resp.state.pose)
        self.get_logger().info(f"Published initial pose for {self.robot_name}")

    def publish_initial_pose(self, pose):
        """Populates and publishes the PoseWithCovarianceStamped message."""
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.stamp = self.get_clock().now().to_msg()
        init_pose.header.frame_id = "map"
        init_pose.pose.pose = pose
        # Set typical covariance values for AMCL initialization
        init_pose.pose.covariance[0] = 0.25  # X covariance
        init_pose.pose.covariance[7] = 0.25  # Y covariance
        init_pose.pose.covariance[35] = math.radians(10) # Yaw covariance
        self.pub.publish(init_pose)

    def reset_callback(self, request, response):
        """Resets the robot's pose in Gazebo and AMCL."""
        # Convert yaw to quaternion
        quat = tf_transformations.quaternion_from_euler(0.0, 0.0, request.yaw)

        reset_pose = Pose()
        reset_pose.position.x = request.x
        reset_pose.position.y = request.y
        reset_pose.position.z = request.z
        reset_pose.orientation.x = quat[0]
        reset_pose.orientation.y = quat[1]
        reset_pose.orientation.z = quat[2]
        reset_pose.orientation.w = quat[3]

        # Create SetEntityState request for the ROBOT
        set_req = SetEntityState.Request()
        set_req.state.name = self.robot_name
        set_req.state.pose = reset_pose
        set_req.state.twist = Twist()
        set_req.state.reference_frame = "world"

        future = self.set_client.call_async(set_req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Robot {self.robot_name} reset in Gazebo"))

        # Update AMCL initialpose
        self.publish_initial_pose(reset_pose)

        response.success = True
        response.message = f"Robot {self.robot_name} reset successfully to x:{request.x}, y:{request.y}, z:{request.z}, yaw:{request.yaw}"
        return response

    def set_entities_callback(self, request, response):
        """Sets the poses of the Goal and the two Obstacles."""
        
        # Helper function to set the pose of a single entity
        def set_entity_pose(name, x, y, z=0.0):
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z # Use provided Z or default to 0.0

            set_req = SetEntityState.Request()
            set_req.state.name = name
            set_req.state.pose = pose
            set_req.state.twist = Twist()
            set_req.state.reference_frame = "world"
            
            future = self.set_client.call_async(set_req)
            # Log completion without waiting for service response to keep the main service quick
            future.add_done_callback(lambda f: self.get_logger().info(f"Entity '{name}' set to x:{x}, y:{y}"))
            return future

        # 1. Set GOAL POSE (using Z=0.05, based on your get_entity_state output)
        set_entity_pose(self.goal_name, request.goal_x, request.goal_y, z=0.1) 

        # 2. Set OBSTACLE 1 POSE (using Z=0.0, based on your get_entity_state output)
        # set_entity_pose(self.obstacle1_name, request.obstacle1_x, request.obstacle1_y, z=0.125)

        # 3. Set OBSTACLE 2 POSE (using Z=0.125, based on your get_entity_state output)
        # set_entity_pose(self.obstacle2_name, request.obstacle2_x, request.obstacle2_y, z=0.125)
        
        response.success = True
        response.message = f"Goal marker successfully set."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GazeboInitialPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()