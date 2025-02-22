#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy


class TopicSubscriber(Node):
    def __init__(self):
        super().__init__('topic_subscriber_node')
        
        # /input_image
        self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10
        )
        
        # /input_query
        self.create_subscription(
            String,
            'input_query',
            self.query_callback,
            10
        )
        
        # /flip_image
        self.create_subscription(
            Bool,
            'flip_image',
            self.flip_callback,
            10
        )
        
        # /output
        self.create_subscription(
            String,
            'output',
            self.output_callback,
            10
        )
        
        self.get_logger().info("subscription started.")
    
    def image_callback(self, msg: Image):
        self.get_logger().info("----- /input_image recieved -----")
        self.get_logger().info(f"Header: {msg.header}")
        self.get_logger().info(f"Height: {msg.height}")
        self.get_logger().info(f"Width: {msg.width}")
        self.get_logger().info(f"Encoding: {msg.encoding}")
        self.get_logger().info(f"Data Length: {len(msg.data)}")
    
    def query_callback(self, msg: String):
        self.get_logger().info("----- /input_query recieved -----")
        self.get_logger().info(f"Data: {msg.data}")
    
    def flip_callback(self, msg: Bool):
        self.get_logger().info("----- /flip_image recieved -----")
        self.get_logger().info(f"Data: {msg.data}")
    
    def output_callback(self, msg: String):
        self.get_logger().info("----- /output recieved -----")
        self.get_logger().info(f"Data: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = TopicSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
