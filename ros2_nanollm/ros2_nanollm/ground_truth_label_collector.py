import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import csv
import os
import threading

class LabelCollectorNode(Node):
    def __init__(self):
        super().__init__('label_collector_node')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/input_image',
            self.listener_callback,
            10)
        
        self.dir_subscription = self.create_subscription(
            String,
            '/save_dir',
            self.save_dir_callback,
            10)

        self.current_label = 1  # 公道
        self.is_stopping = 1    # 停車中 (初期値)
        self.csv_rows = []
        self.save_dir = None
        self.lock = threading.Lock()

        self.get_logger().info("Press 1: 公道, 2: 私道, s: 停車切替, q: 終了")

        self.input_thread = threading.Thread(target=self.keyboard_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

    def save_dir_callback(self, msg: String):
        with self.lock:
            self.save_dir = msg.data
            self.get_logger().info(f"Save directory set to: {self.save_dir}")

    def keyboard_listener(self):
        while rclpy.ok():
            key = input().strip()
            with self.lock:
                if key == '1':
                    self.current_label = 1
                    self.get_logger().info('Label set to 公道 (1)')
                elif key == '2':
                    self.current_label = 2
                    self.get_logger().info('Label set to 私道 (2)')
                elif key.lower() == 's':
                    self.is_stopping = 1 - self.is_stopping
                    self.get_logger().info(f'is_stopping toggled to {self.is_stopping}')
                elif key.lower() == 'q':
                    self.get_logger().info("Saving CSV and exiting...")
                    self.save_csv()
                    rclpy.shutdown()
                    break

    def listener_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        display_text = f'Label: {self.current_label} | Stopping: {self.is_stopping}'
        cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow('Input Image', image)
        cv2.waitKey(1)

        with self.lock:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.csv_rows.append([timestamp, self.current_label, self.is_stopping])

    def save_csv(self):
        if not self.save_dir:
            self.get_logger().error("Save directory not set via /save_dir topic!")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, 'ground_truth.csv')

        try:
            with open(path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Label', 'is_stopping'])
                writer.writerows(self.csv_rows)
            self.get_logger().info(f'CSV saved: {path}')
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LabelCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()