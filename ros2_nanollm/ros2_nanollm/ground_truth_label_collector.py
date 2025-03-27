import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import csv
import os

class LabelCollectorNode(Node):
    def __init__(self):
        super().__init__('label_collector_node')

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/input_image',
            self.listener_callback,
            qos)

        self.dir_subscription = self.create_subscription(
            String,
            '/save_dir',
            self.save_dir_callback,
            10)

        self.current_label = 1
        self.is_stopping = 0
        self.csv_rows = []
        self.save_dir = None
        self.last_input_time = self.get_clock().now()
        self.input_timeout_sec = 3.0

        self.create_timer(0.5, self.check_input_timeout)

        cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Image", 800, 600)

        self.get_logger().info("キー操作: 1=公道, 2=私道, s=停車切替, q=手動保存+終了")

    def save_dir_callback(self, msg: String):
        new_dir = msg.data
        if self.csv_rows and self.save_dir and new_dir != self.save_dir:
            self.get_logger().info(f"/save_dir が変更されました。前のディレクトリ ({self.save_dir}) に保存します。")
            self.save_csv(self.save_dir)
            self.csv_rows.clear()
        self.save_dir = new_dir
        self.get_logger().info(f"新しい保存先ディレクトリが設定されました: {self.save_dir}")

    def listener_callback(self, msg: CompressedImage):
        self.last_input_time = self.get_clock().now()

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("画像のデコードに失敗しました")
            return

        label_text = "Public" if self.current_label == 1 else "Private"
        stopping_text = "True" if self.is_stopping == 1 else "False"
        display_text = f'Label: {label_text} | Stopping: {stopping_text}'
        self.draw_label_overlay(image, display_text)

        cv2.imshow("Input Image", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            self.current_label = 1
            self.get_logger().info('Label set to Public (1)')
        elif key == ord('2'):
            self.current_label = 2
            self.get_logger().info('Label set to Private (2)')
        elif key == ord('s'):
            self.is_stopping = 1 - self.is_stopping
            self.get_logger().info(f'is_stopping toggled to {self.is_stopping}')
        elif key == ord('q'):
            self.get_logger().info("手動保存 & ノード終了要求")
            self.save_csv()
            cv2.destroyAllWindows()
            rclpy.shutdown()

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        csv_label = 1 if self.current_label == 1 else -1
        self.csv_rows.append([timestamp, csv_label, self.is_stopping])

    def draw_label_overlay(self, image, text, position=(10, 100)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        thickness = 8
        # 縁取り（白）
        cv2.putText(image, text, position, font, font_scale, (255, 255, 255), thickness + 4, cv2.LINE_AA)
        # 本文字（緑）
        cv2.putText(image, text, position, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def check_input_timeout(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_input_time).nanoseconds * 1e-9
        if elapsed > self.input_timeout_sec and self.csv_rows:
            self.get_logger().warn(f"{self.input_timeout_sec}秒間 /input_image が届いていません。自動保存します。")
            self.save_csv()
            self.csv_rows.clear()
            self.last_input_time = now

    def save_csv(self, target_dir=None):
        dir_to_use = target_dir or self.save_dir
        if not dir_to_use:
            self.get_logger().error("保存先ディレクトリが指定されていません")
            return

        os.makedirs(dir_to_use, exist_ok=True)
        path = os.path.join(dir_to_use, 'ground_truth.csv')

        try:
            with open(path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Label', 'is_stopping'])
                writer.writerows(self.csv_rows)
            self.get_logger().info(f"CSV を保存しました: {path}")
        except Exception as e:
            self.get_logger().error(f"CSV保存中にエラー発生: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LabelCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()