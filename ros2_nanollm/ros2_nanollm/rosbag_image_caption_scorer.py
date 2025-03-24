# SPDX-FileCopyrightText: Copyright (c) <year>
# SPDX-License-Identifier: Apache-2.0

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nanollm_interfaces.msg import StringStamped
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nano_llm import NanoLLM, ChatHistory
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import os
import csv
import cv2

class RosbagImageCaptionScorer(Node):  # ← クラス名変更

    def __init__(self):
        super().__init__('rosbag_image_caption_scorer')  # ノード名も合わせてわかりやすく

        # パラメータ
        self.declare_parameter('model', "Efficient-Large-Model/VILA1.5-3b")
        self.declare_parameter('api', "mlc")
        self.declare_parameter('quantization', "q4f16_ft")
        self.declare_parameter('is_compressed', False)
        self.declare_parameter('query', "Please classify the location as private road, public road, or parking lot.")

        self.model_name = self.get_parameter('model').get_parameter_value().string_value
        self.query = self.get_parameter('query').get_parameter_value().string_value
        self.is_compressed = self.get_parameter('is_compressed').get_parameter_value().bool_value

        # ディレクトリパスを受け取るサブスクライバ
        self.save_dir = None
        self.dir_subscription = self.create_subscription(
            String,
            'save_dir',
            self.dir_callback,
            10
        )

        # クエリ受信
        self.query_subscription = self.create_subscription(
            String,
            'input_query',
            self.query_listener_callback,
            10
        )

        # 画像受信 (raw or compressed)
        if self.is_compressed:
            qos_profile = QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )
            self.image_subscription = self.create_subscription(
                CompressedImage,
                'input_image',
                self.compressedimage_listener_callback,
                qos_profile
            )
        else:
            self.image_subscription = self.create_subscription(
                Image,
                'input_image',
                self.image_listener_callback,
                10
            )

        self.cv_br = CvBridge()
        self.cv_img = None
        self.image_stamp = None

        # モデルのロード
        self.get_logger().info(f"load model: {self.model_name}")
        self.model = NanoLLM.from_pretrained(self.model_name)

        # チャット履歴用
        self.chat_history = ChatHistory(self.model)

        # Publisher
        self.output_publisher = self.create_publisher(StringStamped, '/output', 10)
        if self.is_compressed:
            self.image_publisher = self.create_publisher(Image, "/source_image", 10)

        # ループ処理
        timer_period = 0.001
        self.timer = self.create_timer(timer_period, self.nano_llm_inference)

    def dir_callback(self, msg: String):
        """画像やCSVの保存先ディレクトリを受け取る"""
        self.save_dir = msg.data
        self.get_logger().info(f"Set output directory to: {self.save_dir}")

    def query_listener_callback(self, msg: String):
        """クエリを受け取って更新"""
        self.get_logger().info(f"query_listener_callback: {msg.data}")
        self.query = msg.data

    def compressedimage_listener_callback(self, data: CompressedImage):
        """CompressedImage を受信して cv_img に変換"""
        self.image_stamp = data.header.stamp
        self.cv_img = self.cv_br.compressed_imgmsg_to_cv2(data, "bgr8")
        # 必要なら再パブリッシュ (raw Image形式)
        if self.is_compressed:
            img_msg = self.cv_br.cv2_to_imgmsg(self.cv_img, encoding="bgr8")
            img_msg.header.stamp = data.header.stamp
            img_msg.header.frame_id = "image"
            self.image_publisher.publish(img_msg)

    def image_listener_callback(self, data: Image):
        """Image を受信して cv_img に変換"""
        self.image_stamp = data.header.stamp
        in_bgr = self.cv_br.imgmsg_to_cv2(data, "bgr8")
        self.cv_img = in_bgr

    def nano_llm_inference(self):
        """
        - 画像があれば LLM で推論
        - 出力を下部黒枠付きの画像に描画し保存
        - CSV に結果を追記
        """
        if self.cv_img is None:
            return

        stamp = self.image_stamp
        prompt = self.query.strip("][()")

        # LLM が内部でPIL画像を要する場合があるのでRGBに変換してPILに渡す
        cv_img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv_img_rgb)

        # チャット履歴に画像とプロンプトを追加
        self.chat_history.append('user', image=pil_img)
        self.chat_history.append('user', prompt, use_cache=True)
        embedding, _ = self.chat_history.embed_chat()

        # 推論実行
        output = self.model.generate(
            inputs=embedding,
            kv_cache=self.chat_history.kv_cache,
            min_new_tokens=10,
            streaming=False,
            do_sample=True,
        )

        # 出力をパブリッシュ
        output_msg = StringStamped()
        output_msg.header.stamp = stamp
        output_msg.data = output
        self.output_publisher.publish(output_msg)
        self.get_logger().info(f"Published output: {output}")

        # ===== 画像下部にテキスト描画し保存 =====
        if self.save_dir:
            inference_img_dir = os.path.join(self.save_dir, "inference_images")
            os.makedirs(inference_img_dir, exist_ok=True)

            timestamp_str = f"{stamp.sec}_{stamp.nanosec}"
            image_file_path = os.path.join(inference_img_dir, f"{timestamp_str}.png")

            # 下部に黒枠を追加した画像を作成
            height, width, _ = self.cv_img.shape
            black_bar_height = height // 8
            new_height = height + black_bar_height

            new_image = np.zeros((new_height, width, 3), dtype=np.uint8)
            new_image[0:height, 0:width] = self.cv_img

            # テキスト描画
            text_str = output
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)

            (text_width, text_height), baseline = cv2.getTextSize(
                text_str, font_face, font_scale, thickness
            )
            text_x = max((width - text_width) // 2, 0)
            text_y = height + (black_bar_height + text_height) // 2

            cv2.putText(
                new_image,
                text_str,
                (text_x, text_y),
                font_face,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

            # 画像保存
            cv2.imwrite(image_file_path, new_image)
            self.get_logger().info(f"Saved inference image: {image_file_path}")

            # CSVにスコアを追記
            score = self.calculate_score(output)
            csv_path = os.path.join(self.save_dir, "inference_result.csv")
            write_header = not os.path.exists(csv_path)

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["Timestamp", "Result", "filepath"])
                writer.writerow([timestamp_str, score, image_file_path])

            self.get_logger().info(f"Appended score {score} to {csv_path}")

        # 推論完了後リセット
        self.chat_history.reset()
        self.cv_img = None

    def calculate_score(self, output: str) -> int:
        """
        LLM 出力の文字列に応じてスコアを付与する。
          1) "parking lot" が含まれる場合 → 2
          2) "private" と "public" の両方含む場合 → 0
          3) "private" のみ → -1
          4) "public" のみ → +1
          5) 上記いずれにも該当しなければ -3
        """
        out_lower = output.lower()

        if "parking lot" in out_lower:
            return 2

        has_private = "private" in out_lower
        has_public = "public" in out_lower

        if has_private and has_public:
            return 0
        if has_private:
            return -1
        if has_public:
            return 1

        return -3

def main(args=None):
    rclpy.init(args=args)
    node = RosbagImageCaptionScorer()  # クラス名を使用
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()