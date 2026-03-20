import time
from multiprocessing import Process, Value, Array
import sys
import os

# Jetson Orin Nanoのモデル認識問題の回避
# gpio_pin_dataモジュールを完全にファイルパスベースでロード
try:
    import importlib.util

    # Jetson.GPIOのインストールパスを探す
    jetson_gpio_paths = [
        '/home/jetson/venv/lib/python3.10/site-packages/Jetson/GPIO',
        '/usr/local/lib/python3.10/dist-packages/Jetson/GPIO',
        '/usr/lib/python3/dist-packages/Jetson/GPIO',
    ]

    gpio_pin_data_path = None
    for base_path in jetson_gpio_paths:
        candidate = os.path.join(base_path, 'gpio_pin_data.py')
        if os.path.exists(candidate):
            gpio_pin_data_path = candidate
            break

    if gpio_pin_data_path:
        # ファイルから直接モジュールをロード
        spec = importlib.util.spec_from_file_location('Jetson.GPIO.gpio_pin_data', gpio_pin_data_path)
        gpio_pin_data = importlib.util.module_from_spec(spec)

        # sys.modulesに先に登録
        sys.modules['Jetson.GPIO.gpio_pin_data'] = gpio_pin_data

        # モジュールを実行
        spec.loader.exec_module(gpio_pin_data)

        # 実行後にパッチを適用
        original_get_model = gpio_pin_data.get_model

        def patched_get_model():
            """Jetson Orin Nanoのモデル認識をフォールバック"""
            try:
                return original_get_model()
            except Exception:
                # モデル認識に失敗した場合、Jetson Orin Nanoとして扱う
                print("Warning: Could not detect Jetson model, assuming JETSON_ORIN_NANO")
                return "JETSON_ORIN_NANO"

        gpio_pin_data.get_model = patched_get_model
        print("Jetson GPIO model detection patched successfully")
except (ImportError, AttributeError, Exception) as e:
    # Jetson.GPIOがない環境では何もしない
    print(f"Note: Could not patch Jetson.GPIO: {e}")
    pass

from pmw3901 import BG_CS_BACK_BCM, BG_CS_FRONT_BCM, PMW3901 as PMW3901_Base
import config

# Jetson互換のPMW3901ラッパー (no_cs属性の問題を回避)
class PMW3901(PMW3901_Base):
    def __init__(self, spi_port=0, spi_cs=1, spi_cs_gpio=BG_CS_FRONT_BCM):
        import time
        import spidev
        try:
            import RPi.GPIO as GPIO
        except:
            import Jetson.GPIO as GPIO

        self.spi_cs_gpio = spi_cs_gpio
        self.spi_dev = spidev.SpiDev()
        self.spi_dev.open(spi_port, spi_cs)
        self.spi_dev.max_speed_hz = 400000

        # Jetsonでno_cs属性がサポートされていない場合はスキップ
        try:
            self.spi_dev.no_cs = True
        except (OSError, AttributeError):
            print("Warning: SPI no_cs not supported, using GPIO CS control only")

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.spi_cs_gpio, GPIO.OUT)

        GPIO.output(self.spi_cs_gpio, 0)
        time.sleep(0.05)
        GPIO.output(self.spi_cs_gpio, 1)

        self._write(0x3a, 0x5a)  # REG_POWER_UP_RESET
        time.sleep(0.02)
        for offset in range(5):
            self._read(0x02 + offset)  # REG_DATA_READY

        self._secret_sauce()

        product_id, revision = self.get_id()
        if product_id != 0x49 or revision != 0x00:
            raise RuntimeError("Invalid Product ID or Revision for PMW3901: 0x{:02x}/0x{:02x}".format(product_id, revision))

class OpticalFlowSensor:
    def __init__(self, polling_interval=0.1, timeout_s=0.005, sampling_count=6):
        # Sensor state
        self.is_active = Value('b', 1)
        self.pixel_motion = Array('f', 2)  # Pixel change detected by the sensor
        self.cumulative_pixel_motion = Array('f', 2)  # Cumulative pixel change for interval
        self.position = Array('f', 2)  # Absolute position in mm
        self.velocity = Array('i', 2)  # Velocity in mm/s
        self.previous_velocity = Array('f', 2)
        self.acceleration = Array('f', 2)

        # Configuration parameters
        self.polling_interval = polling_interval
        self.timeout_s = timeout_s
        self.sampling_count = Value('i', sampling_count)
        self.position_scaling_factor = config.POSITION_SCALING_FACTOR

        # Process management
        self.process = Process(target=self._opticalflow_process)
        self.process.start()

    def _opticalflow_process(self):
        # 子プロセスでもJetson GPIOパッチを適用
        try:
            import importlib.util
            import sys
            import os

            jetson_gpio_paths = [
                '/home/jetson/venv/lib/python3.10/site-packages/Jetson/GPIO',
                '/usr/local/lib/python3.10/dist-packages/Jetson/GPIO',
                '/usr/lib/python3/dist-packages/Jetson/GPIO',
            ]

            gpio_pin_data_path = None
            for base_path in jetson_gpio_paths:
                candidate = os.path.join(base_path, 'gpio_pin_data.py')
                if os.path.exists(candidate):
                    gpio_pin_data_path = candidate
                    break

            if gpio_pin_data_path:
                spec = importlib.util.spec_from_file_location('Jetson.GPIO.gpio_pin_data', gpio_pin_data_path)
                gpio_pin_data = importlib.util.module_from_spec(spec)
                sys.modules['Jetson.GPIO.gpio_pin_data'] = gpio_pin_data
                spec.loader.exec_module(gpio_pin_data)

                original_get_model = gpio_pin_data.get_model

                def patched_get_model():
                    try:
                        return original_get_model()
                    except Exception:
                        return "JETSON_ORIN_NANO"

                gpio_pin_data.get_model = patched_get_model
        except Exception:
            pass

        # 子プロセスでもJetson互換PMW3901を使用
        import time
        import spidev
        from pmw3901 import BG_CS_FRONT_BCM, PMW3901 as PMW3901_Base
        try:
            import RPi.GPIO as GPIO
        except:
            import Jetson.GPIO as GPIO

        class PMW3901_Jetson(PMW3901_Base):
            def __init__(self, spi_port=0, spi_cs=1, spi_cs_gpio=BG_CS_FRONT_BCM):
                self.spi_cs_gpio = spi_cs_gpio
                self.spi_dev = spidev.SpiDev()
                self.spi_dev.open(spi_port, spi_cs)
                self.spi_dev.max_speed_hz = 400000

                # Jetsonでno_cs属性がサポートされていない場合はスキップ
                try:
                    self.spi_dev.no_cs = True
                except (OSError, AttributeError):
                    print("Warning: SPI no_cs not supported, using GPIO CS control only")

                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.spi_cs_gpio, GPIO.OUT)

                GPIO.output(self.spi_cs_gpio, 0)
                time.sleep(0.05)
                GPIO.output(self.spi_cs_gpio, 1)

                self._write(0x3a, 0x5a)  # REG_POWER_UP_RESET
                time.sleep(0.02)
                for offset in range(5):
                    self._read(0x02 + offset)  # REG_DATA_READY

                self._secret_sauce()

                product_id, revision = self.get_id()
                if product_id != 0x49 or revision != 0x00:
                    raise RuntimeError("Invalid Product ID or Revision for PMW3901: 0x{:02x}/0x{:02x}".format(product_id, revision))

        try:
            # spi_cs: SPIチップセレクト番号 (0 or 1, /dev/spidev0.X)
            # spi_cs_gpio: GPIOピン番号 (ソフトウェアCS制御用)
            sensor = PMW3901_Jetson(spi_port=0, spi_cs=1, spi_cs_gpio=BG_CS_FRONT_BCM)
            sensor.set_rotation(0)
            print("Optical Flow Sensor initialized.")
        except Exception as e:
            import traceback
            print(f"Failed to initialize sensor: {e}")
            print("Full traceback:")
            traceback.print_exc()
            self.is_active.value = 0
            return

        while self.is_active.value:
            try:
                self.cumulative_pixel_motion[:] = [0, 0]
                start_time = time.perf_counter()

                for _ in range(self.sampling_count.value):
                    try:
                        motion = sensor.get_motion(self.timeout_s)
                        self.cumulative_pixel_motion[0] += motion[0]
                        self.cumulative_pixel_motion[1] += motion[1]
                    except RuntimeError:
                        continue

                # Calculate elapsed time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Convert cumulative pixel motion to mm
                delta_x = self.cumulative_pixel_motion[0] * self.position_scaling_factor
                delta_y = self.cumulative_pixel_motion[1] * self.position_scaling_factor

                # Update position
                self.position[0] += delta_x
                self.position[1] += delta_y

                # Update velocity
                self.velocity[0] = int(delta_x / elapsed_time)
                self.velocity[1] = int(delta_y / elapsed_time)

                # Update acceleration
                self.acceleration[0] = (self.velocity[0] - self.previous_velocity[0]) / elapsed_time
                self.acceleration[1] = (self.velocity[1] - self.previous_velocity[1]) / elapsed_time

                # Store current velocity as previous velocity for next loop
                self.previous_velocity[:] = self.velocity[:]

            except Exception as e:
                print(f"Error in process loop: {e}")

    def update(self):
        while self.is_active.value:
            self.poll()
            time.sleep(self.polling_interval)

    def poll(self):
        if self.is_active.value:
            return self.velocity[0], self.velocity[1]
        pass

    def run(self):
        self.poll()
        return self.velocity[0], self.velocity[1]

    def shutdown(self):
        self.is_active.value = 0
        self.process.join()
        print("Optical Flow Sensor process terminated.")

    def calibration_check(self, move_distance_mm=50, move_duration_s=5):
        """Check sensor calibration by moving a set distance and duration."""
        print("現在のPOSITION_SCALING_FACTOR: ",config.POSITION_SCALING_FACTOR)
        message = f"Enter：キャリブレーション開始。{move_duration_s}秒以内に{move_distance_mm}mm マシンを前進。"
        input(message)
        print("Starting calibration check...")
        initial_position = list(self.position[:])
        start_time = time.time()

        while time.time() - start_time < move_duration_s:
            print(f"Current Position: {list(self.position[:])} mm")
            time.sleep(self.polling_interval)

        final_position = list(self.position[:])
        moved_distance = [final_position[i] - initial_position[i] for i in range(2)]
        adjusted_scaling_factor = moved_distance[1] / move_distance_mm * self.position_scaling_factor
        print(f"移動量(y): {moved_distance[1]} mm (Expected: {move_distance_mm}mm)")
        print(f"必要に応じ config.py の POSITION_SCALING_FACTOR を修正: {abs(adjusted_scaling_factor)}")

# ROS2の有無を判定してインポート
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Vector3

    class OpticalFlowNode(Node):
        def __init__(self, sensor: OpticalFlowSensor):
            super().__init__('optical_flow_node')

            # OpticalFlowSensor インスタンスを受け取る
            self.sensor = sensor

            # パブリッシャーの設定
            self.velocity_pub = self.create_publisher(Vector3, '/optical_flow/velocity', 10)
            self.position_pub = self.create_publisher(Vector3, '/optical_flow/position', 10)
            self.acceleration_pub = self.create_publisher(Vector3, '/optical_flow/acceleration', 10)

            # タイマー設定（データ送信間隔）
            self.timer_period = self.sensor.polling_interval  # 既存センサーのポーリング間隔を活用
            self.timer = self.create_timer(self.timer_period, self.publish_data)

            self.get_logger().info("OpticalFlowNode initialized.")

        def publish_data(self):
            try:
                # 速度データのパブリッシュ
                velocity_msg = Vector3()
                velocity_msg.x = float(self.sensor.velocity[0])
                velocity_msg.y = float(self.sensor.velocity[1])
                velocity_msg.z = 0.0  # OpticalFlowは2DなのでZは0
                self.velocity_pub.publish(velocity_msg)

                # 位置データのパブリッシュ
                position_msg = Vector3()
                position_msg.x = float(self.sensor.position[0])
                position_msg.y = float(self.sensor.position[1])
                position_msg.z = 0.0
                self.position_pub.publish(position_msg)
               
                # 加速度データのパブリッシュ
                acceleration_msg = Vector3()
                acceleration_msg.x = float(self.sensor.acceleration[0])
                acceleration_msg.y = float(self.sensor.acceleration[1])
                acceleration_msg.z = 0.0
                self.acceleration_pub.publish(acceleration_msg)

                self.get_logger().debug(
                    f"Published - Velocity: {velocity_msg}, Position: {position_msg}, Acceleration: {acceleration_msg}"
                )
            except Exception as e:
                self.get_logger().error(f"Error in publish_data: {e}")

        def shutdown(self):
            self.sensor.shutdown()
            self.get_logger().info("OpticalFlowNode shutting down.")

    def main_ros():
        sensor = OpticalFlowSensor(polling_interval=0.1, sampling_count=10)
        try:
            # ROS2初期化
            rclpy.init()
            # ROSノード起動
            node = OpticalFlowNode(sensor)
            rclpy.spin(node)

        except KeyboardInterrupt:
            print("Shutting down OpticalFlowNode...")
        finally:
            # シャットダウン処理
            if rclpy.ok():
                node.shutdown()
                rclpy.shutdown()

except ImportError:
    # print("ROS2関連ライブラリがインストールされていません。ROS2モードは無効です。")
    rclpy = None

def main_manual():
    sensor = OpticalFlowSensor(polling_interval=0.1, sampling_count=10)
    try:
        answer = ""
        while (answer == ""):
            sensor.calibration_check()
            answer = input("Enter：再キャリブレーション / 任意のキー：速度測定\n")
        while True:
            print(f"Velocity: {sensor.run()} mm/s")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sensor.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Oplicalflow with or without ROS2")
    parser.add_argument('--ros', action='store_true', help="Run with ROS2 node")
    args = parser.parse_args()

    if args.ros and rclpy:
        print("Open another terminal and check the velocity values by typing:\n ros2 topic echo /optical_flow/velocity")
        main_ros()
    else:
        if args.ros and not rclpy:
            print("Warning: ROS2 is not available. Switching to manual mode.")
        main_manual()        
