import multiprocessing
import ctypes
import numpy as np
from time import perf_counter, sleep
from typing import cast
import config
import platform
import signal
import logging
import time
import gc
import subprocess

# Conditional cv2 import based on device type to avoid bus errors on RPi5
if config.DEVICE_TYPE == 'RPI5':
    # Skip cv2 import on RPi5 to avoid bus error
    CV2_AVAILABLE = False
    cv2 = None
    # print("OpenCV import skipped on RPi5 to avoid bus error")
else:
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError as e:
        CV2_AVAILABLE = False
        cv2 = None
        print(f"Warning: OpenCV not available: {e}")

# ROS2の有無を判定してインポート
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from rclpy.executors import MultiThreadedExecutor

    class CameraNode(Node):
        def __init__(self, node_name: str, topic_name: str, frame_id: str, queue_size: int):
            super().__init__(node_name)
            self.publisher = self.create_publisher(Image, topic_name, queue_size)
            self.frame_id = frame_id

        def publish_frame(self, frame: np.ndarray):
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.height, msg.width, channels = frame.shape
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = msg.width * channels
            msg.data = frame.tobytes()
            self.publisher.publish(msg)

    class RVizSubscriber(Node):
        def __init__(self, topic_name: str):
            super().__init__('rviz_subscriber')
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.listener_callback,
                qos_profile_sensor_data=None
            )

        def listener_callback(self, msg):
            self.get_logger().info(f"Receiving frame on {msg.header.frame_id}...")

except ImportError:
    rclpy = None

# 既存のカメララッパー
class BaseCameraWrapper:
    def read(self):
        raise NotImplementedError("Subclasses must implement 'read'.")

    def release(self):
        raise NotImplementedError("Subclasses must implement 'release'.")

    def get_data(self):
        return self.read()[1] # [1]only return image data
    
    def cleanup(self):
        self.release()
        time.sleep(0.05)  # GStreamerリソース解放の最小待機（短縮）
        gc.collect()
        logging.getLogger(__name__).info(f"Camera cleanup complete.")

class PiCameraWrapper(BaseCameraWrapper):
    def __init__(self, device_id):
        # picamera2のログレベルを抑制
        import logging
        logging.getLogger('picamera2').setLevel(logging.ERROR)
        
        from picamera2 import Picamera2
        from libcamera import Transform
        
        self.id = device_id
        self.picam2 = Picamera2(camera_num=device_id)
        
        # Determine flip settings based on camera ID
        if device_id == 0:
            vflip = config.CAMERA_0_VFLIP
            hflip = config.CAMERA_0_HFLIP
        else:
            vflip = config.CAMERA_1_VFLIP
            hflip = config.CAMERA_1_HFLIP
        
        # Transform設定（libcameraのTransformを使用）
        transform = Transform()
        if vflip and hflip:
            # 180度回転
            transform = Transform(vflip=True, hflip=True)
        elif vflip:
            # 垂直反転のみ
            transform = Transform(vflip=True)
        elif hflip:
            # 水平反転のみ
            transform = Transform(hflip=True)
        
        # カメラ設定
        picamera_config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (config.IMAGE_W, config.IMAGE_H)},
            transform=transform  # transformパラメータで反転を設定
        )
        self.picam2.configure(picamera_config)
        
        # フレームレートのみ設定（VerticalFlip/HorizontalFlipは削除）
        controls = {"FrameRate": config.CAMERA_FRAMERATE}
        
        # 利用可能なコントロールを確認してから設定
        available_controls = self.picam2.camera_controls
        if "FrameRate" in available_controls:
            try:
                self.picam2.set_controls(controls)
            except RuntimeError as e:
                print(f"Warning: Could not set FrameRate control: {e}")
                # フレームレートが設定できない場合は続行
        
        self.picam2.start()
        
        # センサー名を取得
        try:
            sensor_name = "Unknown"
            # カメラのプロパティからセンサー名を取得
            camera_properties = self.picam2.camera_properties
            if 'Model' in camera_properties:
                sensor_name = camera_properties['Model']
            logging.getLogger(__name__).info(f"Camera {device_id}: OK ({sensor_name})")
        except Exception:
            logging.getLogger(__name__).info(f"Camera {device_id}: OK")
        
        sleep(0.1)

    def read(self):
        frame = self.picam2.capture_array()
        return True, frame

    def release(self):
        self.picam2.stop()

class JetsonCameraWrapper(BaseCameraWrapper):
    def __init__(self, device_id=0):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for JetsonCameraWrapper but not available")

        self.device_id = device_id
        
        # Determine flip method based on camera ID and config
        if device_id == 0:
            vflip = config.CAMERA_0_VFLIP
            hflip = config.CAMERA_0_HFLIP
        else:
            vflip = config.CAMERA_1_VFLIP
            hflip = config.CAMERA_1_HFLIP
        
        # GStreamer flip-method values:
        # 0: none, 1: counterclockwise, 2: rotate-180, 3: clockwise
        # 4: horizontal-flip, 5: upper-right-diagonal, 6: vertical-flip, 7: upper-left-diagonal
        if vflip and hflip:
            flip_method = 2  # rotate-180 (both flips)
        elif vflip:
            flip_method = 6  # vertical-flip
        elif hflip:
            flip_method = 4  # horizontal-flip
        else:
            flip_method = 0  # none
        
        # nvvideoconvertが利用可能かチェック
        use_nvvideoconvert = self._check_gstreamer_element('nvvideoconvert')
        
        if use_nvvideoconvert:
            # パフォーマンス最適化版: nvvideoconvert + VIC
            self.pipeline = (
                f"nvarguscamerasrc sensor-id={device_id} "
                f"bufapi-version=1 ! "  # 低遅延バッファAPI
                f"video/x-raw(memory:NVMM), width={config.IMAGE_W}, height={config.IMAGE_H}, "
                f"format=(string)NV12, framerate={config.CAMERA_FRAMERATE}/1 ! "
                f"nvvideoconvert "
                f"flip-method={flip_method} "
                f"interpolation-method=0 "  # Nearest（最速）
                f"compute-hw=2 "  # VIC使用（省電力・高速）
                f"nvbuf-memory-type=0 ! "  # Device memory
                "video/x-raw, format=(string)BGRx ! "
                "appsink drop=true max-buffers=1 sync=false emit-signals=false"
            )
            self._using_bgrx = True  # BGRx形式を使用中フラグ
            print(f"Jetson camera@id:{device_id}, flip-method:{flip_method} [Optimized: nvvideoconvert+VIC]")
            
        else:
            # フォールバック: 従来のnvvidconv版（BGRで直接取得）
            self.pipeline = (
                f"nvarguscamerasrc sensor-id={device_id} ! "
                f"video/x-raw(memory:NVMM), width={config.IMAGE_W}, height={config.IMAGE_H}, "
                f"format=(string)NV12, framerate={config.CAMERA_FRAMERATE}/1 ! "
                f"nvvidconv flip-method={flip_method} ! "
                "video/x-raw, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self._using_bgrx = False  # BGR形式を使用中フラグ
            print(f"Jetson camera@id:{device_id}, flip-method:{flip_method} [Legacy: nvvidconv]")
        
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open Jetson CSI camera with ID {device_id}.")
    
    def _check_gstreamer_element(self, element_name):
        """GStreamerエレメントの利用可能性をチェック"""
        try:
            result = subprocess.run(
                ['gst-inspect-1.0', element_name],
                capture_output=True,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def read(self):
        """
        既存インターフェースを維持: BGR形式で返す
        Returns:
            tuple: (ret, frame) - frameはBGR形式（3チャンネル）
        """
        # カメラが解放されている場合はエラーを回避
        if not hasattr(self, 'cap') or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return ret, None

        # BGRx形式の場合はBGRに変換（最初の3チャンネルを抽出）
        if self._using_bgrx and frame.shape[2] == 4:
            # NumPyスライシングで高速にBGRx→BGR変換
            frame = frame[:, :, :3].copy()  # copyで連続メモリを確保

        return ret, frame
    
    def release(self):
        """リソースを解放"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        # 最小限の待機とクリーンアップ
        time.sleep(0.1)  # 短縮: 最小待機時間
        gc.collect()
        # pkillを非ブロッキングで実行（タイムアウト短縮）
        try:
            subprocess.run(['pkill', '-f', 'nvarguscamerasrc'],
                         stderr=subprocess.DEVNULL, timeout=0.5)
        except:
            pass
            
# class JetsonCameraWrapper(BaseCameraWrapper):
#     def __init__(self, device_id=0):
#         if not CV2_AVAILABLE:
#             raise RuntimeError("OpenCV is required for JetsonCameraWrapper but not available")
#         self.device_id = device_id
        
#         # Determine flip method based on camera ID and config
#         if device_id == 0:
#             vflip = config.CAMERA_0_VFLIP
#             hflip = config.CAMERA_0_HFLIP
#         else:
#             vflip = config.CAMERA_1_VFLIP
#             hflip = config.CAMERA_1_HFLIP
        
#         # GStreamer flip-method values:
#         # 0: none, 1: counterclockwise, 2: rotate-180, 3: clockwise
#         # 4: horizontal-flip, 5: upper-right-diagonal, 6: vertical-flip, 7: upper-left-diagonal
#         if vflip and hflip:
#             flip_method = 2  # rotate-180 (both flips)
#         elif vflip:
#             flip_method = 6  # vertical-flip
#         elif hflip:
#             flip_method = 4  # horizontal-flip
#         else:
#             flip_method = 0  # none
        

#         self.pipeline = (
#             f"nvarguscamerasrc sensor-id={device_id} ! "
#             f"video/x-raw(memory:NVMM), width={config.IMAGE_W}, height={config.IMAGE_H}, format=(string)NV12, framerate={config.CAMERA_FRAMERATE}/1 ! "
#             f"nvvidconv flip-method={flip_method} ! "
#             "video/x-raw, format=(string)BGRx ! "
#             "videoconvert ! "
#             "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
#         )

#         self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Failed to open Jetson CSI camera with ID {device_id}.")
#         print(f"Jetson camera@id:{device_id}, flip-method:{flip_method}")

#     def read(self):
#         ret, frame = self.cap.read()
#         if not ret or frame is None:
#             return ret, None
#         return ret, frame

#     def release(self):
#         if hasattr(self, 'cap') and self.cap is not None:
#             self.cap.release()
#             self.cap = None
#         # Force cleanup GST pipeline resources with proper timing
#         import gc
#         import time
#         time.sleep(0.5)  # GStreamerパイプラインの完全停止を待つ
#         gc.collect()
#         # Additional GST cleanup for Jetson cameras
#         try:
#             import subprocess
#             subprocess.run(['pkill', '-f', 'nvarguscamerasrc'], stderr=subprocess.DEVNULL, timeout=2)
#         except:
#             pass


class MultiprocessCameraWrapper(BaseCameraWrapper):
    def __init__(self, base_camera_type: type, device_id: int):
        self.base_camera_type = base_camera_type
        self.device_id = device_id
        self.__buffer = None
        self.__ready = None
        self.__cancel = None
        self.__shape = None
        self.__process = None
        self.__released = False
        self._initialize_shared_memory()

    ###
    def _initialize_shared_memory(self):
        # 仮のカメラインスタンスを使用してフレームサイズを取得
        #temp_camera = self.base_camera_type(device_id=self.device_id)
        #ret, frame = temp_camera.read()
        #if not ret:
        #    raise RuntimeError(f"Failed to capture initial frame for camera {self.device_id}.")
        #height, width, channels = frame.shape
        #self.__shape = (height, width, channels)
        #temp_camera.release()
        height, width, channels= config.IMAGE_H,config.IMAGE_W,config.IMAGE_DEPTH
        print("shape is:",height, width, channels)

        # 共有メモリと同期用イベントの初期化
        self.__buffer = multiprocessing.Array(
            ctypes.c_uint8, height * width * channels)
        #self.__buffer = multiprocessing.sharedctypes.RawArray(
        #    ctypes.c_uint8, height * width * channels)
        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()

        # バックグラウンドプロセスの開始
        self.__process = multiprocessing.Process(
            target=self._capture_loop,
            args=(self.base_camera_type, self.device_id, self.__buffer, self.__ready, self.__cancel),
            daemon=True
        )
        self.__process.start()

    ### printをコメントアウトのこと
    def _capture_loop(self, camera_type: type, device_id: int, buffer: ctypes.Array[ctypes.c_uint8],
                        ready: multiprocessing.Event, cancel: multiprocessing.Event):
            """
            子プロセス内でカメラを初期化し、フレームを共有メモリに書き込む。
            """
            import signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("id",device_id)
            # 子プロセス内でカメラインスタンスを初期化
            camera = camera_type(device_id=device_id)
            try:
                while not cancel.is_set():
                    start_time = perf_counter()
                    ret, frame = camera.read()
                    if ret:
                        ready.clear()
                        np.copyto(np.ctypeslib.as_array(buffer), frame.ravel())
                        ready.set()
                        # FPS計算
                        fps = round(1 / (perf_counter() - start_time), 2)
                        print(f"id:{device_id} - fps: {fps}")
            finally:
                camera.release()
                logging.getLogger(__name__).debug(f"Camera {device_id}: Released.")

    #def read(self):
    #    """共有メモリからフレームを読み取る。"""
    #    self.__ready.wait()
    #    frame = np.frombuffer(self.__buffer, dtype=np.uint8).reshape(self.__shape)
    #    return True, frame.copy()

    def release(self):
        """カメラとプロセスを終了する。"""
        if self.__released:
            return
        self.__cancel.set()
        self.__process.join()
        self.base_camera.release()
        self.__released = True
        logging.getLogger(__name__).debug(f"Camera {self.device_id}: Process terminated.")

def create_camera(device_id, use_multiprocess=False):
    # Improved platform detection
    node_name = platform.uname().node.lower()
    machine = platform.uname().machine.lower()
    
    # Check for device-tree model (more reliable for ARM devices)
    model = ""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().lower()
    except FileNotFoundError:
        pass
    
    # Platform detection logic
    is_raspberry_pi = "raspberrypi" in node_name or "raspberry" in model
    is_jetson = ("jetson" in node_name or "orin" in node_name or "tegra" in model or 
                "jetson" in model or "orin" in model)

    camera = []
    if is_raspberry_pi:
        base_camera_type = PiCameraWrapper
    elif is_jetson:
        base_camera_type = JetsonCameraWrapper
    else:
        raise RuntimeError("Unsupported platform for camera. Only Raspberry Pi and Jetson devices are supported.")

    if use_multiprocess:
        camera = MultiprocessCameraWrapper(base_camera_type=base_camera_type, device_id=i)
    else:
        camera = base_camera_type(device_id=device_id)

    return camera

def create_cameras(use_multiprocess=False):
    # Improved platform detection (same as create_camera)
    node_name = platform.uname().node.lower()
    machine = platform.uname().machine.lower()
    
    # Check for device-tree model (more reliable for ARM devices)
    model = ""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().lower()
    except FileNotFoundError:
        pass
    
    # Platform detection logic
    is_raspberry_pi = "raspberrypi" in node_name or "raspberry" in model
    is_jetson = ("jetson" in node_name or "orin" in node_name or "tegra" in model or 
                "jetson" in model or "orin" in model)

    cameras = []
    if is_raspberry_pi:
        base_camera_type = PiCameraWrapper
    elif is_jetson:
        base_camera_type = JetsonCameraWrapper
    else:
        raise RuntimeError("Unsupported platform for camera. Only Raspberry Pi and Jetson devices are supported.")

    if use_multiprocess:
        cameras = [MultiprocessCameraWrapper(base_camera_type=base_camera_type, device_id=i) for i in range(1)]
    else:
        cameras = [base_camera_type(device_id=i) for i in range(1)]

    return cameras


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Camera wrapper with multiprocess and ROS2 support")
    parser.add_argument("--multiprocess", action="store_true", help="Use multiprocessing for camera access")
    parser.add_argument("--ros", action="store_true", help="Run with ROS2 node")
    parser.add_argument("--vis", action="store_true", help="Run with RViz visualization")
    parser.add_argument("--nogui", action="store_true", help="Disable GUI display")
    args = parser.parse_args()

    cameras = create_cameras(use_multiprocess=args.multiprocess)

    if args.vis and rclpy:
        rclpy.init()
        rviz_nodes = [
            RVizSubscriber(topic_name=f"/camera{i+1}/image_raw") for i in range(len(cameras))
        ]
        executor = MultiThreadedExecutor()
        for node in rviz_nodes:
            executor.add_node(node)

        try:
            print("Starting RViz visualization...")
            executor.spin()
        except KeyboardInterrupt:
            print("\nStopping RViz visualization.")
        finally:
            for node in rviz_nodes:
                node.destroy_node()
            rclpy.shutdown()

    elif args.ros and rclpy:
        rclpy.init()
        nodes = [
            CameraNode(
                node_name=f"camera_node_{i+1}",
                topic_name=f"/camera{i+1}/image_raw",
                frame_id=f"camera_frame{i+1}",
                queue_size=10
            ) for i in range(len(cameras))
        ]
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)

        try:
            while rclpy.ok():
                for i, camera in enumerate(cameras):
                    ret, frame = camera.read()
                    if ret:
                        nodes[i].publish_frame(frame)
                executor.spin_once(timeout_sec=0.01)
        except KeyboardInterrupt:
            print("\nStopping ROS2 nodes.")
        finally:
            for camera in cameras:
                camera.release()
            for node in nodes:
                node.destroy_node()
            rclpy.shutdown()

    else:
        try:
            while True:
                if args.multiprocess:
                    sleep(0.1)  # メインループの待機間隔
                else:
                    start_time = perf_counter()
                    for i, camera in enumerate(cameras):
                        ret, frame = camera.read()
                        if ret and not args.nogui and CV2_AVAILABLE:
                            cv2.imshow(f"Camera {i+1}", frame)
                        elif ret:
                            logging.getLogger(__name__).debug(f"Camera {i+1}: Frame captured.")
                    print("FPS:", 1 / (perf_counter() - start_time))
                    if not args.nogui and CV2_AVAILABLE and cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            print("\nStopping cameras.")
        finally:
            for camera in cameras:
                camera.release()
            if not args.nogui and CV2_AVAILABLE:
                cv2.destroyAllWindows()
