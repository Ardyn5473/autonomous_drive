# aggregator.py

import collections

class DataAggregator:
    """
    複数センサーのデータと、ステアリング／スロットルなどの制御値をまとめて管理するクラス。
    """
    def __init__(self, sensor_instances, max_history=50):
        """
        Args:
            sensor_instances (dict): 
              例: {
                "Fr": ultrasonic.Ultrasonic("Fr"),
                "camera_0": camera.create_camera(...),
                ...
              }
            max_history (int): 過去に保持するフレーム数
        """
        self.sensor_instances = sensor_instances
        self.max_history = max_history

        # センサーごとのリングバッファ（deque）を用意
        # sensor_buffers["Fr"], sensor_buffers["camera_0"] のようにセンサー名で区別
        self.sensor_buffers = {}
        for name in sensor_instances.keys():
            self.sensor_buffers[name] = collections.deque(maxlen=max_history)

        # ステアリング／スロットルを記録するための deque
        self.steering_buffer = collections.deque(maxlen=max_history)
        self.throttle_buffer = collections.deque(maxlen=max_history)

    def update_sensors(self):
        """
        センサーインスタンスから最新値を取得し、各バッファに格納する。
        """
        for name, sensor in self.sensor_instances.items():
            # ultrasonic などは .measure()
            # camera などは .read() メソッドを使用
            # lidar などは .run() メソッドを使用
            if hasattr(sensor, "measure"):
                value = sensor.measure()
            elif name.startswith("camera_") and hasattr(sensor, "read"):
                # カメラの場合は read() メソッドを使用
                ret, frame = sensor.read()
                value = frame if ret and frame is not None else None
            elif name == "lidar" and hasattr(sensor, "run"):
                # LiDARの場合は run() メソッドを使用
                detection_distances, detection_binary, wall_segments, image, measurements = sensor.run()
                # zone_distancesを取得（各ゾーンの中央値距離）
                zone_distances = sensor.zone_distances if hasattr(sensor, 'zone_distances') else detection_distances
                # LiDARデータを辞書形式で格納
                value = {
                    'zone_distances': zone_distances,
                    'detection_distances': detection_distances,
                    'detection_binary': detection_binary,
                    'image': image,
                    'measurements': measurements,
                    'wall_segments': wall_segments
                }
            elif hasattr(sensor, "get_data"):
                value = sensor.get_data()
            else:
                value = None
            self.sensor_buffers[name].append(value)

    def add_control_data(self, steering, throttle):
        """
        1フレームごとに決定したステアリング／スロットルをバッファに追記する
        """
        self.steering_buffer.append(steering)
        self.throttle_buffer.append(throttle)

    def get_latest_sensor_value(self, sensor_name):
        """
        指定したセンサーの最新値を返す。
        """
        buf = self.sensor_buffers.get(sensor_name)
        if not buf or len(buf) == 0:
            return None
        return buf[-1]

    def get_latest_all_sensors(self):
        """
        全センサーの最新値をまとめてdictにして返す。
        例: { 'Fr': 123, 'camera_0': <frame>, ... }
        """
        data_dict = {}
        for name in self.sensor_instances.keys():
            data_dict[name] = self.get_latest_sensor_value(name)
        return data_dict
    
    def get_sensor_history(self, sensor_name):
        """
        指定したセンサーの履歴（古い→新しい順にリスト化）を返す
        """
        buf = self.sensor_buffers.get(sensor_name, [])
        return list(buf)

    def get_latest_control(self):
        """
        ステアリング・スロットルの最新値を返す (steering, throttle) のタプル。
        """
        if len(self.steering_buffer) == 0 or len(self.throttle_buffer) == 0:
            return None, None
        return self.steering_buffer[-1], self.throttle_buffer[-1]

    def get_control_history(self):
        """
        ステアリング・スロットルの履歴をまとめて返す (list of (steering, throttle))。
        """
        # たとえば zip() で2つをまとめる
        return list(zip(self.steering_buffer, self.throttle_buffer))

    def cleanup(self):
        """
        センサーのcleanup等を一括で行う場合はこちらで。
        """
        for sensor in self.sensor_instances.values():
            if hasattr(sensor, "cleanup"):
                sensor.cleanup()
        print("DataAggregator cleanup complete.")
