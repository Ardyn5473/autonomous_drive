# coding:utf-8
import collections
import numpy as np
import time
import torch
from torchvision import transforms
from PIL import Image
import config
from train_pytorch import normalize_ultrasonics
from position_inference import infer_position
from yolo_detection import detect_objects, apply_detection_control_modification, select_model_by_detection, calculate_object_tracking_steering, calculate_obstacle_avoidance_steering
from follow_the_gap import follow_the_gap
import logging

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self):
        #　判断フラグ
        self.in_recovery = False
        self.before_recovery_detection_times = 3 ## 目前に前壁をtimes回検知
        self.recovery_seconds_remaining = 0
        self.recovery_time_start = time.perf_counter()
        self.recovery_time_end = time.perf_counter()
        self.recovery_time_duration = config.RECOVERY_TIME_DURATION
        self.recovery_frames_remaining = 0

        # 操作値出力
        self.message = ""

        # クラス内で操作値保持
        self.steering = 0.0
        self.throttle= 0.0
        
        # 過去の操作値記録回数
        self.records_steering = np.zeros(config.RIGHT_LEFT_RECORD_NUMBER)
        self.records_throttle = np.zeros(config.RIGHT_LEFT_RECORD_NUMBER)
        
        # pid用のタイマー
        self.time_current = time.perf_counter()
        self.time_before = time.perf_counter()
        
        # pid用の最小距離
        self.minimum_distance_current = config.TARGET_RANGE
        self.minimum_distance_before = config.TARGET_RANGE
        self.integral_delta_distance = 0.0

        # 位置推論関連の状態管理
        self.position_inference_counter = 0
        self.current_position_id = None
        self.current_driving_model = None

        # Follow the Gap用LiDARデータ
        self._lidar_data = None

        # YOLO検知関連の状態管理
        self.yolo_detection_counter = 0
        self.current_detections = []
        self.yolo_active_model = None

        # 時系列モデル推論用の状態
        self._seq_frame_buffer = collections.deque(maxlen=50)
        self._seq_transform = None  # 遅延初期化

    def _select_model_by_position(self, mode, position_model, position_models_dict, camera_images, default_model):
        """
        位置推論によるモデル選択

        Args:
            mode: 走行モード
            position_model: 位置推論モデル
            position_models_dict: 位置別モデル辞書
            camera_images: カメラ画像辞書 {'camera_0': image, 'camera_1': image}
            default_model: デフォルトモデル

        Returns:
            selected_model: 選択されたモデル
        """
        if not config.USE_POSITION_SWITCHING or mode == "user" or position_model is None:
            return default_model

        self.position_inference_counter += 1

        # 指定フレーム間隔で位置推論を実行
        if self.position_inference_counter >= config.POSITION_INFERENCE_INTERVAL:
            self.position_inference_counter = 0

            # 位置推論用の画像を選択
            position_image = None
            if hasattr(config, 'POSITION_MODEL_INPUT_IMAGE') and camera_images:
                for ci in range(4):
                    if f"cam{ci}" in config.POSITION_MODEL_INPUT_IMAGE:
                        position_image = camera_images.get(f'camera_{ci}')
                        break

            # デフォルトはcamera_0
            if position_image is None and camera_images:
                position_image = camera_images.get('camera_0')

            if position_image is not None:
                # 位置を推論
                inferred_position, confidence = infer_position(position_model, position_image)

                if inferred_position is not None:
                    # 位置が変わった場合、ログ出力
                    if inferred_position != self.current_position_id:
                        position_name = config.POSITION_CLASS_NAMES[inferred_position] if inferred_position < len(config.POSITION_CLASS_NAMES) else f"Position{inferred_position}"
                        logger.info(f"位置が変更されました: {position_name} (信頼度: {confidence:.2f})")
                        self.current_position_id = inferred_position

                        # 位置に応じたモデルを選択
                        if inferred_position in position_models_dict:
                            self.current_driving_model = position_models_dict[inferred_position]
                            logger.info(f"モデルを切り替え: 位置{inferred_position}用モデル")
                        elif 'default' in position_models_dict:
                            self.current_driving_model = position_models_dict['default']
                            logger.info(f"デフォルトモデルを使用（位置{inferred_position}用モデルなし）")
                        else:
                            self.current_driving_model = default_model
                            logger.info(f"通常モデルを使用（位置{inferred_position}用モデルなし）")

        # 現在のモデルを返す（位置推論していない場合はデフォルト）
        return self.current_driving_model if self.current_driving_model is not None else default_model

    def _select_model_by_yolo(self, mode, yolo_model, yolo_models_dict, inference_camera_image, default_model):
        """
        YOLO物体検知によるモデル選択

        Args:
            mode: 走行モード
            yolo_model: YOLOモデル
            yolo_models_dict: クラス別モデル辞書
            inference_camera_image: 推論用カメラ画像
            default_model: デフォルトモデル

        Returns:
            selected_model: 選択されたモデル
        """
        if not config.USE_YOLO_DETECTION or mode == "user" or yolo_model is None:
            return default_model

        self.yolo_detection_counter += 1

        # 指定フレーム間隔で物体検知を実行
        if self.yolo_detection_counter >= config.YOLO_DETECTION_INTERVAL:
            self.yolo_detection_counter = 0

            if inference_camera_image is not None:
                # 物体検知を実行
                self.current_detections = detect_objects(yolo_model, inference_camera_image)

                # 検知結果の表示
                if config.YOLO_DISPLAY_DETECTIONS and self.current_detections:
                    detection_summary = ", ".join([
                        f"{d['class_name']}({d['confidence']:.2f})"
                        for d in self.current_detections
                    ])
                    logger.info(f"物体検知: {detection_summary}")

                # モデル切り替え（YOLO_MODEL_SWITCHINGが設定されている場合）
                if config.YOLO_MODEL_SWITCHING and yolo_models_dict:
                    self.yolo_active_model, detected_class = select_model_by_detection(
                        self.current_detections, yolo_models_dict, default_model
                    )
                    if detected_class:
                        logger.info(f"検知によるモデル切り替え: {detected_class['class_name']} (信頼度: {detected_class['confidence']:.2f})")
                        return self.yolo_active_model

        # YOLO検知モデルがある場合はそれを返す、なければデフォルト
        return self.yolo_active_model if self.yolo_active_model is not None else default_model

    def compute_motor_commands(self, mode, plan, ranges, model=None, camera_image=None, data_aggregator=None):
        """
        plan: str (go_straight, right_left_3, nn, donkeycar, resnet18, mobilevit_xxs, edgenext_xx_small, gru, tcn, causal_cnn など)
        ranges: dict {"FrFR": xx, "FrLH": xx, ...}
        model: ニューラルネット用のモデル
        camera_image: 画像ベースモデル用の画像 (numpy配列など) - MODEL_INPUT_IMAGEで指定されたカメラの画像
        data_aggregator: データ集約器（時系列モデルのフレーム履歴用）

        Returns:
            steering_value, throttle_value
        """
        if plan == "go_straight":
            return 0.0, config.FORWARD_STRAIGHT

        elif plan == "right_left_3":
            inputs = (ranges["FrLH"], ranges["FrFR"], ranges["FrRH"])
            return self.right_left_3(*inputs)

        elif plan == "right_left_3_records":
            inputs = (ranges["FrLH"], ranges["FrFR"], ranges["FrRH"])
            return self.right_left_3_records(*inputs)

        elif plan == "wall_follow":
            side = config.HAND_SIDE
            range_front = ranges["FrFR"]
            range_front_side = ranges["FrRH"] if side == "right" else ranges["FrLH"]
            range_rear_side = ranges.get("RrRH", range_front_side) if side == "right" else ranges.get("RrLH", range_front_side)
            return self.wall_follow(range_front, range_front_side, range_rear_side, side)

        elif plan == "wall_follow_pid":
            side = config.HAND_SIDE
            range_front = ranges["FrFR"]
            range_front_side = ranges["FrRH"] if side == "right" else ranges["FrLH"]
            range_rear_side = ranges.get("RrRH", range_front_side) if side == "right" else ranges.get("RrLH", range_front_side)
            return self.wall_follow_pid(range_front, range_front_side, range_rear_side, side)

        elif plan == "nn" and model:
            inputs = [ranges[key] for key in ranges]
            return self.nn(model, *inputs)

        elif plan in ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xx_small"] and model and camera_image is not None:
            return self.model_catalog_inference(model, camera_image)

        elif plan in ["gru", "tcn", "causal_cnn"] and model and camera_image is not None:
            return self.sequence_model_inference(model, camera_image, data_aggregator)

        elif plan == "follow_the_gap":
            lidar_data = self._lidar_data
            if lidar_data is not None:
                return follow_the_gap(lidar_data)
            else:
                logger.warning("follow_the_gap: LiDARデータなし")
                return 0.0, 0.0

        else:
            # その他のプラン or エラー
            print("Please select plan from plan list")
            # ここで例外を出すか、0,0を返す
            return 0.0, 0.0
        
    # 前側１センサーを用いた停止
    def recovery_stop(self, ultrasonic_Fr):
        ## 目前に前壁をtimes回検知
        times = 3
        if max(ultrasonic_Fr.records[0:self.before_recovery_detection_times-1]) < config.STOP_RANGE:
                self.in_recovery = True                
                print("停止")

    # 前側3センサーを用いた後退
    def recovery_back(self, data_aggregator):
        """
        直近の超音波値を取得して後退リカバリを判定する。
        """
        # リカバリー中なら残り時間を更新
        if self.in_recovery:
            self.recovery_time_remaining = self.recovery_time_end - time.perf_counter()
            if self.recovery_time_remaining <= 0:
                # 終了時間に達したらリカバリ解除
                self.in_recovery = False
            else:
                print(f"RECOVERY TIME REMAINING: {self.recovery_time_remaining:.2f}")

        else:
            # 1) 過去 self.before_recovery_detection_times フレーム分のセンサー履歴を取得
            n = self.before_recovery_detection_times

            FrFR_history   = data_aggregator.get_sensor_history("FrFR")    # [古い, ..., 新しい]
            FrRH_history = data_aggregator.get_sensor_history("FrRH")
            FrLH_history = data_aggregator.get_sensor_history("FrLH")

            # 2) 直近N件を切り出し
            recent_FrFR   = FrFR_history[-n:]   # 直近 n 件
            recent_FrRH = FrRH_history[-n:]
            recent_FrLH = FrLH_history[-n:]
            
            # 値が取得されている
            if len(recent_FrFR) > 0 and len(recent_FrRH) > 0 and len(recent_FrLH) > 0:
                #直近の最大値の利用するセンサーの中で最小値
                min_of_max = min(max(recent_FrFR), max(recent_FrRH), max(recent_FrLH))

                if min_of_max < config.BACKWARD_RANGE:
                    self.in_recovery = True
                    self.recovery_time_start = time.perf_counter()
                    self.recovery_time_end   = self.recovery_time_start + config.RECOVERY_TIME_DURATION
                    print("RECOVERY START")

        return self.in_recovery

    # 前側３センサーを用いた右左走行
    def right_left_3(self, dis_FrLH, dis_FrFR, dis_FrRH):
        # 検知時の判断
        ## 壁を検知
        if dis_FrFR < config.DETECTION_RANGE or dis_FrLH < config.RIGHT_LEFT_RANGE or dis_FrRH < config.RIGHT_LEFT_RANGE:
            ### 左＜右の距離
            if dis_FrLH < dis_FrRH :
                self.steering =config.RIGHT
                self.throttle = config.FORWARD_STRAIGHT
                #self.message = "右旋回"
            ### 左＞右の距離
            else:
                self.steering =config.LEFT
                self.throttle = config.FORWARD_CORNER
                #self.message = "左旋回"            
        ## 前壁を検知なし
        else: 
            self.steering =config.NEUTRAL
            self.throttle = config.FORWARD_STRAIGHT
            #self.message = "直進中"

        ## モーターへ出力を返す
        if config.TERMINAL_PRINT:
            print(self.message)
        return self.steering, self.throttle

    # 前側３センサーを用いた右左走行　過去の値でスムージング
    def right_left_3_records(self, dis_FrLH, dis_FrFR, dis_FrRH):
        self.steering, self.throttle  = self.right_left_3(dis_FrLH, dis_FrFR, dis_FrRH)

        # 過去の値を記録の一番前に挿入し、最後を消す
        self.records_steering = np.insert(self.records_steering, 0, self.steering)
        self.records_steering = np.delete(self.records_steering,-1)
        self.records_throttle = np.insert(self.records_throttle, 0, self.throttle)
        self.records_throttle = np.delete(self.records_throttle,-1)

        return round(np.mean(self.records_steering),2), round(np.mean(self.records_throttle),2)

    def _calc_wall_angle(self, d_front_side, d_rear_side, side):
        """
        2点のセンサー距離から壁角度を算出する。
        Returns: wall_angle (rad) - 0=平行, 正=ノーズが壁から離れている
        """
        import math
        sin45 = math.sin(math.radians(45))
        cos45 = math.cos(math.radians(45))

        if side == "right":
            dx = d_front_side * sin45 - d_rear_side
            dy = d_front_side * cos45
        else:  # left
            dx = -d_front_side * sin45 + d_rear_side
            dy = d_front_side * cos45

        wall_angle = math.atan2(dx, dy)
        return wall_angle

    # 壁を用いた走行（右手法・左手法を選択可能）
    def wall_follow(self, dis_front, dis_front_side, dis_rear_side, side="right"):
        """
        壁を用いた走行（右手法・左手法対応）。
        dis_front: 前方センサーからの距離
        dis_front_side: 壁側前方センサーからの距離 (FrRH or FrLH)
        dis_rear_side: 壁側後方センサーからの距離 (RrRH or RrLH)
        side: 壁の位置 ('right' または 'left')
        """
        if side not in ["right", "left"]:
            raise ValueError("Invalid side. Expected 'right' or 'left'.")

        # 壁の距離に基づいた調整
        target_range = config.TARGET_RANGE
        adjustment = config.TARGET_RANGE_ADJUSTMENT

        # 検知時の判断
        ## 壁が遠い場合
        if (dis_front_side > target_range + adjustment) and (dis_rear_side > target_range + adjustment):
            self.steering = config.RIGHT if side == "right" else config.LEFT
            self.throttle = config.FORWARD_CORNER
            self.message = f"{side}手法: 壁が遠い、{side}旋回"

        ## 壁が近い場合
        elif (dis_front_side < target_range - adjustment) or (dis_rear_side < target_range - adjustment):
            self.steering = config.LEFT if side == "right" else config.RIGHT
            self.throttle = config.FORWARD_CORNER
            self.message = f"{side}手法: 壁が近い"

        ## 壁が適切な距離にある場合
        else:
            self.steering = config.NEUTRAL
            self.throttle = config.FORWARD_STRAIGHT
            self.message = f"{side}手法: 壁沿い直進中"

        # 壁角度アライメント補正（WALL_FOLLOW_USE_ALIGNMENT有効時）
        if config.WALL_FOLLOW_USE_ALIGNMENT:
            wall_angle = self._calc_wall_angle(dis_front_side, dis_rear_side, side)
            angle_correction = config.WALL_FOLLOW_K_ANGLE * wall_angle
            # 距離判定がNEUTRAL（適切距離）の場合のみ角度補正を適用
            if self.steering == config.NEUTRAL:
                self.steering = max(-1, min(1, angle_correction))
                if abs(wall_angle) > 0.1:
                    self.throttle = config.FORWARD_CORNER

        # デバッグ用メッセージ出力
        if config.TERMINAL_PRINT:
            print(self.message)

        # モーターへ出力を返す
        return self.steering, self.throttle

    # 壁との距離を一定に保つPID制御走行
    def wall_follow_pid(self, ultrasonic_front, ultrasonic_front_side, ultrasonic_rear_side, side):
        """
        壁との距離を一定に保つPID制御走行。
        side: 壁の位置 ('left' または 'right')
        ultrasonic_front: 前方センサーからの距離データ
        ultrasonic_front_side: 壁側前方センサーからの距離データ (FrRH or FrLH)
        ultrasonic_rear_side: 壁側後方センサーからの距離データ (RrRH or RrLH)
        """

        # 時間更新: 現在の時刻と前回の時刻差を計算
        self.time_before = self.time_current
        self.time_current = time.perf_counter()
        delta_t = self.time_current - self.time_before

        # 壁までの最小距離を計算（壁側の前方センサーと後方センサーの最小値）
        self.minimum_distance_before = self.minimum_distance_current
        self.minimum_distance_current = min(ultrasonic_front_side, ultrasonic_rear_side)

        # 偏差を計算: 現在の最小距離と目標距離（TARGET_RANGE）の差
        delta_dis = self.minimum_distance_current - config.TARGET_RANGE

        # 偏差の積分値を更新: 時間方向に積分することで過去の偏差を考慮
        self.integral_delta_distance += delta_dis

        # 距離変化速度（微分項）を計算
        v = (self.minimum_distance_current - self.minimum_distance_before) / delta_t if delta_t > 0 else 0

        # 壁角度項の追加（WALL_FOLLOW_USE_ALIGNMENT有効時）
        if config.WALL_FOLLOW_USE_ALIGNMENT:
            wall_angle = self._calc_wall_angle(ultrasonic_front_side, ultrasonic_rear_side, side)
            angle_term = config.WALL_FOLLOW_K_ANGLE * wall_angle
        else:
            angle_term = 0.0

        # PID制御でステア値を計算
        # - 比例項 (P): 偏差に比例して制御量を計算
        # - 積分項 (I): 偏差の累積を考慮して制御量を補正
        # - 微分項 (D): 変化速度を考慮してスムーズな制御を実現
        # - 壁角度項: 壁との平行度を補正
        steering_gain = config.K_P * delta_dis - config.K_D * v + config.K_I * self.integral_delta_distance + angle_term

        # ステアゲイン値を0 ~ 1に変換
        steering_gain = max(-1, min(1, steering_gain))

        # デバッグ用の出力: PID制御の各項目を出力
        if config.TERMINAL_PRINT:
            self._print_pid_debug(side, steering_gain, delta_dis, self.integral_delta_distance, v)

        # 左右の壁に応じた走行ロジックを実行
        if side == "right":
            self.steering = steering_gain * config.RIGHT
        elif side == "left":
            self.steering = steering_gain * config.LEFT
        else:
            raise ValueError("Invalid side. Expected 'left' or 'right'.")

        # スロットル値も調整
        if abs(self.steering) > 0.7:
            self.throttle = config.FORWARD_CORNER
        else:
            self.throttle = config.FORWARD_STRAIGHT

        # 計算結果を返す: ステアリング値とスロットル値
        return round(self.steering,2), round(self.throttle,2)


    # デバッグ用の補助関数
    def _print_pid_debug(self, side, steering, delta_dis, integral_delta_distance, v):
        side_text = "右手法" if side == "right" else "左手法"
        print(
            f"{side_text} PID制御: "
            f"output={steering:.2f}, [P={config.K_P * delta_dis:.2f}, "
            f"I={config.K_I * integral_delta_distance:.2f}, D={config.K_D * v:.2f}]"
        )

    # 右手法のPIDを用いた走行
    ## TODO:wall_followへ移行、削除予定
    def right_hand_pid(self, ultrasonic_FrRH, ultrasonic_RrRH,
        t=0, integral_delta_dis=0, min_dis=config.TARGET_RANGE):
        # 時間更新
        t_before = t
        t = time.perf_counter()
        delta_t = t-t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrRH, ultrasonic_RrRH)
        # 目標値までの差更新
        delta_dis = min_dis - self.TARGET_RANGE
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
         #速度更新
        v = (min_dis - min_dis_before)/delta_t
        # PID制御でステア値更新
        steering = self.K_P*delta_dis - self.K_D*v + self.K_I*integral_delta_dis 
        ### -100~100に収めて正の割合化
        steering = abs(max(-100,min(100,steering))/100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            #print(self.message)
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steering, self.K_P*delta_dis,self.K_D*v, self.K_I*integral_delta_dis))
        self.steering, self.throttle  = self.right_hand(ultrasonic_FrRH.dis, ultrasonic_RrRH.dis)
        return steering*self.steering, self.throttle

    # 左手法のPIDを用いた走行
    ## TODO:wall_followへ移行、削除予定
    def left_hand_pid(self, ultrasonic_FrLH, ultrasonic_RrLH,
        t=0,integral_delta_dis=0,min_dis=config.TARGET_RANGE):
        # 時間更新
        t_before = t
        t = time.perf_counter()
        delta_t = t-t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrLH.dis,ultrasonic_RrLH.dis)
        # 目標値までの差更新
        delta_dis = min_dis - self.TARGET_RANGE
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
         #速度更新
        v = (min_dis - min_dis_before)/delta_t
        # PID制御でステア値更新
        steering = self.K_P*delta_dis - self.K_D*v + self.K_I*integral_delta_dis 
        ### -100~100に収めて正の割合化
        steering = abs(max(-100,min(100,steering))/100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            #print(self.message)
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steering, self.K_P*delta_dis,self.K_D*v, self.K_I*integral_delta_dis))
        self.steering, self.throttle  = self.left_hand(ultrasonic_FrLH.dis, ultrasonic_RrLH.dis)
        return steering*self.steering, self.throttle

    # Neural Netを用いた走行
    def nn(self, model, *args):
        ultrasonic_values = args
        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        x = torch.tensor(ultrasonic_values, dtype=model_dtype).unsqueeze(0)

        # モデルに正規化パラメータがある場合（data_viewer形式）
        norm_params = getattr(model, '_normalization_params', None)
        if norm_params:
            norm_type = norm_params.get('type', 'zscore')
            if norm_type == 'clip_scale':
                clip_val = norm_params.get('clip_max', 2000.0)
                x = torch.clamp(x, 0, clip_val) / clip_val
            elif 'X_mean' in norm_params and 'X_std' in norm_params:
                mean = torch.tensor(norm_params['X_mean'], dtype=model_dtype)
                std = torch.tensor(norm_params['X_std'], dtype=model_dtype)
                x = (x - mean) / (std + 1e-8)
        else:
            # 従来の正規化（train_pytorch形式）
            x = normalize_ultrasonics(x)

        x = x.to(device)

        # data_viewer形式はforward直接、train_pytorch形式はpredict
        with torch.no_grad():
            if hasattr(model, 'predict') and norm_params is None:
                output = model.predict(model, x).squeeze(0)
            else:
                output = model(x).squeeze(0)

        self.steering = float(output[0])
        self.throttle = float(output[1])

        ## モーターへ出力を返す
        return self.steering, self.throttle
    
    def sequence_model_inference(self, model, img, data_aggregator):
        """
        時系列モデル（GRU/TCN/CausalCNN）を使用して推論を行う。
        フレームバッファに画像を蓄積し、シーケンスとして推論する。
        """
        seq_cfg = getattr(model, '_sequence_config', {})
        seq_len = seq_cfg.get('seq_len', 8)
        img_size = seq_cfg.get('img_size', (128, 128))
        num_sources = seq_cfg.get('num_image_sources', 1)

        # transformの遅延初期化
        if self._seq_transform is None:
            self._seq_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # 現在のフレームをバッファに追加
        self._seq_frame_buffer.append(img)

        # seq_len分のフレームを取得（不足時は最古のフレームで埋める）
        buf = list(self._seq_frame_buffer)
        while len(buf) < seq_len:
            buf.insert(0, buf[0])
        frames = buf[-seq_len:]

        # ego_states: 制御値履歴から構築 [steering, throttle, 0, 0, 0]
        ego_list = []
        if data_aggregator is not None:
            control_history = data_aggregator.get_control_history()
        else:
            control_history = []
        # 不足分をゼロ埋め
        while len(control_history) < seq_len:
            control_history.insert(0, (0.0, 0.0))
        control_history = control_history[-seq_len:]

        try:
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype

            # 画像テンソル構築: (1, T, S, C, H, W)
            img_tensors = []
            for frame in frames:
                pil_img = Image.fromarray(frame)
                t = self._seq_transform(pil_img)  # (C, H, W)
                img_tensors.append(t)
            # (T, C, H, W) -> (T, 1, C, H, W) for S=1
            images = torch.stack(img_tensors).unsqueeze(1)  # (T, S=1, C, H, W)
            images = images.unsqueeze(0).to(device, dtype=model_dtype)  # (1, T, S, C, H, W)

            # ego_states テンソル構築: (1, T, 5)
            ego_np = np.array([[s, t, 0.0, 0.0, 0.0] for s, t in control_history], dtype=np.float32)
            ego_states = torch.from_numpy(ego_np).unsqueeze(0).to(device, dtype=model_dtype)  # (1, T, 5)

            with torch.no_grad():
                trajectory = model(images, ego_states)  # (1, pred_horizon, 2)

            # 最初のステップの予測値を使用
            self.steering = float(trajectory[0, 0, 0].item())
            self.throttle = float(trajectory[0, 0, 1].item())

        except Exception as e:
            logger.error(f"Sequence model inference error: {e}")
            import traceback
            traceback.print_exc()
            self.steering = 0.0
            self.throttle = 0.0

        return self.steering, self.throttle

    def model_catalog_inference(self, model, img):
        """
        model_catalog のモデル（donkey, resnet18, mobilevit_xxs, edgenext_xx_small）を使用して推論を行う
        推論エンジン（PyTorch, TensorRT, OpenVINO）に応じて適切な推論方法を選択
        """
        # 入力データが numpy.ndarray であることを確認
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Input img must be a numpy.ndarray, but got {type(img)}")

        try:
            # 推論エンジンに応じて処理を分岐
            if config.INFERENCE_ENGINE == "tensorrt":
                output = self._tensorrt_inference(model, img)
            elif config.INFERENCE_ENGINE == "openvino":
                output = self._openvino_inference(model, img)
            else:  # pytorch (default)
                output = self._pytorch_inference(model, img)
            
            # 出力の処理
            self._process_model_output(output)
            
        except Exception as e:
            print(f"Model inference error: {e}")
            import traceback
            traceback.print_exc()
            self.steering = 0.0
            self.throttle = 0.0

        # モーターへ出力を返す
        return self.steering, self.throttle

    def _pytorch_inference(self, model, img):
        """PyTorchモデルでの推論"""
        if hasattr(model, 'run'):
            # model_catalogのモデルの.run()メソッドはnumpy配列を期待
            return model.run(img)
        else:
            # フォールバック: この分岐は実際には使用されない（Donkeyモデルは常にrunメソッドを持つ）
            raise RuntimeError("Model must have 'run' method for proper image preprocessing")

    def _tensorrt_inference(self, model, img):
        """TensorRTモデルでの推論"""
        try:
            # TensorRTモデルの場合
            if hasattr(model, 'run'):
                return model.run(img)
            else:
                raise RuntimeError("TensorRT model must have 'run' method for proper image preprocessing")

        except Exception as e:
            print(f"TensorRT inference failed, falling back to PyTorch: {e}")
            return self._pytorch_inference(model, img)

    def _openvino_inference(self, model, img):
        """OpenVINOモデルでの推論"""
        try:
            # model_catalogのOpenVINOラッパーを使用（runメソッドがあればそちらを優先）
            if hasattr(model, 'run'):
                return model.run(img)
            
            # OpenVINOModel（__call__対応）の場合: 前処理してから呼び出す
            if hasattr(model, 'compiled_model'):
                # numpy画像を推論用に前処理
                input_img = img.copy()
                # HWC → CHW変換
                if input_img.ndim == 3 and input_img.shape[2] in (1, 3):
                    input_img = np.transpose(input_img, (2, 0, 1))
                # 正規化 (0-255 → 0-1)
                if input_img.max() > 1.0:
                    input_img = input_img.astype(np.float32) / 255.0
                # バッチ次元追加
                if input_img.ndim == 3:
                    input_img = np.expand_dims(input_img, axis=0)
                
                result = model(input_img)
                # OpenVINOInferenceResult → numpy
                if hasattr(result, 'data'):
                    output = result.data
                else:
                    output = np.array(result)
                
                # 出力をステアリング・スロットルのタプルに変換
                if output.ndim > 1:
                    output = output.flatten()
                if len(output) >= 2:
                    return (float(output[0]), float(output[1]))
                else:
                    return (float(output[0]), 0.0)
            
            raise RuntimeError("OpenVINOモデルに対応する推論メソッドが見つかりません")
            
        except Exception as e:
            print(f"OpenVINO inference failed: {e}")
            import traceback
            traceback.print_exc()
            self.steering = 0.0
            self.throttle = 0.0
            return (0.0, 0.0)

    def _process_model_output(self, output):
        """モデル出力を処理してステアリング・スロットル値を設定"""
        # outputがタプルの場合（angle, throttle）
        if isinstance(output, tuple) and len(output) == 2:
            self.steering = float(output[0])
            self.throttle = float(output[1])
        # outputがTensorの場合
        elif torch.is_tensor(output):
            if output.dim() > 0:
                self.steering = float(output[0].item())
                self.throttle = float(output[1].item()) if len(output) > 1 else 0.0
            else:
                self.steering = float(output.item())
                self.throttle = 0.0
        # outputがnumpy配列の場合
        elif isinstance(output, np.ndarray):
            if output.size > 1:
                self.steering = float(output[0])
                self.throttle = float(output[1]) if len(output) > 1 else 0.0
            else:
                self.steering = float(output.item())
                self.throttle = 0.0
        else:
            print(f"Unexpected output format: {type(output)}")
            self.steering = 0.0
            self.throttle = 0.0

    def cleanup(self):
        print("Planner cleanup complete.")
        pass

# imuを用いた走行制御
class DynamicControl:
    def __init__(self, mode=None):
        self.gain_steering = 1.0
        self.gain_throttle = 1.0

    def update_control(self, throttle_gain, steering_gain):
        """動的制御のゲインを更新"""
        self.gain_throttle = throttle_gain
        self.gain_steering = steering_gain

    def counter_steering(self, gyro_data, steering, throttle):
        """
        カウンターステア強度を計算する関数

        Args:
            gyro_data (dict): ジャイロデータを格納した辞書。キー "z" に回転速度のリストが含まれることを想定。
            steering (float): 現在のステアリング値。
            throttle (float): 現在のスロットル値。

        Returns:
            tuple: 調整後のステアリング値、スロットル値。
        """
        # "z"キーが存在しない、またはリストが空の場合に例外を投げる
        if "z" not in gyro_data or not gyro_data["z"]:
            raise ValueError("gyro_data に 'z' キーが存在しないか、リストが空です。")

        # z軸の回転速度の平均を計算
        average_rotation_speed = abs(sum(gyro_data["z"]) / len(gyro_data["z"]))

        # カウンターステア強度を計算し、1を超えないように制限
        counter_steering_strength = min(1, average_rotation_speed / self.rotation_speed)

        # ステアリング値にカウンターステア強度を適用
        adjusted_steering = steering * (1 - counter_steering_strength)

        # スロットル値をそのまま返却（必要に応じて調整可能）
        adjusted_throttle = throttle

        return adjusted_steering, adjusted_throttle

    def lateral_g_throttle(self, acc_data, jerk_data, steering, throttle):
        """
        横Gスロットル制御を計算する関数

        Args:
            acc_data (dict): 加速度データを格納した辞書。キー "y" にy軸方向のデータが含まれることを想定。
            jerk_data (dict): ジャーク（加速度の時間微分）データを格納した辞書。キー "y" にy軸方向のデータが含まれることを想定。
            steering (float): 現在のステアリング値。
            throttle (float): 現在のスロットル値。

        Returns:
            tuple: 調整後のステアリング値とスロットル値 (steering, throttle)。
        """
        # "y"キーが存在しない、またはリストが空の場合に例外を投げる
        if "y" not in acc_data or not acc_data["y"]:
            raise ValueError("acc_data に 'y' キーが存在しないか、リストが空です。")
        if "y" not in jerk_data or not jerk_data["y"]:
            raise ValueError("jerk_data に 'y' キーが存在しないか、リストが空です。")

        # 最新のy軸加速度とジャークの値を取得
        last_acc_y = acc_data["y"][-1]
        last_jerk_y = jerk_data["y"][-1]

        # 横Gスロットル制御量を計算
        lateral_g_control = abs((last_acc_y * last_jerk_y) * self.Cxy / (1 + self.Ts) * abs(last_jerk_y))

        # スロットル値を横Gスロットル制御量に基づいて制限
        adjusted_throttle = min(1, lateral_g_control)

        # ステアリング値はそのまま返す
        adjusted_steering = steering

        return adjusted_steering, adjusted_throttle

class LapCounter:
    def __init__(self):
        self.current_lap = 0
        self.last_checkpoint_time = None

    def increment_lap(self):
        """周回数を1増加させる"""
        self.current_lap += 1
        print(f"Lap incremented: {self.current_lap}")

    def reset_lap(self):
        """周回数をリセット"""
        self.current_lap = 0

    def get_lap_count(self):
        """現在の周回数を取得"""
        return self.current_lap
    
# TODO:CustompPlanの相談
class MyCustomPlanner(Planner):
    pass
        
class DefaultPlanner(Planner):
    def __init__(self):
        super().__init__()
    
    def planning_seaquence(self, mode, plan, data_aggregator, model, inference_camera_image=None,
                          position_model=None, position_models_dict=None,
                          yolo_model=None, yolo_models_dict=None,
                          camera_images=None, ranges=None, lidar_data=None):
        """
        判断シーケンス（モデル選択含む）

        Args:
            mode: 走行モード
            plan: プラン名
            data_aggregator: データ集約器
            model: 基本モデル
            inference_camera_image: 推論用カメラ画像
            position_model: 位置推論モデル
            position_models_dict: 位置別モデル辞書
            yolo_model: YOLOモデル
            yolo_models_dict: クラス別モデル辞書
            camera_images: カメラ画像辞書
            ranges: 測距センサーデータ（位置名: 距離値の辞書、ultrasonic/lidar共通）

        Returns:
            steering_value, throttle_value: 制御値
        """
        # 最優先: リカバリー状態に入るか確認（早期リターン）
        if config.RECOVERY_MODE == "back" and mode != "user":
            if self.recovery_back(data_aggregator):
                return config.LEFT, config.REVERSE

        # モデル選択（自動運転モード時のみ）
        active_model = model
        if mode != "user":
            # 位置推論によるモデル選択
            active_model = self._select_model_by_position(
                mode, position_model, position_models_dict, camera_images, active_model
            )

            # YOLO検知によるモデル選択（優先）
            active_model = self._select_model_by_yolo(
                mode, yolo_model, yolo_models_dict, inference_camera_image, active_model
            )

        # Follow the Gap用LiDARデータ
        if lidar_data is not None:
            self._lidar_data = lidar_data
        elif "lidar" in getattr(config, 'ACTIVE_SENSORS', []):
            self._lidar_data = data_aggregator.get_latest_sensor_value("lidar")

        # 測距センサーデータ（run.pyから渡される、既にマッピング済み）
        # rangesがNoneの場合は後方互換のためdata_aggregatorから取得
        if ranges is None:
            ranges = {}
            for sensor_position in config.ULTRASONIC_SENSOR_LIST:
                ranges[sensor_position] = data_aggregator.get_latest_sensor_value(sensor_position)

        # 制御値を計算（選択されたモデルを使用）
        steering_value, throttle_value = self.compute_motor_commands(
            mode, plan, ranges, active_model, inference_camera_image,
            data_aggregator=data_aggregator
        )

        # YOLO検知による制御値修正（自動運転時のみ）
        if config.USE_YOLO_DETECTION and mode != "user" and self.current_detections:
            # 1. 障害物回避制御（最優先：ステアリング補正）
            obstacle_avoidance_applied = False
            if config.USE_YOLO_OBSTACLE_AVOIDANCE:
                avoidance_steering, obstacle_info = calculate_obstacle_avoidance_steering(
                    self.current_detections, config.IMAGE_W, config.IMAGE_H
                )
                if obstacle_info:
                    steering_value += avoidance_steering
                    # 範囲制限
                    steering_value = max(-1.0, min(1.0, steering_value))
                    obstacle_avoidance_applied = True
                    if config.YOLO_DISPLAY_DETECTIONS:
                        logger.info(
                            f"障害物回避: {obstacle_info['class_name']} "
                            f"(信頼度: {obstacle_info['confidence']:.2f}, "
                            f"サイズ比: {obstacle_info['area_ratio']:.2%}, "
                            f"回避方向: {obstacle_info['avoidance_direction']}, "
                            f"補正: {obstacle_info['steering_offset']:.2f})"
                        )

            # 2. 物体追従制御（障害物回避が適用されていない場合のみ）
            if config.USE_YOLO_OBJECT_TRACKING and not obstacle_avoidance_applied:
                steering_offset, tracking_info = calculate_object_tracking_steering(
                    self.current_detections, config.IMAGE_W
                )
                if tracking_info:
                    steering_value += steering_offset
                    # 範囲制限
                    steering_value = max(-1.0, min(1.0, steering_value))
                    if config.YOLO_DISPLAY_DETECTIONS:
                        logger.info(
                            f"物体追従: {tracking_info['class_name']} "
                            f"(信頼度: {tracking_info['confidence']:.2f}, "
                            f"オフセット: {tracking_info['offset']:.2f}, "
                            f"補正: {tracking_info['steering_offset']:.2f})"
                        )

            # 3. YOLO制御ルール適用（スロットル修正等）
            modified_steering, modified_throttle, applied_rule = apply_detection_control_modification(
                self.current_detections, steering_value, throttle_value
            )
            if applied_rule:
                if config.YOLO_DISPLAY_DETECTIONS:
                    logger.info(f"制御修正適用: {applied_rule['description']} ({applied_rule['class_name']}, {applied_rule['confidence']:.2f})")
                steering_value = modified_steering
                throttle_value = modified_throttle

        return steering_value, throttle_value

# ROS2の有無を判定してインポート
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Bool, String, Float32MultiArray
    from geometry_msgs.msg import Twist

    class PlannerNode(Node):
        def __init__(self):
            super().__init__('planner_node')

            # DefaultPlannerインスタンス化
            self.planner = DefaultPlanner()

            # 状態変数
            self.mode = "user"
            self.joystick_steering = 0.0
            self.joystick_throttle = 0.0
            self.ranges = {}
            self.steering = 0.0
            self.throttle = 0.0

            # サブスクライバー
            self.create_subscription(String, '/joy/mode', self.mode_callback, 10)
            self.create_subscription(Twist, '/cmd_vel_joy', self.joy_cmd_callback, 10)
            self.create_subscription(Float32MultiArray, '/ultrasonic_data', self.ultrasonic_callback, 10)

            # パブリッシャー（統一: /cmd_vel）
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

            # タイマーで定期的にプランニング実行
            self.timer = self.create_timer(0.05, self.planning_loop)

            self.get_logger().info(f"Planner node started (plan={config.PLAN})")

        def mode_callback(self, msg):
            self.mode = msg.data

        def joy_cmd_callback(self, msg):
            self.joystick_steering = msg.angular.z
            self.joystick_throttle = msg.linear.x

        def ultrasonic_callback(self, msg):
            for i, name in enumerate(config.ULTRASONIC_SENSOR_LIST):
                if i < len(msg.data):
                    self.ranges[name] = msg.data[i]

        def planning_loop(self):
            cmd = Twist()

            if self.mode == "user":
                cmd.angular.z = float(self.joystick_steering)
                cmd.linear.x = float(self.joystick_throttle)
            else:
                # 自動モード: DefaultPlannerで計算
                steering, throttle = self.planner.compute_motor_commands(
                    self.mode, config.PLAN, self.ranges)
                cmd.angular.z = float(steering)
                cmd.linear.x = float(throttle)

            self.cmd_pub.publish(cmd)

    def main_ros(args=None):
        rclpy.init(args=args)
        node = PlannerNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

except ImportError:
    # print("ROS2関連ライブラリがインストールされていません。ROS2モードは無効です。")
    rclpy = None

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Planning only with ROS2")
    parser.add_argument('--ros', action='store_true', help="Run with ROS2 node")
    args = parser.parse_args()

    if args.ros and rclpy:
        print("Start with ROS2")
        main_ros()
    else:
        # right_left_3のステアリング値計算テスト
        print("=" * 60)
        print("right_left_3 ステアリング値計算テスト")
        print("=" * 60)
        print("3つのセンサー値を入力してステアリング値を計算します")
        print("終了するには Ctrl+C を押してください\n")
        
        planner = Planner()
        
        try:
            while True:
                print("-" * 40)
                # センサー値のテストパターン
                test_cases = [
                    {"FrLH": 300, "FrFR": 500, "FrRH": 600, "name": "左壁接近"},
                    {"FrLH": 600, "FrFR": 500, "FrRH": 300, "name": "右壁接近"},
                    {"FrLH": 400, "FrFR": 200, "FrRH": 400, "name": "前方障害物"},
                    {"FrLH": 800, "FrFR": 800, "FrRH": 800, "name": "障害物なし"},
                    {"FrLH": 200, "FrFR": 600, "FrRH": 700, "name": "左壁非常に接近"},
                ]
                
                print("テストパターンを選択:")
                for i, case in enumerate(test_cases, 1):
                    print(f"{i}. {case['name']} (左:{case['FrLH']}mm, 前:{case['FrFR']}mm, 右:{case['FrRH']}mm)")
                print("6. カスタム値を入力")
                
                choice = input("\n選択 (1-6): ").strip()
                
                if choice in ['1', '2', '3', '4', '5']:
                    idx = int(choice) - 1
                    dis_FrLH = test_cases[idx]["FrLH"]
                    dis_FrFR = test_cases[idx]["FrFR"]
                    dis_FrRH = test_cases[idx]["FrRH"]
                    print(f"\n選択: {test_cases[idx]['name']}")
                elif choice == '6':
                    try:
                        dis_FrLH = float(input("左センサー値 (FrLH) [mm]: "))
                        dis_FrFR = float(input("前センサー値 (FrFR) [mm]: "))
                        dis_FrRH = float(input("右センサー値 (FrRH) [mm]: "))
                    except ValueError:
                        print("無効な入力です。数値を入力してください。")
                        continue
                else:
                    print("無効な選択です。")
                    continue
                
                # right_left_3メソッドを呼び出してステアリング値を計算
                steering, throttle = planner.right_left_3(dis_FrLH, dis_FrFR, dis_FrRH)
                
                # 結果を表示
                print(f"\n【センサー値】")
                print(f"  左(FrLH): {dis_FrLH:6.1f} mm")
                print(f"  前(FrFR): {dis_FrFR:6.1f} mm")
                print(f"  右(FrRH): {dis_FrRH:6.1f} mm")
                print(f"\n【計算結果】")
                print(f"  ステアリング: {steering:6.2f} ", end="")
                if steering < 0:
                    print("(左旋回)")
                elif steering > 0:
                    print("(右旋回)")
                else:
                    print("(直進)")
                print(f"  スロットル:   {throttle:6.2f}")
                
                # 判定ロジックの説明
                print(f"\n【判定理由】")
                if dis_FrFR < config.DETECTION_RANGE:
                    print(f"  前方に障害物検知 (前センサー {dis_FrFR}mm < {config.DETECTION_RANGE}mm)")
                    if dis_FrLH < dis_FrRH:
                        print(f"  左側が近い ({dis_FrLH}mm < 右{dis_FrRH}mm) → 右旋回")
                    else:
                        print(f"  右側が近い ({dis_FrRH}mm <= 左{dis_FrLH}mm) → 左旋回")
                elif dis_FrLH < config.RIGHT_LEFT_RANGE:
                    print(f"  左壁に接近 (左センサー {dis_FrLH}mm < {config.RIGHT_LEFT_RANGE}mm) → 右旋回")
                elif dis_FrRH < config.RIGHT_LEFT_RANGE:
                    print(f"  右壁に接近 (右センサー {dis_FrRH}mm < {config.RIGHT_LEFT_RANGE}mm) → 左旋回")
                else:
                    print(f"  障害物なし → 直進")
                
                input("\nEnterキーを押して続行...")
                
        except KeyboardInterrupt:
            print("\n\nテストを終了します。")

