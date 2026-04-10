# run.py
# coding:utf-8

# libcameraのログを抑制（importより前に設定）
import os
os.environ['LIBCAMERA_LOG_LEVELS'] = 'ERROR'

if __name__ == "__main__":
    #　myparam_run.pyで２重にimportされるのを防ぐ
    import config
else:
    #　myparam_run.pyから起動したときにmyparam_run.pyのconfigを使う
    import myparam_run
    config = myparam_run.config

import logging
# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("ライブラリの初期化に数秒かかります...")

import time
from datetime import datetime
from pytz import timezone
jst = timezone('Asia/Tokyo')  # 日本時間のタイムゾーンを取得
import sys
import signal
import atexit
import gc
import shutil

# togikaidriveのモジュール
if config.SIM_MODE:
    import motor_sim as motor
else:
    import motor

from planner import DefaultPlanner
import monitor
from joystick import Joystick, KeyboardController
from pwm_controller import PWMController
from record_manager import RecordManager
from data_aggregator import DataAggregator

# 以下はconfig.pyでの設定によりimport
if "ultrasonic" in config.ACTIVE_SENSORS:
    if config.SIM_MODE:
        import ultrasonic_sim as ultrasonic
    else:
        import ultrasonic

if "camera_0" in config.ACTIVE_SENSORS or "camera_1" in config.ACTIVE_SENSORS:
    if config.SIM_MODE:
        import camera_sim as camera
    else:
        import camera

if "imu" in config.ACTIVE_SENSORS:
    if config.SIM_MODE:
        import imu_sim as imu
    else:
        import imu

if "lidar" in config.ACTIVE_SENSORS:
    import lidar
# if "opticalflow" in config.ACTIVE_SENSORS: import opticalflow

# AIモデルのチェック、利用したいモデルがあればリストに追加する
if config.PLAN in ["nn", "donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]:
    from train_pytorch import NeuralNetwork, ConvolutionalNeuralNetwork, load_model, get_model_from_catalog
    import torch
    
# OpenVINO推論エンジンのインポート
if getattr(config, 'INFERENCE_ENGINE', 'pytorch') == 'openvino':
    from openvino_inference import OpenVINOModel, load_openvino_model
    logger.info("OpenVINO推論エンジンを使用します")

# 位置推論モデルのインポート（必要な場合）
if config.USE_POSITION_SWITCHING:
    try:
        from position_inference import (
            load_position_model,
            load_position_specific_models,
            infer_position
        )
        import sys
        submodule_path = os.path.join(os.path.dirname(__file__), 'annotation_training_d2j')
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
        from model_catalog import get_model as get_location_model
        logger.info("位置推論モデルモジュールをインポートしました")
    except ImportError as e:
        logger.error(f"位置推論モジュールのインポートに失敗: {e}")
        config.USE_POSITION_SWITCHING = False

# YOLO物体検知のインポート（必要な場合）
if config.USE_YOLO_DETECTION:
    try:
        from yolo_detection import (
            load_yolo_model,
            load_yolo_specific_models,
            detect_objects,
            apply_detection_control_modification,
            select_model_by_detection
        )
        from ultralytics import YOLO
        logger.info("YOLOモジュールをインポートしました")
    except ImportError as e:
        logger.error(f"YOLOモジュールのインポートに失敗: {e}")
        logger.error("Ultralyticsをインストールしてください: pip install ultralytics")
        config.USE_YOLO_DETECTION = False


# センサー値やパラメータをブラウザで変更できるmonitorを利用
monitor_thread = None
if config.MONITOR:
    import threading
    monitor_thread = threading.Thread(
        target=monitor.run,
        kwargs={"host": "0.0.0.0", "port": config.MONITOR_PORT, "debug": False},
        daemon=True
    )
    monitor_thread.start()

# --- 初期化 ---
def initialize_system():
    """
    各種設定値の確認、モジュールのインスタンスを初期化する。
    """
    # デバイス検出とプラットフォーム設定
    from device_detection import detect_device
    device_info = detect_device()
    config.DEVICE_TYPE = device_info.device_type
    config.PLATFORM_NAME = device_info.platform_name
    config.GPIO_BACKEND = device_info.gpio_backend
    config.I2C_BUS = device_info.i2c_bus
    logger.info(f"Platform detected: {config.PLATFORM_NAME}, I2C Bus: {config.I2C_BUS}")

    # 選択したプランチェック
    logger.info(f"PLAN: {config.PLAN}")
    if config.PLAN not in config.PLAN_LIST:
        logger.error("Please set plan from %s", config.PLAN_LIST)
        sys.exit()

    # モジュールの初期化
    motor_instance = motor.Motor()

    # 有効なセンサーインスタンスを作成
    active_sensor_instances = {}
    if "ultrasonic" in config.ACTIVE_SENSORS:
        active_sensor_instances.update({
            sensor_name: ultrasonic.Ultrasonic(sensor_name=sensor_name)
            for sensor_name in config.ULTRASONIC_SENSOR_LIST
        })
    if "imu" in config.ACTIVE_SENSORS:
        active_sensor_instances["imu"] = imu.BNO055()
    if "camera_0" in config.ACTIVE_SENSORS:
        print("\n--- カメラ初期化開始 (camera_0) ---")
        active_sensor_instances["camera_0"] = camera.create_camera(device_id=0)
        print("--- カメラ初期化完了 (camera_0) ---\n")
    if "camera_1" in config.ACTIVE_SENSORS:
        print("--- カメラ初期化開始 (camera_1) ---")
        active_sensor_instances["camera_1"] = camera.create_camera(device_id=1)
        print("--- カメラ初期化完了 (camera_1) ---\n")
    if "lidar" in config.ACTIVE_SENSORS:
        print("\n--- LiDAR初期化開始 ---")
        active_sensor_instances["lidar"] = lidar.create_lidar(lidar_type=config.LIDAR_TYPE)
        print("--- LiDAR初期化完了 ---")
        # LiDARのスキャン開始を待つ
        print("LiDARのスキャン開始を待機中...")
        time.sleep(3)
        print("--- LiDAR準備完了 ---\n")
    # if "opticalflow" in config.ACTIVE_SENSORS:
    #     active_sensor_instances["opticalflow"] = opticalflow.Opticalflow()

    # プランナーの初期化
    planner_instance = DefaultPlanner()
    # planner_instance = MyCustomPlanner()

    # モデルの初期ロード
    model = reload_model()

    # ★追加：初期化時に model が None かどうかを明示
    logger.info(
        f"INIT CHECK | PLAN={config.PLAN} | model={'OK' if model is not None else 'None'}"
    )

    # 位置推論システムの初期化
    position_model = None
    position_models_dict = {}
    if config.USE_POSITION_SWITCHING:
        position_model = load_position_model()
        position_models_dict = load_position_specific_models()
        if position_model is None:
            logger.warning("位置推論モデルのロードに失敗しました。通常モードで動作します")
            config.USE_POSITION_SWITCHING = False

    # YOLO物体検知システムの初期化
    yolo_model = None
    yolo_models_dict = {}
    if config.USE_YOLO_DETECTION:
        yolo_model = load_yolo_model()
        yolo_models_dict = load_yolo_specific_models()
        if yolo_model is None:
            logger.warning("YOLOモデルのロードに失敗しました。通常モードで動作します")
            config.USE_YOLO_DETECTION = False

    logger.info("System initialized.")
    return motor_instance, active_sensor_instances, planner_instance, model, position_model, position_models_dict, yolo_model, yolo_models_dict


def reload_model():
    """
    現在の config に基づいてモデルを再ロードする。
    GPUが利用可能な場合は自動的にGPUに配置する。
    """
    model = None

    # ★追加：今の設定を必ず出す（PLAN/MODEL_NAME/MODEL_DIR）
    logger.info(
        f"MODEL CONFIG | PLAN={getattr(config, 'PLAN', None)} | "
        f"MODEL_NAME={getattr(config, 'MODEL_NAME', None)} | "
        f"MODEL_DIR={getattr(config, 'MODEL_DIR', None)}"
    )

    print(config.PLAN, config.MODEL_NAME)

    # MODEL_NAMEがNoneまたは空の場合はモデルをロードしない
    if not config.MODEL_NAME:
        logger.warning(f"MODEL_NAME is not set or empty. Skipping model loading. (PLAN: {config.PLAN})")
        return None

    config.MODEL_PATH = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

    # ★追加：モデルパスを明示
    logger.info(f"MODEL PATH | {config.MODEL_PATH}")

    # モデルファイルの存在確認
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model file not found: {config.MODEL_PATH}")
        logger.info("Please check MODEL_NAME in config.py or train a model first.")
        return None

　　 # --- OpenVINO推論エンジンモード ---
    if getattr(config, 'INFERENCE_ENGINE', 'pytorch') == 'openvino':
        try:
            model = load_openvino_model(config.MODEL_PATH, device_name="CPU")
            model._plan = config.PLAN
            logger.info(f"OpenVINOモデルロード完了: {config.MODEL_NAME} (PLAN: {config.PLAN})")
            return model
        except Exception as e:
            logger.error(f"OpenVINOモデルのロードに失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # デバイスの設定（GPUが利用可能な場合はGPUを使用）
    if config.PLAN in ["nn", "donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}")
        else:
            logger.info("Running on CPU")
    else:
        device = None

    if config.PLAN == "nn":
        input_dim = len(config.ULTRASONIC_SENSOR_LIST)
        output_dim = 2
        model = NeuralNetwork(input_dim, output_dim, config.HIDDEN_DIM, config.NUM_HIDDEN_LAYERS)
        try:
            load_model(model, model_path=config.MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
        # GPUに移動
        if device is not None:
            model = model.to(device)
        logger.info(f"NeuralNetwork model reloaded on {device}")

    elif config.PLAN in ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]:
        # model_catalogからモデルを取得
        try:
            model = get_model_from_catalog(config.PLAN)
            if model is not None:
                try:
                    load_model(model, model_path=config.MODEL_PATH)
                except Exception as e:
                    logger.error(f"Failed to load model weights: {e}")
                    return None
                # GPUに移動
                if device is not None:
                    model = model.to(device)
                logger.info(f"{config.PLAN} model loaded from model_catalog and reloaded from {config.MODEL_PATH} on {device}")
                # 推論に使用するカメラを表示
                if hasattr(config, 'MODEL_INPUT_IMAGE'):
                    logger.info(f"Inference will use camera based on MODEL_INPUT_IMAGE: {config.MODEL_INPUT_IMAGE}")
            else:
                logger.warning(f"Could not create model for PLAN: {config.PLAN}")
                model = None
        except Exception as e:
            logger.error(f"Failed to load model from catalog for {config.PLAN}: {e}")
            model = None
    else:
        logger.warning("PLAN is not supported for model loading. No model reloaded.")

    # GPUメモリ情報を表示
    if model is not None and device is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        logger.info(f"GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

    # ★追加：モデルロード結果を確定ログとして出す（最重要）
    if model is None:
        logger.error("❌ MODEL LOAD RESULT: model is None (ロード失敗 or スキップ)")
    else:
        logger.info(
            f"✅ MODEL LOAD RESULT: model loaded successfully | "
            f"type={type(model)} | path={config.MODEL_PATH}"
        )

    return model


# --- 終了処理 ---
def cleanup_system(motor_instance, planner_instance, active_sensor_instances_dict, controller_instance=None):
    import time
    logger.info("終了処理を開始...")

    # コントローラーのクリーンアップ
    if controller_instance is not None:
        if hasattr(controller_instance, 'close') and callable(controller_instance.close):
            try:
                logger.info("コントローラーをクリーンアップ中...")
                controller_instance.close()
            except Exception as e:
                logger.error(f"Error during controller cleanup: {e}")

    # motor_instanceがNoneでない場合のみモーターを停止
    if motor_instance is not None:
        try:
            motor_instance.set_steering_pwm_value(config.NEUTRAL)
            motor_instance.set_throttle_pwm_value(config.STOP)
            time.sleep(0.1)  # モーター停止を確実にする

            # モーターインスタンスのクリーンアップ
            motor_instance.cleanup()
        except Exception as e:
            logger.error(f"Error during motor cleanup: {e}")

    # 各センサーインスタンスのクリーンアップ（カメラを先にクリーンアップ）
    if active_sensor_instances_dict:
        # カメラインスタンスを先にクリーンアップ
        camera_sensors = [name for name in active_sensor_instances_dict.keys() if name.startswith('camera_')]
        for camera_name in camera_sensors:
            if camera_name in active_sensor_instances_dict:
                sensor = active_sensor_instances_dict[camera_name]
                if hasattr(sensor, 'cleanup') and callable(sensor.cleanup):
                    logger.info(f"Cleaning up {camera_name}...")
                    try:
                        sensor.cleanup()
                        time.sleep(0.1)  # カメラ間のクリーンアップ間隔（短縮）
                    except Exception as e:
                        logger.error(f"Error cleaning up {camera_name}: {e}")

        # その他のセンサーをクリーンアップ
        for sensor_name, sensor in active_sensor_instances_dict.items():
            if not sensor_name.startswith('camera_'):
                if hasattr(sensor, 'cleanup') and callable(sensor.cleanup):
                    try:
                        sensor.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up {sensor_name}: {e}")

    # プランナーインスタンスのクリーンアップ
    if planner_instance is not None:
        try:
            planner_instance.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up planner: {e}")

    # monitorスレッドの終了を待つ
    if config.MONITOR and monitor_thread and monitor_thread.is_alive():
        logger.info("monitorスレッドの終了を待機中...")
        monitor.shutdown_signal = True
        monitor_thread.join(timeout=1.0)  # タイムアウト短縮

    # 最終的なシステムクリーンアップ
    gc.collect()  # ガベージコレクションを強制実行
    logger.info("System cleanup complete.")


def _cleanup_empty_record_folders(record_manager):
    """記録データが空の場合に作成されたフォルダを削除"""
    try:
        if config.SAVE_FORMAT == "donkeycar":
            # Donkeycar形式の場合、data_ディレクトリを削除
            if hasattr(record_manager, 'record_directory') and record_manager.record_directory:
                if os.path.exists(record_manager.record_directory):
                    # フォルダが空または僅かなファイルのみの場合削除
                    files = os.listdir(record_manager.record_directory)
                    if len(files) <= 1:  # manifest.jsonとmeta.jsonのみ、またはimagesフォルダのみ
                        shutil.rmtree(record_manager.record_directory)
                        logger.info(f"空の記録フォルダを削除: {record_manager.record_directory}")
                        # クラス変数もリセット
                        record_manager._current_session_dir = None
        else:
            # CSV/NDJSON形式の場合、作成されたimagesディレクトリを削除
            if hasattr(record_manager, 'image_directory') and record_manager.image_directory:
                if os.path.exists(record_manager.image_directory):
                    files = os.listdir(record_manager.image_directory)
                    if len(files) == 0:  # 完全に空の場合
                        os.rmdir(record_manager.image_directory)
                        logger.info(f"空の画像フォルダを削除: {record_manager.image_directory}")
                        # クラス変数もリセット
                        record_manager._current_images_dir = None
    except Exception as e:
        logger.warning(f"空フォルダ削除中にエラー: {e}")


# グローバル変数（シグナルハンドラ用）
motor_instance = None
active_sensor_instances = None
planner_instance = None
record_manager = None
joystick = None
cleanup_done = False  # 重複終了処理防止フラグ


def signal_handler(sig, frame):
    """シグナルハンドラ（Ctrl-C対応）"""
    global cleanup_done
    if cleanup_done:
        return  # 既に終了処理済み

    cleanup_done = True
    logger.info("\nシグナルを受信しました。終了処理を実行中...")
    if record_manager:
        if hasattr(record_manager, 'records') and len(record_manager.records) > 0:
            record_manager.save_data()
        else:
            # 記録データが空の場合、作成されたフォルダを削除
            _cleanup_empty_record_folders(record_manager)
    if motor_instance and planner_instance and active_sensor_instances:
        cleanup_system(motor_instance, planner_instance, active_sensor_instances, joystick)
    sys.exit(0)


# ============================================================================
# 自動走行のメイン関数
# ============================================================================
if __name__ == "__main__":
    # シグナルハンドラの登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # センサー設定チェック
    has_distance_sensors = "ultrasonic" in config.ACTIVE_SENSORS or "lidar" in config.ACTIVE_SENSORS
    has_camera = "camera_0" in config.ACTIVE_SENSORS or "camera_1" in config.ACTIVE_SENSORS

    # 距離センサーが必要なプラン
    distance_required_plans = ["right_left_3", "right_left_3_records", "wall_follow", "wall_follow_pid", "go_straight", "nn"]
    # カメラのみで動作可能なプラン
    camera_required_plans = ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]
    # 距離センサーとカメラの両方が必要なプラン（カスタムプラン用）
    both_sensors_required_plans = []  # ユーザーがカスタムプランを追加する場合はここに記述
    # センサー不要のマニュアルプラン
    manual_plans = ["manual"]

    try:
        # 初期化
        motor_instance, active_sensor_instances, planner_instance, model, position_model, position_models_dict, yolo_model, yolo_models_dict = initialize_system()
        data_aggregator = DataAggregator(sensor_instances=active_sensor_instances, max_history=10)

        # コントローラーの選択
        controller_type = getattr(config, 'CONTROLLER_TYPE', 'joystick').lower()

        if controller_type == "pwm":
            print("PWMコントローラーモードを使用")
            joystick = PWMController()
        elif controller_type == "keyboard":
            print("キーボードコントローラーモードを使用")
            joystick = KeyboardController()
        elif config.HAVE_JOYSTICK:
            print("ジョイスティックモードを使用")
            joystick = Joystick()
        else:
            # manualプランの場合はキーボード操作
            if config.PLAN in manual_plans:
                print("マニュアルプラン検出: キーボード操作モードを使用")
                joystick = KeyboardController()
            else:
                print("デフォルト: ジョイスティックモードを使用")
                joystick = Joystick()  # 従来通り

        # RecordManagerのインスタンスを作成（セッション全体で一つ）
        record_manager = RecordManager()
        is_recording = False  # 記録中フラグ
        recording_start_time = None  # 記録開始時刻

        if joystick.HAVE_CONTROLLER:
            print(f"{controller_type.upper()}コントローラー検出: userモードで即座に開始")
        else:
            print(f"{controller_type.upper()}コントローラー未接続... autoモードで開始します")
            config.HAVE_JOYSTICK = False

            # センサー設定と走行プランの整合性チェック
            can_auto_drive = False
            missing_sensors = []

            if config.PLAN in distance_required_plans:
                if has_distance_sensors:
                    can_auto_drive = True
                else:
                    missing_sensors.append("測距センサー（ultrasonic/lidar）")
            elif config.PLAN in camera_required_plans:
                if has_camera:
                    can_auto_drive = True
                else:
                    missing_sensors.append("カメラ（camera_0/camera_1）")
            elif config.PLAN in both_sensors_required_plans:
                if has_distance_sensors and has_camera:
                    can_auto_drive = True
                else:
                    if not has_distance_sensors:
                        missing_sensors.append("測距センサー（ultrasonic/lidar）")
                    if not has_camera:
                        missing_sensors.append("カメラ（camera_0/camera_1）")
            elif config.PLAN in manual_plans:
                # マニュアルプランは常に手動操作のみ
                print("マニュアルプラン: 手動操作専用モード")
                can_auto_drive = False
                mode = "user"
                auto_mode_disabled = True  # 自動モード切り替えを無効化
            else:
                # 不明なプラン
                print(f"⚠️  警告: 不明なプラン '{config.PLAN}' です、手動操作（userモード）のみ使用可能です")
                mode = "user"
                auto_mode_disabled = True

            if not can_auto_drive and config.PLAN not in manual_plans:
                print(f"⚠️  プラン '{config.PLAN}' に必要なセンサー: {', '.join(missing_sensors)}")
                print("⚠️  手動操作（userモード）のみ使用可能です、- Sボタンでのautoモード切り替えを無効化しています")
                mode = "user"
                auto_mode_disabled = True
            elif config.PLAN in manual_plans:
                pass
            else:
                print("Sボタンでモード切り替え、Yボタンで記録開始/停止")
                mode = "user"
                auto_mode_disabled = False

            started = True  # 即座に開始

        if not config.HAVE_JOYSTICK:
            # ジョイスティックなしの場合
            if config.PLAN in manual_plans:
                # マニュアルプランの場合はキーボード操作を使用
                print("マニュアルプラン: 手動操作専用モード")
                print("Rキーで記録開始/停止")
                mode = "user"
                started = True
                auto_mode_disabled = True
            else:
                # 自動走行プランの場合の可能性チェック
                can_auto_drive = False
                if config.PLAN in distance_required_plans and has_distance_sensors:
                    can_auto_drive = True
                elif config.PLAN in camera_required_plans and has_camera:
                    can_auto_drive = True

                if not can_auto_drive:
                    print("❌ エラー: 自動走行に必要なセンサーが無効です")
                    if config.PLAN in distance_required_plans:
                        print(f"❌ プラン '{config.PLAN}' には測距センサー（ultrasonic/lidar）が必要です")
                    elif config.PLAN in camera_required_plans:
                        print(f"❌ プラン '{config.PLAN}' にはカメラが必要です")
                    print("❌ ジョイスティックもないため操作不可能です")
                    print("解決方法:")
                    print("1. 必要なセンサーをconfig.pyのACTIVE_SENSORSに追加")
                    print("2. HAVE_JOYSTICKをTrueにしてジョイスティックを使用")
                    print("3. PLAN='manual'にしてキーボード操作を使用")
                    sys.exit(1)

                print("ジョイスティックなし: Enterキー待機中...")
                input("Enterを押して走行開始！")
                mode = "auto"
                started = True
                auto_mode_disabled = False
                # ジョイスティックなしの場合は自動で記録開始
                is_recording = True
                recording_start_time = time.time()
                print("Recording started")

        # ============================================================================
        # メインループ
        # ============================================================================
        start_time = time.time()
        while True:
            # モニター処理
            if config.MONITOR:
                ## 変数再設定（Set Config のフラグが True の場合、再初期化）
                if monitor.set_config_reload:
                    logger.info("Set Config detected. Reinitializing system...")
                    time.sleep(0.1)  # 無限ループ防止のため短いスリープを挿入
                    model = reload_model()
                    monitor.set_config_reload = False  # フラグをリセット
                    continue  # 残りの処理をスキップ

                ## 一時停止
                if monitor.realtime_data["pause_drive"]:
                    logger.info("メインループを一時停止中...")
                    motor_instance.set_steering_pwm_value(config.NEUTRAL)  # モーターを停止
                    motor_instance.set_throttle_pwm_value(config.STOP)    # スロットルを停止
                    time.sleep(0.1)  # 無限ループ防止のため短いスリープを挿入
                    continue  # 残りの処理をスキップ

            # ============================================================================
            # 認知
            # ============================================================================
            ## コントローラの状態確認（コントローラがある場合はmodeを切替えするため）
            if config.HAVE_JOYSTICK:
                joystick.poll()
                mode = joystick.mode[0]

            ## センサー値更新
            data_aggregator.update_sensors()
            ## 必要に応じてセンサーの最新値を取り出す
            sensor_data = data_aggregator.get_latest_all_sensors()

            # 測距センサーデータ（ultrasonicまたはlidar）
            # 注：両方が有効な場合はultrasonicを優先
            ranges = {}
            if "ultrasonic" in config.ACTIVE_SENSORS:
                # Ultrasonicセンサーの場合
                for us_name in config.ULTRASONIC_SENSOR_LIST:
                    ranges[us_name] = data_aggregator.get_latest_sensor_value(us_name)
            elif "lidar" in config.ACTIVE_SENSORS:
                # LiDARの場合、ゾーン別測距データを取得してrangesに格納
                lidar_sensor = active_sensor_instances.get("lidar")
                if lidar_sensor and hasattr(lidar_sensor, 'zone_distances'):
                    # ULTRASONIC_SENSOR_LISTと同じゾーン名を使用
                    for i, zone_name in enumerate(config.ULTRASONIC_SENSOR_LIST):
                        if i < len(lidar_sensor.zone_distances):
                            zone_value = lidar_sensor.zone_distances[i]
                            ranges[zone_name] = zone_value
                            # sensor_dataにも追加（記録用）
                            sensor_data[zone_name] = zone_value
                        else:
                            ranges[zone_name] = 0
                            sensor_data[zone_name] = 0

            camera_image_0 = data_aggregator.get_latest_sensor_value("camera_0") if "camera_0" in config.ACTIVE_SENSORS else None
            camera_image_1 = data_aggregator.get_latest_sensor_value("camera_1") if "camera_1" in config.ACTIVE_SENSORS else None

            ### CNNモデル用のカメラ画像を選択（MODEL_INPUT_IMAGEの設定に基づく）
            inference_camera_image = None
            if config.PLAN in ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]:
                if hasattr(config, 'MODEL_INPUT_IMAGE'):
                    if "cam1" in config.MODEL_INPUT_IMAGE:
                        inference_camera_image = camera_image_1
                    elif "cam0" in config.MODEL_INPUT_IMAGE:
                        inference_camera_image = camera_image_0
                    else:
                        # デフォルトはcamera_0
                        inference_camera_image = camera_image_0
                else:
                    # MODEL_INPUT_IMAGEが未定義の場合はcamera_0を使用
                    inference_camera_image = camera_image_0

            # カメラ画像辞書を作成（planner に渡す用）
            camera_images = {
                'camera_0': camera_image_0,
                'camera_1': camera_image_1
            }

            # ============================================================================
            # 判断（位置推論・YOLO検知・モデル選択はplannerで実施）
            # ============================================================================
            ## 手動運転
            if mode == "user":
                print("🧑 USER MODE: joystick/manual driving")
                steering_value, throttle_value = joystick.steering, joystick.throttle
            ## 自動運転
            else:  # auto
                print("🤖 AUTO MODE: calling planner with model =", "OK" if model else "None")

                # ★追加：planner に渡す直前の入力状況を出す（画像/モデル）
                print(
                    f"PLANNER INPUT | "
                    f"plan={config.PLAN} | "
                    f"model={'OK' if model else 'None'} | "
                    f"img={'YES' if inference_camera_image is not None else 'NO'} | "
                    f"ranges_keys={list(ranges.keys())}"
                )

                steering_value, throttle_value = planner_instance.planning_seaquence(
                    mode,
                    config.PLAN,
                    data_aggregator,
                    model=model if config.PLAN in ["nn", "donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"] else None,
                    inference_camera_image=inference_camera_image,
                    position_model=position_model,
                    position_models_dict=position_models_dict,
                    yolo_model=yolo_model,
                    yolo_models_dict=yolo_models_dict,
                    camera_images=camera_images,
                    ranges=ranges
                )

            if mode == "auto_str":
                throttle_value = joystick.throttle

            # ============================================================================
            # 操作
            # ============================================================================
            motor_instance.set_steering_pwm_value(steering_value)
            motor_instance.set_throttle_pwm_value(throttle_value)

            # ============================================================================
            # 記録
            # ============================================================================
            ## data_aggregator に制御値を追加記録（履歴管理）
            data_aggregator.add_control_data(steering_value, throttle_value)

            ## 外部記録保存
            timestamp = datetime.now(jst).strftime("%Y%m%d%H%M%S%f")

            if config.HAVE_JOYSTICK or config.PLAN in manual_plans:
                # ジョイスティック使用時またはmanualプラン：記録ON/OFFを制御
                if joystick.recording and not is_recording:
                    # 記録開始または再開
                    is_recording = True
                    recording_start_time = time.time()
                    print("*** Recording started/resumed ***")
                elif not joystick.recording and is_recording:
                    # 記録停止
                    is_recording = False
                    recording_start_time = None
                    print("*** Recording stopped ***")

                # 記録中の場合はデータを記録（ブレーキ中も含む）
                if is_recording:
                    record_manager.record_data(timestamp, mode, sensor_data, steering_value, throttle_value)
            else:
                # ジョイスティックなしの自動走行の場合は常に記録
                if is_recording:
                    record_manager.record_data(timestamp, mode, sensor_data, steering_value, throttle_value)

            ## ターミナル出力
            record_count = len(record_manager.records)
            # 経過時間の計算
            elapsed_time = ""
            if recording_start_time:
                elapsed_seconds = int(time.time() - recording_start_time)
                minutes = elapsed_seconds // 60
                seconds = elapsed_seconds % 60
                elapsed_time = f" {minutes:02d}:{seconds:02d}"

            # 位置情報の表示用文字列（plannerから取得）
            position_info = ""
            if config.USE_POSITION_SWITCHING and planner_instance.current_position_id is not None:
                position_name = config.POSITION_CLASS_NAMES[planner_instance.current_position_id] if planner_instance.current_position_id < len(config.POSITION_CLASS_NAMES) else f"Pos{planner_instance.current_position_id}"
                position_info = f" [{position_name}]"

            # 検知情報の表示用文字列（plannerから取得）
            detection_info = ""
            if config.USE_YOLO_DETECTION and config.YOLO_DISPLAY_DETECTIONS and planner_instance.current_detections:
                detected_objects = ", ".join([d["class_name"] for d in planner_instance.current_detections[:3]])  # 最大3つ表示
                detection_info = f" [Det: {detected_objects}]"

            if config.TERMINAL_PRINT and is_recording:
                terminal_output = record_manager.generate_terminal_output(
                    elapsed_time, record_count, mode, steering_value, throttle_value, ranges
                )
                print(terminal_output)
            elif config.TERMINAL_PRINT:
                # 記録停止中または ブレーキ中の簡易出力
                if is_recording and hasattr(joystick, 'is_braking') and joystick.is_braking:
                    status = f"[Braking Records:{record_count}{elapsed_time}]"
                elif is_recording:
                    status = f"[Recording Records:{record_count}{elapsed_time}]"
                else:
                    status = "[Stopped]"
                print(f"{status}{position_info}{detection_info} Mode:{mode}, Steering:{steering_value:.2f}, Throttle:{throttle_value:.2f}, Sensors:{ranges}")

            ## モニター出力
            if config.MONITOR:
                monitor.update_data(
                    mode=mode,
                    steering_value=steering_value,
                    throttle_value=throttle_value,
                    ranges=ranges,
                    timestamp=timestamp,
                    camera_image_0=camera_image_0,
                    camera_image_1=camera_image_1,
                )

    except KeyboardInterrupt:
        logger.info("終了処理を実行中...")
    finally:
        # 終了処理（重複防止）
        if not cleanup_done:
            # 記録データがある場合は保存、空の場合はフォルダを削除
            if record_manager:
                if len(record_manager.records) > 0:
                    record_manager.save_data()
                else:
                    # 記録データが空の場合、作成されたフォルダを削除
                    _cleanup_empty_record_folders(record_manager)
            cleanup_system(motor_instance, planner_instance, active_sensor_instances, joystick)
