# config.py
# coding:utf-8
import os

# ============================================================================
# デバイス設定
# ============================================================================
# デバイス関連設定（実際の検出はrun.pyで実行）
# デフォルト値（run.pyでdevice_detection.pyにより上書きされる）
DEVICE_TYPE = "UNKNOWN"
PLATFORM_NAME = "Unknown"
GPIO_BACKEND = "gpiozero"
I2C_BUS = 7

# 使用するセンサー
#ACTIVE_SENSORS = ["lidar"] # Nagoya講座用
ACTIVE_SENSORS = ["lidar","camera_0"] 
#ACTIVE_SENSORS = ["ultrasonic","camera_0"]  #"ultrasonic","lidar"
#ACTIVE_SENSORS = ["camera_0", "camera_1"]
#ACTIVE_SENSORS = ["ultrasonic","camera_0", "camera_1"]
#ACTIVE_SENSORS = ["ultrasonic", "camera_0", "camera_1", "imu", "lidar", "opticalflow"]

# ============================================================================
# モーター制御基本設定
# ============================================================================
# モーターへの入力値 （-1~1で設定）
## 短時間の講座で利用
## 一定スロットル出力用
FORWARD_STRAIGHT = 0.4 #ストレートでの値, joy_accel1
FORWARD_CORNER = 0.3 #カーブでのの値, joy_accel2
STOP = 0
REVERSE = -1
## 一定ステアリング出力用
LEFT = -1
NEUTRAL = 0
RIGHT = 1

# ============================================================================
# 超音波センサ/LiDAR 検知範囲設定
# ============================================================================
# 超音波センサの検知範囲設定（単位: mm）
# 停止範囲（汎用）
STOP_RANGE = 250  # 停止判断に使用する汎用範囲
BACKWARD_RANGE = 130  # 後退判断に使用する範囲

# 障害物回避の基準距離
DETECTION_RANGE = 300  # 検知開始距離
RIGHT_LEFT_RANGE = 550  # 右左折判定基準距離

# 右手法/左手法での目標範囲
TARGET_RANGE = 200  # 右手法/左手法で共有する目標距離
TARGET_RANGE_ADJUSTMENT = 25  # 目標距離近辺で操作変更を実施する基準値（±）

## PIDパラメータ(PDまでを推奨)
K_P = 0.005 #0.005
K_I = 0.0 #0.00001
K_D = 0.0005 #0.0005

# ============================================================================
# 走行プラン（判断モード）選択
# ============================================================================
PLAN_LIST = [
    "manual",
    "go_straight",
    "right_left_3",
    "right_left_3_records",
    "wall_follow",
    "wall_follow_pid",
    "nn",
    "donkeycar",
    "resnet18",
    "mobilevit_xxs",
    "edgenext_xxsmall"
]

PLAN = "nn"

# ============================================================================
# 各種走行モード固有のパラメータ
# ============================================================================
# wall_follow モード関連パラメータ
HAND_SIDE = "right" #"left"

# right_left_3_records モード関連パラメータ
## 過去の操作値記録回数
RIGHT_LEFT_RECORD_NUMBER = 3

# ============================================================================
# 復帰モード設定
# ============================================================================
RECOVERY_MODE = "back" #none, back
RECOVERY_STREERING = LEFT # 復帰時のステアリング値
RECOVERY_TIME_DURATION = 1 #復帰処理を行う時間（秒）
RECOVERY_BRAKING = 1 #ブレーキ回数、ブレーキにはReverseを利用

# ============================================================================
# 車両調整用パラメータ（ハードウェア設定）
# ============================================================================
# 車両調整用パラメータ(motor.pyで調整した後値を入れる)

## RCのPWM信号のチャネル
CHANNEL_STEERING = 0
CHANNEL_THROTTLE = 1 

## ステアリングのPWMの値
STEERING_CENTER_PWM = 390 #400~420:newcar, #340~360:oldcar
STEERING_WIDTH_PWM = 80
STEERING_RIGHT_PWM = STEERING_CENTER_PWM - STEERING_WIDTH_PWM #STEERING_CENTER_PWM - STEERING_WIDTH_PWM
STEERING_LEFT_PWM =  STEERING_CENTER_PWM + STEERING_WIDTH_PWM #STEERING_CENTER_PWM + STEERING_WIDTH_PWM
### !!!ステアリングを壊さないための上限下限の値設定  
STEERING_HI_LIMIT = 500
STEERING_LO_LIMIT = 300


## アクセルのPWM値
## モーターの回転音を聞き、音が変わらないところが最大/最小値とする
THROTTLE_STOPPED_PWM = 380 #めやす：370~400
THROTTLE_FORWARD_PWM = 500
THROTTLE_REVERSE_PWM = 280
THROTTLE_WIDTH_PWM = 100  #motor.pyの確認用

# ============================================================================
# 機械学習モデル設定（NN/CNN）
# ============================================================================
## モデルのパス
MODEL_DIR = "models"
#MODEL_NAME = "nn_20250926_012842_20250926_0127_record"
MODEL_NAME = "nn_20260313_230707_5_64_64_2.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

## 推論エンジンの選択
# すべてのモデル（自動運転、位置推論、YOLO）に適用されます
# 'pytorch': 通常のPyTorchモデル（学習時・開発時推奨）
# 'tensorrt': TensorRT最適化モデル - Jetson等で高速推論（Jetson推奨）
#             自動運転: MODEL_NAME_tensorrt.pt
#             位置推論: POSITION_MODEL_NAME_tensorrt.pt
#             YOLO: YOLO_MODEL_PATH.engine
# 'openvino': OpenVINO最適化モデル - CPU推論最適化（x86/RPi推奨）
#             自動運転: MODEL_NAME_openvino.xml/.bin
#             位置推論: POSITION_MODEL_NAME_openvino.xml/.bin
#             YOLO: YOLO_MODEL_PATH_openvino_model/
INFERENCE_ENGINE = "pytorch"  # 学習時はpytorchを使用  


## NNモデルのパラメータ
HIDDEN_DIM = 64 #（隠れ層のノード数）
NUM_HIDDEN_LAYERS = 3 #（隠れ層の数）

## 学習時のパラメータ
BATCH_SIZE = 64
EPOCHS = 30

## CNNモデル（donkey, resnet18等）の入力画像設定
# 学習時と推論時の両方で使用するカメラを指定
# "cam0/image_array": camera_0の画像
# "cam1/image_array": camera_1の画像
MODEL_INPUT_IMAGE = "cam1/image_array"
###利用しない：# "cam/image_array": 単体/結合画像（複数カメラがある場合）

# Early Stopping設定
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 5  # 検証損失が改善しなくなってから待機するエポック数
EARLY_STOPPING_MIN_DELTA = 1e-6  # 改善と判断する最小変化量

# データオーグメンテーション設定
USE_DATA_AUGMENTATION = True
# 水平反転
AUG_USE_FLIP = True
AUG_FLIP_PROB = 0.5
# 色調整
AUG_USE_COLOR = True
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.2
# 幾何変換
AUG_USE_GEOMETRY = True
AUG_ROTATION_DEGREES = 5
AUG_TRANSLATE_RATIO = 0.1
# ランダムイレース
AUG_USE_ERASE = True
AUG_ERASE_PROB = 0.5
AUG_ERASE_MIN_RATIO = 0.02
AUG_ERASE_MAX_RATIO = 0.2

## モデルの種類
MODEL_TYPE = "regression" #regression, categorical
### categoricalのカテゴリ設定、カテゴリ数は揃える↓　
NUM_CATEGORIES = 3
# -100~100の範囲で小さな値→大きな値の順にする（しないとValueError: bins must increase monotonically.）
CATEGORIES_STEERING = [RIGHT, NEUTRAL, LEFT]
CATEGORIES_THROTTLE = [FORWARD_CORNER, FORWARD_STRAIGHT, FORWARD_CORNER] #Strのカテゴリに合わせて設定
# 超音波センサーの距離値をNORMALIZEするスケール
NORMALIZE_RANGE = 2000

"""一旦保留
bins_Str = [-101] # -101は最小値-100を含むため設定、境界の最大値は100
# 分類の境界：binを設定(pd.cutで使う)
for i in range(NUM_CATEGORIES):
    bins_Str.append((CATEGORIES_STEERING[i]+CATEGORIES_STEERING[min(i+1,NUM_CATEGORIES-1)])/2)
bins_Str[-1] = 100
"""

# ============================================================================
# 超音波センサ設定
# ============================================================================
## 超音波センサの最大測定距離 ~4000(mm)
CUTOFF_RANGE = 2000 
## 超音波センサの測定回数、ultrasonic.pyチェック用
SAMPLING_TIMES = 100

# 超音波センサの設定
## 使う超音波センサ位置の指示、計測ループが遅い場合は数を減らす
## ["RrLH", "FrLH", "FrFR", "FrRH","RrRH"] = [真左, 前方左, 前方, 前方右, 真右]
#ULTRASONIC_SENSOR_LIST = ["FrLH","FrFR","FrRH"]
### ５つ使う場合はこちらをコメントアウト外す
ULTRASONIC_SENSOR_LIST = ["RrLH", "FrLH", "FrFR", "FrRH","RrRH"]
### ８つ使う場合ははこちらのコメントアウト外す
#ULTRASONIC_SENSOR_LIST.extend(["BackRH", "Back", "BackLH"])

# GPIOピン番号:超音波センサの位置の対応とPWMピンのチャンネル
ULTRASONIC_ECHO_PIN_NUMBER=[11,13,15,29,31,33,35,37]
ULTRASONIC_TRIGER_PIN_NUMBER=[12,16,18,22,32,36,38,40]
ULTRASONIC_ECHO_PINS = {name: ULTRASONIC_ECHO_PIN_NUMBER[i] for i, name in enumerate(ULTRASONIC_SENSOR_LIST)}
ULTRASONIC_TRIG_PINS = {name: ULTRASONIC_TRIGER_PIN_NUMBER[i] for i, name in enumerate(ULTRASONIC_SENSOR_LIST)}

# ============================================================================
# カメラ設定
# ============================================================================
IMAGE_W = 224 #160
IMAGE_H = 224 #120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = 60 #DRIVE_LOOP_HZ
# カメラ0のフリップ設定
CAMERA_0_VFLIP = False
CAMERA_0_HFLIP = False
# カメラ1のフリップ設定
CAMERA_1_VFLIP = True
CAMERA_1_HFLIP = True
#IMSHOW = False #　画像を表示するか
IMAGE_CONCAT_DIRECTION = "horizontal"  # "horizontal" or "vertical" - 複数カメラ画像の結合方向

SAVE_CONCATENATED_IMAGE = False  # 結合画像を保存するかどうか（True: 保存する, False: 保存しない）
RESIZE_CONCATENATED_IMAGE = True  # 結合画像をIMAGE_W x IMAGE_Hにリサイズするかどうか（True: リサイズする, False: リサイズしない）

# ============================================================================
# コントローラー設定
# ============================================================================
## コントローラータイプの選択
# "joystick": USBジョイスティック/ゲームパッド
# "pwm": プロポ（I2C経由でPWM信号を読み取り）
# "keyboard": キーボード（フォールバック）
CONTROLLER_TYPE = "joystick"  # joystick, pwm, keyboard


# ============================================================================
# LiDAR設定
# ============================================================================
# LiDARを有効にするか
HAVE_LIDAR = True
LIDAR_TYPE = "TMINI"  # "TMINI", "UST20", "NONE"

# 共通設定
LIDAR_IMAGE_W = 224
LIDAR_IMAGE_H = 224
SAVE_LIDAR_IMAGES = True    # LiDAR画像保存機能
SAVE_LIDAR_DATA = False      # LiDARの点群データをnumpy binary形式で保存
LIDAR_BINARY_IMAGE = False # LiDAR画像を白黒2値で表示するか（Trueの場合、点群は全て白、背景は黒）
WEB_SERVER_PORT = 8080      # Lidar単体確認用ウェブサーバーポート

# LiDAR画像縮尺設定
LIDAR_IMAGE_SCALE_FACTOR = 0.8  # 画像サイズに対するスケール係数 (0.0-1.0)
LIDAR_IMAGE_METERS_PER_PIXEL = 0.018  # 1ピクセルあたりの実際の距離（メートル）

# LiDAR搭載位置オフセット（mm単位）
# 車両中心を原点として、前方が正のY、右が正のX
LIDAR_OFFSET_X = 0    # 左右方向のオフセット（右が正）
LIDAR_OFFSET_Y = 330-450/2    # 前後方向のオフセット（前が正）

# 共通ゾーン名の定義（4ゾーン）
## 超音波センサーの各ポイントと共通名
ZONE_NAMES = ["RrLH", "FrLH", "FrFR", "FrRH", "RrRH"]

# 検出設定
LIDAR_DETECT_POINTS_THRESHOLD = 10    # 検出点数閾値（lidar毎にデータ点数異なるため注意）
# ゾーン別検出閾値（5つのゾーンそれぞれに設定可能）
LIDAR_DETECT_POINTS_THRESHOLD_ZONE = [
    LIDAR_DETECT_POINTS_THRESHOLD*2,  # Zone 0: 左後方 (RrLH)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 1: 左前方 (FrLH)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 2: 前方 (FrFR)
    LIDAR_DETECT_POINTS_THRESHOLD,  # Zone 3: 右前方 (FrRH)
    LIDAR_DETECT_POINTS_THRESHOLD*2   # Zone 4: 右後方 (RrRH)
]

LIDAR_DETECT_DISTANCE_THRESHOLD = 300    # 検出距離閾値　unit:mm
# ゾーン別検出閾値（5つのゾーンそれぞれに設定可能）
LIDAR_DETECT_DISTANCE_THRESHOLD_ZONE = [
    LIDAR_DETECT_DISTANCE_THRESHOLD +0,  # Zone 0: 左後方 (RrLH)
    LIDAR_DETECT_DISTANCE_THRESHOLD,  # Zone 1: 左前方 (FrLH)
    LIDAR_DETECT_DISTANCE_THRESHOLD + 100, #togikai -100,  # Zone 2: 前方 (FrFR)
    LIDAR_DETECT_DISTANCE_THRESHOLD,  # Zone 3: 右前方 (FrRH)
    LIDAR_DETECT_DISTANCE_THRESHOLD +0   # Zone 4: 右後方 (RrRH)
]
LIDAR_WALL_DISTANCE = LIDAR_DETECT_DISTANCE_THRESHOLD # 壁検出（描画）ロジック用

# 壁検出設定
LIDAR_DETECT_WALLS = False  # 壁検出を有効にするか

# 使用する検出手法（速度順）
# 選択肢: 'distance_based', 'split_merge', 'sliding_window', 'ransac', 'hybrid'
# distance_based: max_linearity=0で全領域検出、>0で直線のみ検出
LIDAR_DETECTION_METHOD = 'distance_based'

# 壁として認識する点間の最大距離 (mm)
LIDAR_WALL_MAX_GAP = 300

# 壁セグメントとして必要な最小点数
LIDAR_WALL_MIN_POINTS = 25

# 最大許容直線偏差（低いほど厳密な直線を要求）
LIDAR_WALL_MAX_LINEARITY = 0.08

# Split-Merge法用パラメータ
LIDAR_SPLIT_EPSILON = 90  # 分割閾値 (mm) - 点から直線への最大許容距離
LIDAR_MIN_SEGMENT_LENGTH = 900  # 最小セグメント長 (mm)
LIDAR_USE_ADAPTIVE = True  # 適応的閾値を使用するか
LIDAR_USE_2D_OPTIMIZATION = True  # 2D最適化を使用するか

# RANSAC法用パラメータ
LIDAR_RANSAC_THRESHOLD = 60  # RANSAC残差閾値 (mm)
LIDAR_MIN_INLIER_RATIO = 0.6  # 最小インライア率
LIDAR_RANSAC_MAX_TRIALS = 150  # RANSAC最大試行回数
LIDAR_EARLY_STOP_RATIO = 0.9  # RANSAC早期終了閾値

# スライディングウィンドウ法用パラメータ
LIDAR_WINDOW_SIZE = 20  # ウィンドウサイズ（点数）
LIDAR_WINDOW_STRIDE = 5  # ウィンドウの移動幅（点数）
LIDAR_OVERLAP_THRESHOLD = 700  # 重複閾値 (mm)

# Hybrid法用パラメータ
LIDAR_CONFIDENCE_THRESHOLD = 0.8  # RANSAC検証の信頼度閾値

# セグメント統合用パラメータ
LIDAR_MERGE_ANGLE_THRESHOLD = 10  # 統合時の角度閾値 (度)
LIDAR_MERGE_DISTANCE_THRESHOLD = 100  # 統合時の距離閾値 (mm)

# ============================================================================
# LiDAR機種別設定
# ============================================================================

if LIDAR_TYPE == "TMINI":
    # YDLIDAR TMINI 設定
    LIDAR_SCAN_RATE = 10         # スキャンレート (Hz) - 6-12
    LIDAR_DATA_POINTS = int(4000/LIDAR_SCAN_RATE)  # TMINIは最大測定周波数4000kHz、スキャンレートに応じて点数変化
    LIDAR_ANGLE_RANGE = 360     # 度
    LIDAR_ANGLE_START = 0    # 度
    LIDAR_ANGLE_END = 360       # 度
    LIDAR_ANGLE_OFFSET = 90      # 度 - LiDARの向きを調整するオフセット値（正の値で右回転）
    LIDAR_CLOCKWISE = True      # スキャン方向（True:時計回り、False:反時計回り）

    # 通信設定
    LIDAR_COMM_TYPE = "serial"
    LIDAR_SERIAL_PORT = "/dev/ttyS0"  # Bluetooth無効化後のハードウェアUART (GPIO 14/15)
    LIDAR_SERIAL_BAUDRATE = 230400

    # 単位系設定
    LIDAR_UNIT_TYPE = "m"       # TMINIのネイティブ単位系（"m" or "mm"）
    LIDAR_TARGET_UNIT = "mm"    # システム内部で使用する単位系（"m" or "mm"）

    # 測定範囲
    LIDAR_MIN_DISTANCE = 20        # unit:mm
    LIDAR_MAX_DISTANCE = 4000      # unit:mm (TMINIの実用範囲)
    LIDAR_IGNORE_DISTANCE = 150    # unit:mm, ライダー近傍の部品を無視する距離

    # ゾーンインデックスの定義（ZONE_NAMESに対応）
    # 5つのゾーンに分割したLiDAR検出範囲を定義
    # TMINIの400点を5つのゾーンに分割
    # 360度を400点で分割: 1点あたり0.9度
    # 角度オフセット90度により、インデックス0は車両の右方向（90度）を指す
    # RrLH, FrLH, FrFR, FrRH, RrRH の順
    ZONE_INDEX = [
        [x for x in range(317, 350)],   # RrLH: 左後方 (-75°~-45° = 285°~315° = インデックス317~350)
        [x for x in range(350, 383)],   # FrLH: 左前方 (-45°~-15° = 315°~345° = インデックス350~383)
        [x for x in range(383, 400)]+[x for x in range(0, 17)],   # FrFR: 前方 (-15°~15° = 345°~15° = インデックス383~400,0~17)
        [x for x in range(17, 50)],     # FrRH: 右前方 (15°~45° = インデックス17~50)
        [x for x in range(50, 83)]      # RrRH: 右後方 (45°~75° = インデックス50~83)
    ]

elif LIDAR_TYPE == "UST20":
    # 北陽 UST-20 設定
    LIDAR_SCAN_RATE = 40  # スキャンレート (Hz)
    LIDAR_DATA_POINTS = 1081
    LIDAR_CLOCKWISE = False     # スキャン方向（True:時計回り、False:反時計回り）- UST-20は反時計回り
    LIDAR_ANGLE_RANGE = 270     # 度
    LIDAR_ANGLE_START = -135   # 度（インデックス0の角度、LidarImageConverterの表示調整）
    LIDAR_ANGLE_END = 135       # 度（最後のインデックスの角度）
    LIDAR_ANGLE_STEP = 4        #
    LIDAR_ANGLE_OFFSET = 90   # 度（車両の向きとLidarImageConverterの表示調整、0が右向き）

    # 通信設定
    LIDAR_COMM_TYPE = "ethernet"
    LIDAR_IP_ADDRESS = "192.168.0.10"
    LIDAR_PORT = 10940

    # 単位系設定
    LIDAR_UNIT_TYPE = "mm"      # UST-20のネイティブ単位系（"m" or "mm"）
    LIDAR_TARGET_UNIT = "mm"    # システム内部で使用する単位系（"m" or "mm"）

    # 測定範囲
    LIDAR_MIN_DISTANCE = 100       # unit:mm (UST-20:20mm~)
    LIDAR_MAX_DISTANCE = 20000     # unit:mm (の最大範囲)
    LIDAR_IGNORE_DISTANCE = 100    # unit:mm, ライダー近傍の部品を無視する距離

    # ゾーンインデックスの定義（ZONE_NAMESに対応）
    # RrLH, FrLH, FrFR, FrRH, RrRH の順
    ZONE_INDEX = [
        [x for x in range(180 *LIDAR_ANGLE_STEP,     240 *LIDAR_ANGLE_STEP)],  # RrLH: 左後方
        [x for x in range(150 *LIDAR_ANGLE_STEP,     180 *LIDAR_ANGLE_STEP)],     # FrLH: 左前方
        [x for x in range(120 *LIDAR_ANGLE_STEP,     150 *LIDAR_ANGLE_STEP)],     # FrFR: 前方
        [x for x in range(90 *LIDAR_ANGLE_STEP,      120 *LIDAR_ANGLE_STEP)],      # FrRH: 右前方
        [x for x in range(30 *LIDAR_ANGLE_STEP,      90 *LIDAR_ANGLE_STEP)]     # RrRH: 右後方
    ]

else:
    # LiDARが設定されていない場合のデフォルト値
    LIDAR_TYPE = "NONE"
    LIDAR_DATA_POINTS = 0
    LIDAR_ANGLE_RANGE = 0
    LIDAR_ANGLE_START = 0
    LIDAR_ANGLE_END = 0
    LIDAR_ANGLE_OFFSET = 0
    LIDAR_CLOCKWISE = True
    LIDAR_SCAN_RATE = 1  # スキャンレート (Hz)
    LIDAR_MIN_DISTANCE = 20        # unit:mm
    LIDAR_MAX_DISTANCE = 4000      # unit:mm
    LIDAR_IGNORE_DISTANCE = 150    # unit:mm
    ZONE_INDEX = [[], [], [], [], []]  # 空のゾーンインデックス

# ============================================================================
# LiDAR自動スロットル調整機能
# ============================================================================
# 特定のゾーンに障害物がない場合、スロットルを自動的に設定値に変更する機能
LIDAR_THROTTLE_ENABLED = False # True: 有効, False: 無効

# 監視するゾーンのインデックス (0-based)
# Zone 0: 左後方 (RrLH)
# Zone 1: 左前方 (FrLH)
# Zone 2: 前方 (FrFR)
# Zone 3: 右前方 (FrRH)
# Zone 4: 右後方 (RrRH)
LIDAR_THROTTLE_ZONE = 2  # 前方ゾーンを監視

# 障害物検出の距離閾値 (mm単位)
# この距離より近くに障害物がない場合、スロットルを設定値に変更
LIDAR_THROTTLE_DISTANCE = 4000

# 障害物がない時のスロットル値
LIDAR_THROTTLE_VALUE = 1  # 最大スロットル


# コントローラー（ジョイスティックの設定）
HAVE_JOYSTICK = True #True
JOYSTICK_STEERING_SCALE = 1.0 # default:1, left=-1, right=1に調整
JOYSTICK_THROTTLE_SCALE =  -1.0 # default:-1, reverse=-1, forward=1に調整
#CONTROLLER_TYPE = 'F710'
JOYSTICK_DEVICE_FILE = "/dev/input/js0" 
## jsが↑のパスで表示されない場合はREADMEを確認し、ドライバーのインストールが必要
## ジョイスティックのボタンとスティック割り当て
# F710の操作設定 #割り当て済み
JOYSTICK_A = 0 #ブレーキ
JOYSTICK_B = 1 #アクセル２
JOYSTICK_X = 2 #アクセル１
JOYSTICK_Y = 3 #記録停止開始
JOYSTICK_LB = 4
JOYSTICK_RB = 5
JOYSTICK_BACK = 6
JOYSTICK_S = 7 #自動/手動走行切り替え
JOYSTICK_LOGICOOL = 8
JOYSTICK_LSTICKB = 9
JOYSTICK_RSTICKB = 10
JOYSTICK_AXIS_LEFT = 0 #ステアリング（左右）
JOYSTICK_AXIS_RIGHT = 4 #スロットル（上下）
JOYSTICK_HAT_LR = 0
JOYSTICK_HAT_DU = 1

## プロポPWM信号設定（read_pwm_signals.pyでキャリブレーション）
# I2C経由でプロポから直接PWM信号を読み取る設定
PWM_I2C_ADDRESS = 0x08         # PWMコントローラのI2Cアドレス
PWM_I2C_BUS = 7                # I2Cバス番号（Jetson Orin Nano: 7, RPi: 1）
PWM_RAW_TO_US_SCALE = 1000.0   # RAW値をマイクロ秒に変換するスケール

# プロポPWM値の範囲（read_pwm_signals.pyでキャリブレーション後に設定）
# CH1: ステアリングチャンネル
PWM_CH1_LEFT_RAW = 1829           # ステアリング左最大時のRAW値
PWM_CH1_CENTER_RAW = 1542         # ステアリング中央時のRAW値
PWM_CH1_RIGHT_RAW = 1206          # ステアリング右最大時のRAW値

# CH2: スロットルチャンネル
PWM_CH2_FORWARD_RAW = 1098        # スロットル前進最大時のRAW値
PWM_CH2_NEUTRAL_RAW = 1476        # スロットル中立時のRAW値
PWM_CH2_REVERSE_RAW = 1896        # スロットル後退最大時のRAW値

# ============================================================================
# IMU/ジャイロ設定
# ============================================================================
# ジャイロを使った動的制御モード選択
MODE_DYNAMIC_CONTROL = "counter_steering" #"lateral_g_throttle"

# ============================================================================
# オプティカルフローセンサー設定
# ============================================================================
# 速度(mm/2)を算出するための調整値、路面から30mmの位置にセンサー設置で0.1程度
POSITION_SCALING_FACTOR =  0.1

# ============================================================================
# 出力・モニタリング設定
# ============================================================================
##ターミナルへの出力
TERMINAL_PRINT = True

## 走行中のデータ確認用WEBアプリ 下記のport番号に出力
MONITOR = True #True
MONITOR_PORT = 8000

# ============================================================================
# 走行記録設定
# ============================================================================
## 測定データの保存場所
RECORD_FILE_NAME = "record"
RECORDS_DIRECTORY = "records"
RECORDS_DIRECTORY_ULTRASONIC_TEST = "records/ultrasonic_test.csv"
SAVE_FORMAT = "donkeycar" # csv, ndjson, donkeycar 
IMAGES_DIRECTORY = "images"

# ============================================================================
# シミュレーションモード
# ============================================================================
SIM_MODE = False


# ============================================================================
# 位置推論とモデル切り替え設定
# ============================================================================
# annotation_training_d2jで学習した位置推論モデルを使用した自動運転モデル切り替え
USE_POSITION_SWITCHING = False  # 位置推論によるモデル切り替えを有効化
POSITION_MODEL_NAME = None  # 位置推論モデルのファイル名（例: "resnet18_location_20250101.pth"）
POSITION_MODEL_TYPE = "resnet18_location"  # 位置推論モデルのアーキテクチャ（donkey_location, resnet18_location）
POSITION_NUM_CLASSES = 8  # 位置クラス数（annotation_training_d2jのデフォルトは8）
POSITION_MODEL_INPUT_IMAGE = "cam1/image_array"

# 位置ごとのモデルマッピング（位置クラスID → 自動運転モデルファイル名）
# 例: 位置0=直線、位置1=左カーブ、位置2=右カーブなど（アノテーション時に定義）
POSITION_MODELS_MAP = {
    0: "model_position0.pth",  # 位置0用の自動運転モデル
    1: "model_position1.pth",  # 位置1用の自動運転モデル
    2: "model_position2.pth",  # 位置2用の自動運転モデル
    3: "model_position3.pth",  # 位置3用の自動運転モデル
    # 必要に応じて位置4-7も追加
}

# 位置クラスの名前（ログ表示用、アノテーション時の定義に合わせる）
POSITION_CLASS_NAMES = [
    "Position0", "Position1", "Position2", "Position3",
    "Position4", "Position5", "Position6", "Position7"
]

# 位置推論の頻度（フレーム数、推論コスト削減のため）
POSITION_INFERENCE_INTERVAL = 5  # 5フレームに1回位置推論を実行

# デフォルトモデル（位置が推論できない場合や、マッピングにない位置の場合に使用）
POSITION_DEFAULT_MODEL = None  # Noneの場合はMODEL_NAMEを使用

# ============================================================================
# YOLO物体検知による制御修正とモデル切り替え設定
# ============================================================================
## yolo_detection.pyにロジックは実装
# YOLOモデルで物体を検知し、検知結果に応じて制御値を修正したりモデルを切り替える
USE_YOLO_DETECTION = False  # YOLO物体検知機能を有効化

# YOLOモデル設定
YOLO_MODEL_PATH = "models/yolov8n.pt"  # YOLOモデルファイルパス（yolov8n.pt, yolov8s.pt等）
YOLO_CONFIDENCE_THRESHOLD = 0.5  # 検知信頼度閾値（0.0-1.0）
YOLO_IOU_THRESHOLD = 0.45  # NMS（Non-Maximum Suppression）のIoU閾値
YOLO_INPUT_SIZE = 640  # YOLO入力画像サイズ（640, 320等）
YOLO_DETECTION_INTERVAL = 3  # 物体検知の実行間隔（フレーム数）

# 検知結果に基づく制御修正設定
## yolo_detection.pyにロジックは実装
YOLO_CONTROL_RULES = {
    # 検知クラスID: {"steering_offset": 値, "throttle_scale": 倍率, "priority": 優先度}
    # カスタム学習クラス用設定（car, route, signal, stop, park）
    0: {  # car（車）を検知
        "steering_offset": 0.0,
        "throttle_scale": 0.5,  # 50%に減速
        "priority": 8,
        "description": "Car detected - Reduce speed"
    },
    1: {  # route（ルート/道路）を検知
        "steering_offset": 0.0,
        "throttle_scale": 1.0,  # 通常速度
        "priority": 5,
        "description": "Route detected - Normal speed"
    },
    2: {  # signal（信号）を検知
        "steering_offset": 0.0,
        "throttle_scale": 0.6,  # 60%に減速
        "priority": 9,
        "description": "Signal detected - Prepare to stop"
    },
    3: {  # stop（停止標識）を検知
        "steering_offset": 0.0,
        "throttle_scale": 0.0,  # 完全停止
        "priority": 10,
        "description": "Stop sign - Full stop"
    },
    4: {  # park（駐車エリア）を検知
        "steering_offset": 0.0,
        "throttle_scale": 0.3,  # 30%に減速
        "priority": 7,
        "description": "Parking area - Slow down"
    },
}

# 検知結果に基づくモデル切り替え設定
YOLO_MODEL_SWITCHING = {
    # 検知クラスID: モデルファイル名
    # カスタム学習クラス用（car, route, signal, stop, park）
    0: "car_traffic_model.pth",  # 車検知時は交通状況対応モデル
    1: "route_following_model.pth",  # ルート検知時は経路追従モデル
    2: "signal_aware_model.pth",  # 信号検知時は信号対応モデル
    3: "stop_zone_model.pth",  # 停止標識検知時は停止エリアモデル
    4: "parking_model.pth",  # 駐車エリア検知時は駐車モデル
    # 必要に応じて追加・削除
}

# 検知対象クラスのフィルタリング（Noneの場合は全クラス検知）
YOLO_TARGET_CLASSES = None  # 例: [0, 1, 2, 3, 4]  # カスタムクラスのみ検知する場合

# 検知結果の表示設定
YOLO_DISPLAY_DETECTIONS = True  # 検知結果をターミナルに表示
YOLO_SAVE_ANNOTATED_IMAGES = False  # 検知結果を画像に描画して保存

# YOLO物体追従制御設定
USE_YOLO_OBJECT_TRACKING = False  # 検知物体に向かって進む制御を有効化
YOLO_TRACKING_TARGET_CLASSES = [4]  # 追従対象クラスID（例: [1]はroute）
YOLO_TRACKING_STEERING_GAIN = 0.8  # ステアリング補正ゲイン（0.0-2.0推奨）
YOLO_TRACKING_CENTER_DEADZONE = 0.1  # 中心不感帯（画像幅に対する比率、0.0-0.3推奨）

# YOLO障害物回避制御設定
USE_YOLO_OBSTACLE_AVOIDANCE = False  # 障害物回避制御を有効化
YOLO_OBSTACLE_CLASSES = [0]  # 回避対象クラスID（例: [0]はcar）
YOLO_OBSTACLE_AVOIDANCE_GAIN = 1.2  # 回避ステアリングゲイン（0.5-2.0推奨）
YOLO_OBSTACLE_SIZE_THRESHOLD = 0.15  # 回避判定する物体サイズ閾値（画像面積比、0.05-0.3推奨）
YOLO_OBSTACLE_CENTER_ZONE = 0.4  # 中央エリア幅（画像幅に対する比率、0.3-0.6推奨）

# YOLOクラス名（カスタム学習用）
# 学習データに合わせて設定してください
YOLO_CLASS_NAMES = {
    0: "car",      # 車
    1: "route",    # ルート/道路
    2: "signal",   # 信号
    3: "stop",     # 停止標識
    4: "park",     # 駐車エリア
}

# YOLOクラス名（COCO dataset参考）
# COCO datasetを使用する場合は以下に切り替え
# YOLO_CLASS_NAMES = {
#     0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
#     5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
#     10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
#     14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"
#     # 以下省略（COCO datasetは80クラス）
# }


