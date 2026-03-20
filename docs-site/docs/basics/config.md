# config.py 設定リファレンス

config.pyの主要な設定項目を解説します。

---

## ハンズオン向けクイック設定

### 走行モード一覧

| モード | 説明 | 用途 |
|-------|------|------|
| `manual` | 手動操作 | データ収集 |
| `go_straight` | 直進 | 動作確認 |
| `right_left_3` | 3センサーで障害物回避 | ルールベース走行 |
| `wall_follow` | 壁沿い走行 | 右手法/左手法 |
| `wall_follow_pid` | PID制御壁沿い走行 | PID制御の学習 |
| `nn` | ニューラルネットワーク | センサー値ベースの学習 |
| `donkeycar` | 軽量CNN | 画像ベースの学習（おすすめ） |
| `resnet18` | ResNet18 | 高精度画像認識 |

### 最小限の設定変更

```python
# 1. 走行モードを選択
PLAN = "donkeycar"  # または "manual", "nn" など

# 2. 使用するセンサーを選択
ACTIVE_SENSORS = ["lidar", "camera_0"]  # または ["ultrasonic", "camera_0"]

# 3. 速度を調整
FORWARD_STRAIGHT = 0.4  # 直線用（0.3〜0.5推奨）
FORWARD_CORNER = 0.3    # カーブ用（0.2〜0.4推奨）

# 4. 学習済みモデルを指定（自動走行時）
MODEL_NAME = "donkeycar_20251205_150000.pth"
```

---

## デバイス設定

```python
# デバイス自動検出（通常は変更不要）
DEVICE_TYPE = "auto"  # "rpi4", "rpi5", "jetson", "auto"
```

## 走行モード

```python
# 判断モード選択
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
PLAN = "right_left_3"
```

## センサー設定

### 超音波センサー

```python
# センサーリスト
ULTRASONIC_SENSOR_LIST = ["FrLH", "FrFR", "FrRH"]

# 測定パラメータ
SAMPLING_TIMES = 3          # サンプリング回数
CUTOFF_RANGE = 2000         # カットオフ距離(mm)
DETECTION_RANGE = 500       # 検知範囲(mm)
STOP_RANGE = 100            # 停止距離(mm)
RIGHT_LEFT_RANGE = 400      # 左右判断距離(mm)
```

### カメラ

```python
# カメラ設定
HAVE_CAMERA = True
CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
CAMERA_FPS = 30
```

## モーター設定

### ステアリング

```python
# ステアリングのPWM値
STEERING_CENTER_PWM = 370
STEERING_WIDTH_PWM = 80
STEERING_RIGHT_PWM = STEERING_CENTER_PWM + STEERING_WIDTH_PWM
STEERING_LEFT_PWM = STEERING_CENTER_PWM - STEERING_WIDTH_PWM
```

### スロットル

```python
# スロットルのPWM値
THROTTLE_STOPPED_PWM = 370
THROTTLE_FORWARD_PWM = 500
THROTTLE_REVERSE_PWM = 300
```

### 出力値

```python
# ステアリング出力（-1〜1）
LEFT = -1
NEUTRAL = 0
RIGHT = 1

# スロットル出力（-1〜1）
FORWARD_S = 0.6       # ストレート速度
FORWARD_C = 0.4       # カーブ速度
STOP = 0
REVERSE = -1
```

## コントローラー設定

```python
# コントローラータイプ
CONTROLLER_TYPE = "joystick"  # "joystick", "pwm", "keyboard"

# ジョイスティック設定
HAVE_JOYSTICK = True
JOYSTICK_STEERING_SCALE = 1.0
JOYSTICK_THROTTLE_SCALE = -1.0
JOYSTICK_DEVICE_FILE = "/dev/input/js0"

# ボタン割り当て
JOYSTICK_A = 0
JOYSTICK_B = 1
JOYSTICK_X = 2
JOYSTICK_Y = 3
JOYSTICK_S = 7

# 軸割り当て
JOYSTICK_AXIS_LEFT = 0
JOYSTICK_AXIS_RIGHT = 4
```

## PID制御設定

```python
# 壁沿い走行
HAND_SIDE = "right"   # "left", "right"
TARGET_RANGE = 200    # 目標距離(mm)

# PIDパラメータ
K_P = 0.005
K_I = 0.0
K_D = 0.0005
```

## 機械学習設定

```python
# NN有効化
HAVE_NN = True

# モデルパス
MODEL_DIR = "models"
MODEL_NAME = "model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# ハイパーパラメータ
HIDDEN_DIM = 64
NUM_HIDDEN_LAYERS = 3
BATCH_SIZE = 8
EPOCHS = 5

# 推論エンジン
INFERENCE_ENGINE = "pytorch"  # "tensorrt", "openvino"

# 正規化
NORMALIZE_RANGE = 2000
```

## データ保存設定

```python
# 保存形式
SAVE_FORMAT = "donkeycar"  # "csv", "donkeycar"

# 保存先
DATA_DIR = "data"
```

## 復帰モード設定

```python
# 復帰モード
RECOVERY_MODE = "back"        # "none", "back"
RECOVERY_STREERING = LEFT
RECOVERY_TIME_DURATION = 1    # 秒
RECOVERY_BRAKING = 1
```

## I2C設定（プロポ用）

```python
# PWMコントローラー
PWM_I2C_ADDRESS = 0x08
PWM_I2C_BUS = 7               # Jetson: 7, RPi: 1

# キャリブレーション値
PWM_CH1_LEFT_RAW = 1098
PWM_CH1_CENTER_RAW = 1519
PWM_CH1_RIGHT_RAW = 1916
PWM_CH2_FORWARD_RAW = 1896
PWM_CH2_NEUTRAL_RAW = 1468
PWM_CH2_REVERSE_RAW = 1098
```
