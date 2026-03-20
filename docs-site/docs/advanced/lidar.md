# LiDAR統合機能

## 概要

2D LiDARセンサーを超音波センサーの代替または補完として使用できます。LiDARデータは超音波センサーと同じ形式に変換されるため、既存のプランナーをそのまま使用できます。

## 対応LiDAR

- Hokuyo URG-04LX-UG01
- その他SCIP2.0プロトコル対応LiDAR

## ゾーン構成

LiDARの360度スキャンデータを5つのゾーンに分割して、超音波センサーと同等の形式に変換します。

```
        FrFR (前方)
          │
    FrLH ─┼─ FrRH
   (左前) │ (右前)
          │
    RrLH ─┴─ RrRH
   (左後)   (右後)
```

## config.pyでの設定

```python
# LiDARを有効にする
HAVE_LIDAR = True

# アクティブセンサーに追加
ACTIVE_SENSORS = ["lidar"]  # または ["ultrasonic", "lidar"]

# LiDAR接続設定
LIDAR_PORT = "/dev/ttyACM0"  # シリアルポート
LIDAR_BAUDRATE = 115200

# ゾーン設定（角度範囲）
LIDAR_ZONES = {
    "RrLH": (225, 270),   # 左後方
    "FrLH": (270, 315),   # 左前方
    "FrFR": (315, 45),    # 前方（0度をまたぐ）
    "FrRH": (45, 90),     # 右前方
    "RrRH": (90, 135),    # 右後方
}

# 点数閾値（この点数以上の点群があるゾーンを障害物と判定）
LIDAR_POINT_THRESHOLD = 3

# 距離閾値（この距離より近いゾーンを障害物と判定、単位:mm）
LIDAR_DISTANCE_THRESHOLD = 500

# ゾーン別の閾値設定も可能
LIDAR_ZONE_THRESHOLDS = {
    "FrFR": 300,   # 前方は近めに設定
    "FrLH": 400,
    "FrRH": 400,
    "RrLH": 500,
    "RrRH": 500,
}
```

## run.pyでの使用方法

```python
# LiDARデータは超音波センサーと同じ形式でアクセス可能
ultrasonic_data = data_aggregator.get_ultrasonic_data()
# ultrasonic_data = {"FrLH": 450, "FrFR": 800, "FrRH": 420, ...}

# 超音波センサーとLiDARを切り替えて使用
if config.HAVE_LIDAR:
    sensor_data = lidar.get_ranges()
else:
    sensor_data = ultrasonic.get_ranges()

# planner.pyでの使用（変更不要）
steering, throttle = planner.plan(sensor_data)
```

## lidar.pyの機能

```python
# 取得できるデータ
detection_distances  # 各ゾーンの最小距離のリスト [RrLH, FrLH, FrFR, FrRH, RrRH]
detection_binary     # 各ゾーンの検出有無 [0/1, 0/1, 0/1, 0/1, 0/1]
image               # LiDAR画像（OpenCV形式）
measurements        # 全点群データ
detection_details   # 詳細な検出情報
```

## 利点

| 項目 | 超音波センサー | LiDAR |
|------|--------------|-------|
| 精度 | ±3mm | ±30mm |
| 範囲 | 15度（片側） | 240度〜360度 |
| 更新速度 | 〜100Hz | 〜40Hz |
| コスト | 低い | 高い |
| 複数障害物 | 検知困難 | 同時検知可能 |

## 注意事項

- LiDARはシリアル接続のため、権限設定が必要な場合があります
- ネットワーク接続のLiDARはIPアドレスの設定が必要です
- 超音波センサーとの併用時はデータの統合方法を検討してください

## デバッグ方法

```bash
# LiDAR単体での動作確認
python lidar.py

# run.py実行時のログ確認
python run.py --debug
# "--- LiDAR初期化開始 ---" が表示されることを確認
```

---

## ネットワーク接続LiDARの設定

### ネットワーク構成

WiFiでインターネット接続しつつ、EthernetでLiDARに接続する場合の設定です。

```
ミニカー
├── WiFi (wlP1p1s0) → インターネット
└── Ethernet (enP8p1s0) → LiDAR (192.168.0.10)
```

### 設定手順

```bash
# 接続名を確認
nmcli connection show

# WiFiを高優先（低メトリック値）に設定
sudo nmcli connection modify "your-wifi-name" ipv4.route-metric 100

# Ethernetの優先度を下げてLiDAR専用に設定
sudo nmcli connection modify "hokuyo-lidar" ipv4.route-metric 1000
sudo nmcli connection modify "hokuyo-lidar" ipv4.never-default yes

# LiDARへの静的ルートを追加
sudo nmcli connection modify "hokuyo-lidar" +ipv4.routes "192.168.0.10/32"

# 接続を再起動して設定を適用
sudo nmcli connection down "hokuyo-lidar" && sudo nmcli connection up "hokuyo-lidar"
```

### 設定確認

```bash
# インターネット接続（WiFi経由）
ping -c 3 google.com

# LiDAR接続（Ethernet経由）
ping -c 3 192.168.0.10
```
