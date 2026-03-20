# monitor.py

import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.gen
import tornado.concurrent
import asyncio
import time
import cv2
import config  # config.pyの定数を更新するため
import threading
from datetime import datetime
import logging
import os
import numpy as np
import json
import base64
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import platform

# 走行中に取得されるセンサーデータやステアリング値などを保持する辞書
# run.py のメインループから update_data() を介して書き込まれる想定
realtime_data = {
    "mode": None,
    "steering_value": 0.0,
    "throttle_value": 0.0,
    "ranges": {},  # 例: {"Fr": 100.0, "FrLH": 50.0, ...}
    "imu_data": None,
    "timestamp": None,
    # 画像フレーム (numpy配列) はここに入る
    "camera_image_0": None,
    "camera_image_1": None,

    # 走行一時停止用のフラグ例
    "pause_drive": False,
}

# setconfigでの再ロード用
set_config_reload = False

# 終了シグナル用フラグ
shutdown_signal = False

# 単独起動時のカメラインスタンス
camera_instances = {}
camera_update_thread = None

# 単独起動時のセンサーインスタンス
active_sensor_instances = {}
data_aggregator_instance = None

# WebSocketクライアント管理
websocket_clients = set()

# スレッドプール
executor = ThreadPoolExecutor(max_workers=4)

# 画像エンコード用のロック
image_lock = threading.Lock()

# 画像キャッシュ（フレーム変更検出用）
last_frame_hash = None
last_encoded_image = None

# 高速化用の設定
FAST_MODE = True  # 高速モードフラグ
SKIP_FRAME_COUNT = 0  # フレームスキップ数（0=全フレーム処理）
frame_skip_counter = 0


#------------------------------------------------------------------------------#
# HTMLテンプレートを動的に生成（templatesフォルダが無い場合の対応）
#------------------------------------------------------------------------------#
def create_template_if_needed():
    """templatesフォルダとmonitor.htmlを作成（存在しない場合）"""
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created {templates_dir} directory")

    index_html_path = os.path.join(templates_dir, "monitor.html")
    if not os.path.exists(index_html_path):
        # センサー可視化対応のHTMLテンプレートを作成
        html_content = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ロボットカー監視システム</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: white;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        }
        
        .status-panel {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .status-card {
            flex: 1;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
            text-align: center;
        }
        
        .status-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .main-content {
            display: flex;
            gap: 20px;
        }
        
        .left-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .right-panel {
            flex: 1;
        }
        
        .sensor-visualization {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
        }
        
        .camera-feed {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .camera-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            background-color: #0a0a0a;
        }
        
        .controls {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
        }
        
        .control-button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .pause-button {
            background-color: #ff6b6b;
            color: white;
        }
        
        .resume-button {
            background-color: #51cf66;
            color: white;
        }
        
        .pause-button:hover {
            background-color: #ff5252;
        }
        
        .resume-button:hover {
            background-color: #40c057;
        }
        
        canvas {
            border: 2px solid #444;
            border-radius: 8px;
            background-color: #0a0a0a;
            display: block;
            margin: 20px auto;
        }
        
        .legend {
            margin-top: 20px;
            padding: 15px;
            background-color: #333;
            border-radius: 8px;
        }
        
        .legend-item {
            display: inline-block;
            margin: 5px 15px;
            font-size: 14px;
        }
        
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 15px;
            margin-right: 8px;
            vertical-align: middle;
            border-radius: 3px;
        }
        
        .sensor-data {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .sensor-item {
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        .sensor-label {
            font-size: 12px;
            color: #ccc;
            margin-bottom: 5px;
        }
        
        .sensor-value {
            font-size: 18px;
            font-weight: bold;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        .connected {
            background-color: #51cf66;
            color: white;
        }
        
        .disconnected {
            background-color: #ff6b6b;
            color: white;
        }
        
        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .status-panel {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">接続中...</div>
    
    <div class="container">
        <div class="header">
            <h1>ロボットカー監視システム</h1>
            <div id="timestamp">-</div>
        </div>
        
        <div class="status-panel">
            <div class="status-card">
                <div class="status-label">動作モード</div>
                <div class="status-value" id="mode">-</div>
            </div>
            <div class="status-card">
                <div class="status-label">ステアリング</div>
                <div class="status-value" id="steering">0.0</div>
            </div>
            <div class="status-card">
                <div class="status-label">スロットル</div>
                <div class="status-value" id="throttle">0.0</div>
            </div>
            <div class="status-card">
                <div class="status-label">一時停止</div>
                <div class="status-value" id="pauseStatus">正常動作</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="left-panel">
                <div class="sensor-visualization">
                    <h3>超音波センサー可視化</h3>
                    <canvas id="sensorCanvas" width="800" height="600"></canvas>
                    
                    <div class="legend">
                        <h4>凡例</h4>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #ff4444;"></span>
                            危険域 (0-300mm)
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #ffaa00;"></span>
                            警告域 (300-600mm)
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #44ff44;"></span>
                            安全域 (600mm以上)
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background-color: #4444ff;"></span>
                            車体
                        </div>
                    </div>
                </div>
                
                <div class="controls">
                    <h3>制御</h3>
                    <button class="control-button pause-button" onclick="pauseDrive()">一時停止</button>
                    <button class="control-button resume-button" onclick="resumeDrive()">再開</button>
                </div>
            </div>
            
            <div class="right-panel">
                <div class="camera-feed">
                    <h3>カメラ映像</h3>
                    <img id="cameraImage" class="camera-image" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="カメラ映像">
                    <div style="margin-top: 10px; font-size: 12px; color: #ccc;">
                        リアルタイム映像
                    </div>
                </div>
                
                <div class="sensor-data">
                    <h3>センサーデータ</h3>
                    <div class="sensor-grid">
                        <div class="sensor-item">
                            <div class="sensor-label">左後方 (RrLH)</div>
                            <div class="sensor-value" id="sensorRrLH">-</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-label">前左 (FrLH)</div>
                            <div class="sensor-value" id="sensorFrLH">-</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-label">前方 (FrFR)</div>
                            <div class="sensor-value" id="sensorFrFR">-</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-label">前右 (FrRH)</div>
                            <div class="sensor-value" id="sensorFrRH">-</div>
                        </div>
                        <div class="sensor-item">
                            <div class="sensor-label">右後方 (RrRH)</div>
                            <div class="sensor-value" id="sensorRrRH">-</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket接続
        let ws = null;
        let connectionRetryCount = 0;
        const maxRetries = 5;
        
        // センサー可視化キャンバス（グローバル変数）
        let sensorCanvas = null;
        let sensorCtx = null;
        
        // センサー配置定義（ビーム幅統一版）
        const sensorConfig = {
            // 前方センサー群
            FrFR: { 
                angle: -Math.PI / 2,   // 前方中央（上方向）
                arc: Math.PI / 6,      // 30度の扇
                x: 0, y: -45,          // 車体前方（円の縁）
                label: '前方'
            },
            FrLH: { 
                angle: -Math.PI / 2 - Math.PI / 6,  // 前方左（上から30度左）
                arc: Math.PI / 6,      // 30度の扇
                x: -30, y: -32,        // 前方左寄り
                label: '前左'
            },
            FrRH: { 
                angle: -Math.PI / 2 + Math.PI / 6,  // 前方右（上から30度右）
                arc: Math.PI / 6,      // 30度の扇
                x: 30, y: -32,         // 前方右寄り
                label: '前右'
            },
            // 側方センサー群（ビーム幅をFrと同じに統一）
            RrLH: { 
                angle: Math.PI,        // 真左方向
                arc: Math.PI / 6,      // 30度の扇（45度から30度に変更）
                x: -45, y: 0,          // 車体左側（円の縁）
                label: '左後'
            },
            RrRH: { 
                angle: 0,              // 真右方向
                arc: Math.PI / 6,      // 30度の扇（45度から30度に変更）
                x: 45, y: 0,           // 車体右側（円の縁）
                label: '右後'
            }
        };

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                    updateConnectionStatus(true);
                    connectionRetryCount = 0;
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (e) {
                        console.error('Failed to parse WebSocket message:', e);
                    }
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    updateConnectionStatus(false);
                    
                    if (connectionRetryCount < maxRetries) {
                        connectionRetryCount++;
                        setTimeout(connectWebSocket, 2000 * connectionRetryCount);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
            } catch (error) {
                console.error('Failed to create WebSocket:', error);
                updateConnectionStatus(false);
            }
        }

        function handleWebSocketMessage(data) {
            if (data.type === 'image' && data.data) {
                const img = document.getElementById('cameraImage');
                img.src = 'data:image/jpeg;base64,' + data.data;
            } else if (data.type === 'sensor_data' && data.data) {
                updateSensorData(data.data);
            }
        }

        function updateConnectionStatus(connected) {
            const status = document.getElementById('connectionStatus');
            if (connected) {
                status.textContent = '接続済み';
                status.className = 'connection-status connected';
            } else {
                status.textContent = '切断中';
                status.className = 'connection-status disconnected';
            }
        }

        function updateSensorData(data) {
            // ステータス情報の更新
            document.getElementById('mode').textContent = data.mode || '-';
            document.getElementById('steering').textContent = data.steering_value?.toFixed(2) || '0.0';
            document.getElementById('throttle').textContent = data.throttle_value?.toFixed(2) || '0.0';
            document.getElementById('pauseStatus').textContent = data.pause_drive ? '一時停止中' : '正常動作';
            document.getElementById('timestamp').textContent = data.timestamp || '-';
            
            // センサー値の更新
            const ultrasonicRanges = data.ranges || {};
            
            // 個別センサー値の表示
            document.getElementById('sensorRrLH').textContent = 
                ultrasonicRanges.RrLH ? `${ultrasonicRanges.RrLH.toFixed(0)}mm` : '-';
            document.getElementById('sensorFrLH').textContent = 
                ultrasonicRanges.FrLH ? `${ultrasonicRanges.FrLH.toFixed(0)}mm` : '-';
            document.getElementById('sensorFrFR').textContent = 
                ultrasonicRanges.FrFR ? `${ultrasonicRanges.FrFR.toFixed(0)}mm` : '-';
            document.getElementById('sensorFrRH').textContent = 
                ultrasonicRanges.FrRH ? `${ultrasonicRanges.FrRH.toFixed(0)}mm` : '-';
            document.getElementById('sensorRrRH').textContent = 
                ultrasonicRanges.RrRH ? `${ultrasonicRanges.RrRH.toFixed(0)}mm` : '-';
            
            // センサー可視化の更新
            drawSensorVisualization(ultrasonicRanges);
        }

        function drawVehicle(centerX, centerY) {
            sensorCtx.save();
            sensorCtx.translate(centerX, centerY);
            
            // 外側の円を描画
            const circleRadius = 40;
            sensorCtx.fillStyle = '#4444ff';
            sensorCtx.beginPath();
            sensorCtx.arc(0, 0, circleRadius, 0, 2 * Math.PI);
            sensorCtx.fill();
            
            // 円の輪郭
            sensorCtx.strokeStyle = '#6666ff';
            sensorCtx.lineWidth = 3;
            sensorCtx.stroke();
            
            // 内側の三角形（進行方向を示す）
            sensorCtx.fillStyle = '#ffffff';
            sensorCtx.beginPath();
            sensorCtx.moveTo(0, -25);    // 頂点（前方）
            sensorCtx.lineTo(-18, 15);   // 左下
            sensorCtx.lineTo(18, 15);    // 右下
            sensorCtx.closePath();
            sensorCtx.fill();
            
            // 三角形の輪郭
            sensorCtx.strokeStyle = '#cccccc';
            sensorCtx.lineWidth = 2;
            sensorCtx.stroke();
            
            // センサー位置をマーク
            sensorCtx.fillStyle = '#ffff00';
            sensorCtx.strokeStyle = '#ff8800';
            sensorCtx.lineWidth = 1;
            Object.entries(sensorConfig).forEach(([key, config]) => {
                sensorCtx.beginPath();
                sensorCtx.arc(config.x, config.y, 5, 0, 2 * Math.PI);
                sensorCtx.fill();
                sensorCtx.stroke();
            });
            
            sensorCtx.restore();
        }

        function getDistanceColor(distance) {
            if (distance < 300) {
                return '#ff4444'; // 赤（危険）
            } else if (distance < 600) {
                return '#ffaa00'; // オレンジ（警告）
            } else {
                return '#44ff44'; // 緑（安全）
            }
        }

        function drawSensorFan(centerX, centerY, sensorKey, distance) {
            const config = sensorConfig[sensorKey];
            if (!config) return;

            sensorCtx.save();
            sensorCtx.translate(centerX + config.x, centerY + config.y);
            
            // 距離に応じた扇の長さ（最大200px）
            const maxRange = Math.min(distance / 5, 200);
            
            // 扇型を描画
            sensorCtx.fillStyle = getDistanceColor(distance);
            sensorCtx.globalAlpha = 0.6;
            sensorCtx.beginPath();
            sensorCtx.moveTo(0, 0);
            sensorCtx.arc(0, 0, maxRange, 
                   config.angle - config.arc / 2, 
                   config.angle + config.arc / 2);
            sensorCtx.closePath();
            sensorCtx.fill();
            
            // 扇の輪郭
            sensorCtx.globalAlpha = 1.0;
            sensorCtx.strokeStyle = getDistanceColor(distance);
            sensorCtx.lineWidth = 2;
            sensorCtx.stroke();
            
            // 距離テキスト
            sensorCtx.fillStyle = '#000000';  // 黒色に変更
            sensorCtx.font = '12px Arial';
            sensorCtx.textAlign = 'center';
            const textX = Math.cos(config.angle) * (maxRange + 20);
            const textY = Math.sin(config.angle) * (maxRange + 20);
            sensorCtx.fillText(`${distance}mm`, textX, textY);
            sensorCtx.fillText(config.label, textX, textY + 15);
            
            sensorCtx.restore();
        }

        function drawGrid(centerX, centerY) {
            sensorCtx.strokeStyle = '#333333';
            sensorCtx.lineWidth = 1;
            
            // 同心円グリッド（100mm間隔）
            for (let r = 50; r <= 400; r += 50) {
                sensorCtx.beginPath();
                sensorCtx.arc(centerX, centerY, r, 0, 2 * Math.PI);
                sensorCtx.stroke();
                
                // 距離ラベル
                sensorCtx.fillStyle = '#666666';
                sensorCtx.font = '10px Arial';
                sensorCtx.fillText(`${r * 5}mm`, centerX + r + 5, centerY);
            }
            
            // 十字線
            sensorCtx.beginPath();
            sensorCtx.moveTo(centerX - 400, centerY);
            sensorCtx.lineTo(centerX + 400, centerY);
            sensorCtx.moveTo(centerX, centerY - 400);
            sensorCtx.lineTo(centerX, centerY + 400);
            sensorCtx.stroke();
        }

        function drawSensorValues(sensorData) {
            // センサー値を左端に縦に表示（背景なし）
            const startX = 5;  // 左端に配置
            const startY = 5;   // キャンバス上端にさらに近づける
            const lineHeight = 25;
            
            // センサーの表示順序
            const sensorOrder = ['RrLH', 'FrLH', 'FrFR', 'FrRH', 'RrRH'];
            
            // センサー値をすべて黒色で表示
            sensorCtx.fillStyle = '#000000';
            sensorCtx.textAlign = 'left';
            sensorCtx.font = '12px Arial';
            
            sensorOrder.forEach((key, index) => {
                const value = sensorData[key];
                if (value !== undefined) {
                    const y = startY + 15 + (index * lineHeight);
                    // キー名と値を1行で表示
                    sensorCtx.fillText(`${key}: ${value} mm`, startX, y);
                }
            });
        }

        function drawSensorVisualization(ultrasonicRanges = {}) {
            if (!sensorCanvas || !sensorCtx) {
                console.error('Sensor canvas not initialized');
                return;
            }
            
            // キャンバスをクリア
            sensorCtx.clearRect(0, 0, sensorCanvas.width, sensorCanvas.height);
            
            const centerX = sensorCanvas.width / 2 + 40;  // チャートを右にずらす
            const centerY = sensorCanvas.height / 2 + 20;  // チャートを下にずらして上部の切れを回避
            
            // グリッドを描画
            drawGrid(centerX, centerY);
            
            // デフォルトセンサー値（初期表示用）
            const defaultSensorValues = {
                RrLH: 850,
                FrLH: 450,
                FrFR: 250,
                FrRH: 650,
                RrRH: 1200
            };
            
            // データがない場合はデフォルト値を使用
            const sensorData = (ultrasonicRanges && Object.keys(ultrasonicRanges).length > 0) ? ultrasonicRanges : defaultSensorValues;
            
            // センサー扇型を描画
            Object.entries(sensorData).forEach(([key, value]) => {
                if (sensorConfig[key] && value !== undefined) {
                    drawSensorFan(centerX, centerY, key, value);
                }
            });
            
            // 車体を最後に描画（上に表示）
            drawVehicle(centerX, centerY);
            
            // センサー値を左側に縦に表示
            drawSensorValues(sensorData);
        }

        // 制御関数
        function pauseDrive() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'control',
                    action: 'pause'
                }));
            }
        }

        function resumeDrive() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'control',
                    action: 'resume'
                }));
            }
        }

        // 初期化
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, initializing sensor canvas...');
            
            // キャンバス要素の確認と初期化
            sensorCanvas = document.getElementById('sensorCanvas');
            if (!sensorCanvas) {
                console.error('Canvas element not found!');
                return;
            }
            sensorCtx = sensorCanvas.getContext('2d');
            console.log('Canvas found:', sensorCanvas.width, 'x', sensorCanvas.height);
            
            // 初期描画
            try {
                drawSensorVisualization();
                console.log('Initial sensor visualization drawn');
            } catch (error) {
                console.error('Error drawing sensor visualization:', error);
            }
            
            // WebSocket接続開始
            connectWebSocket();
            
            // 定期的なデータ取得（WebSocketが使えない場合の fallback）
            setInterval(function() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    fetch('/get_data')
                        .then(response => response.json())
                        .then(data => updateSensorData(data))
                        .catch(error => console.error('Failed to fetch data:', error));
                }
            }, 1000);
        });
    </script>
</body>
</html>'''
        
        with open(index_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Created {index_html_path} with sensor visualization")


#------------------------------------------------------------------------------#
# Tornado WebSocket ハンドラー
#------------------------------------------------------------------------------#
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True
    
    def open(self):
        websocket_clients.add(self)
        print(f"[DEBUG] WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    def on_close(self):
        websocket_clients.discard(self)
        print(f"WebSocket client disconnected. Total clients: {len(websocket_clients)}")
    
    async def on_message(self, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'control':
                response = await self.handle_control(data)
                await self.write_message(json.dumps(response))
            elif data.get('type') == 'get_config':
                response = await self.handle_get_config()
                await self.write_message(json.dumps(response))
            elif data.get('type') == 'set_config':
                response = await self.handle_set_config(data)
                await self.write_message(json.dumps(response))
        except Exception as e:
            await self.write_message(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def handle_control(self, data):
        action = data.get('action')
        if action == 'pause':
            realtime_data['pause_drive'] = True
            return {'type': 'control_response', 'status': 'paused'}
        elif action == 'resume':
            realtime_data['pause_drive'] = False
            return {'type': 'control_response', 'status': 'resumed'}
        else:
            return {'type': 'control_response', 'status': 'unknown command'}
    
    async def handle_get_config(self):
        try:
            config_values = {
                key: getattr(config, key, None)
                for key in ALLOWED_CONFIG_KEYS
            }
            return {'type': 'config_response', 'status': 'ok', 'config': config_values}
        except Exception as e:
            return {'type': 'config_response', 'status': 'error', 'message': str(e)}
    
    async def handle_set_config(self, data):
        global set_config_reload
        config_data = data.get('config', {})
        updated_keys = []
        errors = []
        
        for key, value in config_data.items():
            if key not in ALLOWED_CONFIG_KEYS:
                errors.append(f"{key} is not an allowed config key.")
                continue
            
            # 型チェック
            if key == "PLAN":
                if value not in config.PLAN_LIST:
                    errors.append(f"Invalid PLAN: '{value}'. Must be in {config.PLAN_LIST}.")
                    continue
            
            if key == "HAND_SIDE":
                if value not in ["right", "left"]:
                    errors.append(f"Invalid HAND_SIDE: '{value}'. Must be 'right' or 'left'.")
                    continue
            
            # 数値系の変換
            if key in ["FORWARD_STRAIGHT", "FORWARD_CORNER", "STOP_RANGE", "BACKWARD_RANGE",
                       "DETECTION_RANGE", "RIGHT_LEFT_RANGE", "TARGET_RANGE", "TARGET_RANGE_ADJUSTMENT",
                       "K_P", "K_I", "K_D", "RECOVERY_TIME"]:
                try:
                    value = float(value)
                except ValueError:
                    errors.append(f"{key} must be float, got {value}")
                    continue
            
            if key in ["RIGHT_LEFT_RECORD_NUMBER", "RECOVERY_BRAKING"]:
                try:
                    value = int(value)
                except ValueError:
                    errors.append(f"{key} must be int, got {value}")
                    continue
            
            setattr(config, key, value)
            updated_keys.append(key)
        
        set_config_reload = True
        
        return {
            'type': 'config_response',
            'status': 'partial_success' if errors else 'ok',
            'updated_keys': updated_keys,
            'errors': errors
        }

#------------------------------------------------------------------------------#
# 非同期画像処理
#------------------------------------------------------------------------------#
def encode_image_to_base64(frame):
    """画像をbase64にエンコードする関数（超高速版）"""
    try:
        if FAST_MODE:
            # 超高速モード: 品質50、より小さなサイズに
            height, width = frame.shape[:2]
            if width > 320:  # 大きすぎる場合はリサイズ
                scale = 320 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # 品質を大幅に下げて高速化
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        else:
            # 通常モード
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        if ret:
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}")
    return None

def get_combined_frame():
    """カメラ画像を結合して返す（最適化版）"""
    cam0 = realtime_data.get("camera_image_0")
    cam1 = realtime_data.get("camera_image_1")

    # フレーム処理を簡素化
    if cam0 is not None and cam1 is not None:
        if hasattr(config, 'IMAGE_CONCAT_DIRECTION') and config.IMAGE_CONCAT_DIRECTION == "vertical":
            return np.vstack([cam0, cam1])
        else:
            return np.hstack([cam0, cam1])
    elif cam0 is not None:
        return cam0
    elif cam1 is not None:
        return cam1
    else:
        return None

async def broadcast_image():
    """画像をWebSocket経由でブロードキャストする（超高速版）"""
    global websocket_clients, last_frame_hash, last_encoded_image, frame_skip_counter
    
    if not websocket_clients:
        return
    
    try:
        # フレームスキップによる負荷軽減
        if FAST_MODE:
            frame_skip_counter += 1
            if frame_skip_counter <= SKIP_FRAME_COUNT:
                return  # このフレームはスキップ
            frame_skip_counter = 0  # カウンターリセット
        
        frame = get_combined_frame()
        if frame is None:
            return
        
        # より高速なフレーム変更検出（サイズベース + 簡易チェック）
        if FAST_MODE:
            # 高速モード: 簡易な変更検出
            frame_signature = frame.shape + (frame[::50, ::50].mean(),)  # サンプリングベース
        else:
            # 通常モード: ハッシュベース
            frame_signature = hash(frame.tobytes())
        
        if frame_signature == last_frame_hash and last_encoded_image is not None:
            # フレームが変更されていない場合はキャッシュを使用
            base64_image = last_encoded_image
        else:
            # 新しいフレームをエンコード
            loop = tornado.ioloop.IOLoop.current()
            base64_image = await loop.run_in_executor(executor, encode_image_to_base64, frame)
            if base64_image:
                last_frame_hash = frame_signature
                last_encoded_image = base64_image
        
        if base64_image:
            # より軽量なメッセージ形式
            if FAST_MODE:
                message = {
                    'type': 'image',
                    'data': base64_image
                    # timestampを省略して軽量化
                }
            else:
                message = {
                    'type': 'image',
                    'data': base64_image,
                    'timestamp': datetime.now().isoformat()
                }
            
            # バイナリ送信準備（JSONエンコードを最適化）
            message_json = json.dumps(message, separators=(',', ':'))  # 空白を削除
            
            # 並列でWebSocketクライアントに送信
            tasks = []
            for client in list(websocket_clients):  # コピーを作成
                tasks.append(_send_raw_message_to_client(client, message_json))
            
            # 並列実行
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 失敗したクライアントを削除
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        client = list(websocket_clients)[i] if i < len(websocket_clients) else None
                        if client:
                            websocket_clients.discard(client)
            
    except Exception as e:
        print(f"[ERROR] broadcast_image error: {e}")

async def _send_message_to_client(client, message):
    """個別クライアントへのメッセージ送信"""
    try:
        await client.write_message(json.dumps(message))
    except Exception:
        raise  # エラーを上位に伝播

async def _send_raw_message_to_client(client, message_json):
    """個別クライアントへの生JSON送信（高速版）"""
    try:
        await client.write_message(message_json)
    except Exception:
        raise  # エラーを上位に伝播

async def broadcast_sensor_data():
    """センサーデータをWebSocket経由でブロードキャストする"""
    global websocket_clients
    if websocket_clients:
        data_to_send = {}
        for k, v in realtime_data.items():
            if k not in ["camera_image_0", "camera_image_1"]:
                data_to_send[k] = v
        
        message = {
            'type': 'sensor_data',
            'data': data_to_send
        }
        
        disconnected_clients = set()
        for client in websocket_clients:
            try:
                await client.write_message(json.dumps(message))
            except Exception:
                disconnected_clients.add(client)
        
        websocket_clients -= disconnected_clients

#------------------------------------------------------------------------------#
# Tornado HTTP ハンドラー
#------------------------------------------------------------------------------#
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("monitor.html")

class GetDataHandler(tornado.web.RequestHandler):
    def get(self):
        data_to_return = {}
        for k, v in realtime_data.items():
            if k not in ["camera_image_0", "camera_image_1"]:
                data_to_return[k] = v
        self.write(json.dumps(data_to_return))
        self.set_header("Content-Type", "application/json")

class ControlHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            data = json.loads(self.request.body)
            action = data.get("action")
            
            if action == "pause":
                realtime_data["pause_drive"] = True
                self.write(json.dumps({"status": "paused"}))
            elif action == "resume":
                realtime_data["pause_drive"] = False
                self.write(json.dumps({"status": "resumed"}))
            else:
                self.set_status(400)
                self.write(json.dumps({"status": "unknown command"}))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"status": "error", "message": str(e)}))
        
        self.set_header("Content-Type", "application/json")

#------------------------------------------------------------------------------#
# 5) config.py の定数を変更
#------------------------------------------------------------------------------#
ALLOWED_CONFIG_KEYS = {
    # 変更を許可するキー名のセット
    "FORWARD_STRAIGHT",
    "FORWARD_CORNER",
    "STOP",
    "REVERSE",
    "LEFT",
    "NEUTRAL",
    "RIGHT",
    "STOP_RANGE",
    "BACKWARD_RANGE",
    "DETECTION_RANGE",
    "RIGHT_LEFT_RANGE",
    "TARGET_RANGE",
    "TARGET_RANGE_ADJUSTMENT",
    "K_P",
    "K_I",
    "K_D",
    "PLAN",
    "HAND_SIDE",
    "RIGHT_LEFT_RECORD_NUMBER",
    "RECOVERY_MODE",
    "RECOVERY_STREERING",
    "RECOVERY_TIME",
    "RECOVERY_BRAKING",
    "USE_PLOTTER",
    "MODEL_NAME",
}

class GetConfigHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            config_values = {
                key: getattr(config, key, None)
                for key in ALLOWED_CONFIG_KEYS
            }
            self.write(json.dumps({"status": "ok", "config": config_values}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"status": "error", "message": str(e)}))
        
        self.set_header("Content-Type", "application/json")


class SetConfigHandler(tornado.web.RequestHandler):
    def post(self):
        global set_config_reload
        
        try:
            data = json.loads(self.request.body)
            if not data:
                self.set_status(400)
                self.write(json.dumps({"status": "error", "message": "No JSON data"}))
                return
            
            updated_keys = []
            errors = []
            
            for key, value in data.items():
                if key not in ALLOWED_CONFIG_KEYS:
                    errors.append(f"{key} is not an allowed config key.")
                    continue
                
                # 型チェック
                if key == "PLAN":
                    if value not in config.PLAN_LIST:
                        errors.append(f"Invalid PLAN: '{value}'. Must be in {config.PLAN_LIST}.")
                        continue
                
                if key == "HAND_SIDE":
                    if value not in ["right", "left"]:
                        errors.append(f"Invalid HAND_SIDE: '{value}'. Must be 'right' or 'left'.")
                        continue
                
                # 数値系
                if key in ["FORWARD_STRAIGHT", "FORWARD_CORNER", "STOP_RANGE", "BACKWARD_RANGE",
                           "DETECTION_RANGE", "RIGHT_LEFT_RANGE", "TARGET_RANGE", "TARGET_RANGE_ADJUSTMENT",
                           "K_P", "K_I", "K_D", "RECOVERY_TIME"]:
                    try:
                        value = float(value)
                    except ValueError:
                        errors.append(f"{key} must be float, got {value}")
                        continue
                
                if key in ["RIGHT_LEFT_RECORD_NUMBER", "RECOVERY_BRAKING"]:
                    try:
                        value = int(value)
                    except ValueError:
                        errors.append(f"{key} must be int, got {value}")
                        continue
                
                setattr(config, key, value)
                updated_keys.append(key)
            
            set_config_reload = True
            
            if errors:
                self.set_status(400)
                response = {
                    "status": "partial_success",
                    "updated_keys": updated_keys,
                    "errors": errors
                }
            else:
                response = {
                    "status": "ok",
                    "updated_keys": updated_keys
                }
            
            self.write(json.dumps(response))
            
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({"status": "error", "message": str(e)}))
        
        self.set_header("Content-Type", "application/json")

#------------------------------------------------------------------------------#
# モデル管理
#------------------------------------------------------------------------------#
class GetModelsHandler(tornado.web.RequestHandler):
    def get(self):
        models_folder = './models'
        if not os.path.exists(models_folder):
            self.write(json.dumps({"models": []}))
            return
        
        models = [
            f for f in os.listdir(models_folder)
            if os.path.isfile(os.path.join(models_folder, f)) and not f.endswith('.png')
        ]
        
        self.write(json.dumps({"models": models}))
        self.set_header("Content-Type", "application/json")


#------------------------------------------------------------------------------#
# run.py からデータを更新するための関数
#------------------------------------------------------------------------------#
def update_data(mode=None,
                steering_value=None,
                throttle_value=None,
                ranges=None,
                imu_data=None,
                timestamp=None,
                camera_image_0=None,
                camera_image_1=None):
    """
    run.pyのメインループなどから呼び出されてリアルタイムデータを更新する。
    ここで受け取った値をrealtime_dataに格納し、/get_data で参照できるようにする。
    """
    if mode is not None:
        realtime_data["mode"] = mode
    if steering_value is not None:
        realtime_data["steering_value"] = steering_value
    if throttle_value is not None:
        realtime_data["throttle_value"] = throttle_value
    # rangesが渡された場合は優先、なければrangesを使用（後方互換性）
    sensor_ranges = ranges if ranges is not None else ranges
    if sensor_ranges is not None:
        realtime_data["ranges"] = sensor_ranges
    if imu_data is not None:
        realtime_data["imu_data"] = imu_data
    if timestamp is not None:
        realtime_data["timestamp"] = convert_timestamp(timestamp)
    if camera_image_0 is not None:
        realtime_data["camera_image_0"] = camera_image_0
    if camera_image_1 is not None:
        realtime_data["camera_image_1"] = camera_image_1

#------------------------------------------------------------------------------#
# 表示のための補助関数
#------------------------------------------------------------------------------#
def convert_timestamp(timestamp):
    # タイムスタンプ文字列の長さチェック
    if len(timestamp) != 20:
        raise ValueError("タイムスタンプの形式が正しくありません。20桁である必要があります。")

    # 年月日、時分秒、ミリ秒を抽出
    year = int(timestamp[0:4])
    month = int(timestamp[4:6])
    day = int(timestamp[6:8])
    hour = int(timestamp[8:10])
    minute = int(timestamp[10:12])
    second = int(timestamp[12:14])
    microsecond = int(timestamp[14:20])  # ミリ秒以下は6桁で解釈

    # datetimeオブジェクトを作成
    dt = datetime(year, month, day, hour, minute, second, microsecond)

    # 見やすいフォーマットに変換
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ミリ秒まで表示
    return formatted_time


#------------------------------------------------------------------------------#
# Tornado アプリケーション起動
#------------------------------------------------------------------------------#
def make_app():
    # templatesフォルダとindex.htmlを作成（存在しない場合）
    create_template_if_needed()
    
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws", WebSocketHandler),
        (r"/get_data", GetDataHandler),
        (r"/control", ControlHandler),
        (r"/get_config", GetConfigHandler),
        (r"/set_config", SetConfigHandler),
        (r"/get_models", GetModelsHandler),
    ], template_path="templates", static_path="static")

async def periodic_broadcast():
    """定期的なデータブロードキャスト"""
    while not shutdown_signal:
        try:
            # 画像とセンサーデータを並行で送信
            await asyncio.gather(
                broadcast_image(),
                broadcast_sensor_data(),
                return_exceptions=True
            )
            if FAST_MODE:
                await asyncio.sleep(0.02)  # 高速モード
            else:
                await asyncio.sleep(0.05)  # 通常モード
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)

def open_browser(url, delay=1.5):
    """ブラウザを開く関数（遅延実行）"""
    time.sleep(delay)
    try:
        # プラットフォームに応じて適切な方法でブラウザを開く
        if platform.system() == 'Linux':
            # Jetson (ARM Linux) の場合、環境変数DISPLAYをチェック
            if 'DISPLAY' in os.environ:
                webbrowser.open(url)
            else:
                print(f"No display found. Please open browser manually: {url}")
        else:
            webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open browser manually: {url}")

def run(host="0.0.0.0", port=8000, debug=False, open_browser_on_start=True):
    """
    Tornadoアプリケーションを起動
    """
    app = make_app()
    app.listen(port, address=host)
    
    # 定期ブロードキャストタスクを開始
    tornado.ioloop.IOLoop.current().spawn_callback(periodic_broadcast)
    
    # print(f"Tornado server started on {host}:{port}")
    print(f"Sensor visualization available at: http://localhost:{port}\n")
    
    # ブラウザを自動的に開く
    if open_browser_on_start:
        url = f"http://localhost:{port}"
        browser_thread = threading.Thread(target=open_browser, args=(url,), daemon=True)
        browser_thread.start()
    
    tornado.ioloop.IOLoop.current().start()

def init_cameras_standalone():
    """単独起動時のカメラ初期化（config.pyとcamera.pyを使用）"""
    global camera_instances
    import camera
    
    print(f"Initializing cameras with config: {config.IMAGE_W}x{config.IMAGE_H}@{config.CAMERA_FRAMERATE}fps")
    print(f"Active sensors: {config.ACTIVE_SENSORS}")
    
    # config.pyのACTIVE_SENSORS設定に基づいてカメラを初期化
    if "camera_0" in config.ACTIVE_SENSORS:
        try:
            camera_instances["camera_0"] = camera.create_camera(device_id=0)
            print(f"Camera 0 initialized: {config.IMAGE_W}x{config.IMAGE_H}")
        except Exception as e:
            print(f"Failed to initialize camera 0: {e}")
    
    if "camera_1" in config.ACTIVE_SENSORS:
        try:
            camera_instances["camera_1"] = camera.create_camera(device_id=1)
            print(f"Camera 1 initialized: {config.IMAGE_W}x{config.IMAGE_H}")
        except Exception as e:
            print(f"Failed to initialize camera 1: {e}")

def init_sensors_standalone():
    """単独起動時のセンサー初期化（ultrasonicセンサーなど）"""
    global active_sensor_instances
    active_sensor_instances = {}
    
    try:
        # ultrasonicセンサーが設定されている場合は初期化を試行
        if "ultrasonic" in config.ACTIVE_SENSORS and hasattr(config, 'ULTRASONIC_SENSOR_LIST'):
            print(f"Attempting to initialize ultrasonic sensors: {config.ULTRASONIC_SENSOR_LIST}")
            
            # data_aggregatorの初期化を試行
            try:
                from data_aggregator import DataAggregator
                global data_aggregator_instance
                data_aggregator_instance = DataAggregator()
                print("DataAggregator initialized for ultrasonic sensors")
                return True
            except Exception as e:
                print(f"Failed to initialize DataAggregator for ultrasonic: {e}")
                return False
        else:
            print("No ultrasonic sensors configured in ACTIVE_SENSORS")
            return False
    except Exception as e:
        print(f"Error initializing sensors: {e}")
        return False

def update_cameras_standalone():
    """単独起動時のカメラ更新ループ（config.py設定に基づく）"""
    global camera_instances, shutdown_signal, data_aggregator_instance
    
    # config.pyのフレームレートに基づいたスリープ時間を計算
    frame_interval = 1.0 / config.CAMERA_FRAMERATE if hasattr(config, 'CAMERA_FRAMERATE') else 0.033
    
    while not shutdown_signal:
        try:
            # config.pyのACTIVE_SENSORSに基づいてカメラを更新
            if "camera_0" in config.ACTIVE_SENSORS and "camera_0" in camera_instances:
                ret, frame = camera_instances["camera_0"].read()
                if ret and frame is not None:
                    # config.pyのフリップ設定を適用（camera.pyで既に処理済み）
                    realtime_data["camera_image_0"] = frame
            
            if "camera_1" in config.ACTIVE_SENSORS and "camera_1" in camera_instances:
                ret, frame = camera_instances["camera_1"].read()
                if ret and frame is not None:
                    # config.pyのフリップ設定を適用（camera.pyで既に処理済み）
                    realtime_data["camera_image_1"] = frame
            
            # 実際のultrasonicセンサー値を取得（利用可能な場合）
            ranges = {}
            if 'data_aggregator_instance' in globals() and data_aggregator_instance is not None:
                try:
                    # センサー値を更新
                    data_aggregator_instance.update_sensors()
                    
                    # ultrasonicセンサー値を取得
                    if hasattr(config, 'ULTRASONIC_SENSOR_LIST'):
                        for us_name in config.ULTRASONIC_SENSOR_LIST:
                            try:
                                value = data_aggregator_instance.get_latest_sensor_value(us_name)
                                if value is not None:
                                    ranges[us_name] = value
                            except Exception as e:
                                print(f"Failed to get {us_name} sensor value: {e}")
                except Exception as e:
                    print(f"Failed to update sensors: {e}")
            
            # データの更新（run.pyと同様のフォーマット）
            realtime_data["mode"] = "standalone_test"
            realtime_data["steering_value"] = 0.0
            realtime_data["throttle_value"] = 0.0
            
            # ranges: 実際の値がある場合はそれを使用、ない場合はテストデータ
            if ranges:
                realtime_data["ranges"] = ranges
                print(f"Using real ultrasonic data: {ranges}")
            elif "ranges" not in realtime_data or not realtime_data["ranges"]:
                realtime_data["ranges"] = {
                    "RrLH": 850.0,  # 真左
                    "FrLH": 450.0,  # 前方左
                    "FrFR": 250.0,  # 前方（近い）
                    "FrRH": 650.0,  # 前方右
                    "RrRH": 1200.0  # 真右
                }
            
            realtime_data["timestamp"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
            
            time.sleep(frame_interval)  # config.pyのフレームレートに合わせる
            
        except Exception as e:
            print(f"[ERROR] Camera update error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

def cleanup_cameras_standalone():
    """単独起動時のカメラクリーンアップ"""
    global camera_instances
    
    for name, camera in camera_instances.items():
        if camera:
            try:
                camera.cleanup()
                print(f"{name} cleaned up")
            except Exception as e:
                print(f"Failed to cleanup {name}: {e}")
    
    camera_instances.clear()
    
    # 追加のクリーンアップ - GST リソースの完全な解放
    import time
    import gc
    time.sleep(0.2)  # GST パイプラインの完全な停止を待つ
    gc.collect()  # ガベージコレクションを強制実行

if __name__ == '__main__':
    print("Monitor starting in standalone mode...")
    print(f"Using config.py settings:")
    print(f"  Image size: {config.IMAGE_W}x{config.IMAGE_H}")
    print(f"  Framerate: {config.CAMERA_FRAMERATE}fps")
    print(f"  Active sensors: {config.ACTIVE_SENSORS}")
    print(f"  Image concat direction: {getattr(config, 'IMAGE_CONCAT_DIRECTION', 'horizontal')}")
    print(f"  Sensor visualization: Enabled")
    
    # config.pyの設定に基づいてカメラを初期化
    init_cameras_standalone()
    
    # センサーの初期化を試行
    sensors_initialized = init_sensors_standalone()
    if sensors_initialized:
        print("Real ultrasonic sensors will be used when available")
    else:
        print("Using test ultrasonic data only")
    
    # カメラ更新スレッドを開始
    camera_update_thread = threading.Thread(target=update_cameras_standalone, daemon=True)
    camera_update_thread.start()
    
    try:
        # Tornadoアプリを起動（ブラウザ自動起動あり）
        run(debug=False, open_browser_on_start=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # 終了処理
        shutdown_signal = True
        if camera_update_thread:
            camera_update_thread.join(timeout=2.0)
        cleanup_cameras_standalone()
        print("Cleanup complete")