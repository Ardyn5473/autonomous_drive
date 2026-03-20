"""
Training Manager module for data_viewer.
Handles neural network training, model management, and MLflow integration.

学習の実行・モデルの保存/読込・MLflow記録を一括管理するモジュール。

学習の全体フロー:
  1. データの前処理 (_build_input_matrix)
     走行ログから入力行列 X と教師データ y を作る
  2. モデル構築 (NeuralNetwork)
     入力→隠れ層→出力のネットワークを作る
  3. 学習ループ (_train_model_internal)
     エポックごとに「順伝播→損失計算→逆伝播→パラメータ更新」を繰り返す
  4. モデル保存
     学習済みの重みを .pth ファイルに保存する
"""

import os
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim                        # 最適化アルゴリズム（Adam等）
from torch.utils.data import DataLoader, TensorDataset  # ミニバッチ管理
from send2trash import send2trash

import mlflow              # 実験管理ツール: パラメータ・損失・モデルを記録
import mlflow.pytorch

from neural_network import NeuralNetwork, TrainingConfig


class TrainingManager:
    """
    学習のライフサイクル全体を管理するクラス。

    責務:
    - 学習の開始/停止/進捗管理（バックグラウンドスレッドで実行）
    - モデルファイルの保存/読込/削除
    - 学習済みモデルによる推論（predict）
    - MLflow への実験ログ記録
    """

    def __init__(self, models_dir: str = "models", mlruns_dir: str = "mlruns"):
        self.models_dir = Path(models_dir)    # 学習済みモデルの保存先
        self.mlruns_dir = Path(mlruns_dir)    # MLflow の実験ログ保存先

        # ディレクトリがなければ作成
        self.models_dir.mkdir(exist_ok=True)
        self.mlruns_dir.mkdir(exist_ok=True)

        # --- 学習の進捗状態（UIのプログレスバーに使う） ---
        self.progress = {
            "is_training": False,       # 現在学習中かどうか
            "current_epoch": 0,         # 現在のエポック番号
            "total_epochs": 0,          # 合計エポック数
            "start_time": None,         # 学習開始時刻
            "train_losses": [],         # 訓練損失の履歴
            "val_losses": [],           # 検証損失の履歴
            "should_stop": False        # 停止リクエストフラグ
        }

        # --- 現在ロード中のモデル ---
        self.current_model: Optional[NeuralNetwork] = None   # PyTorchモデル本体
        self.current_model_info: Optional[Dict] = None       # メタ情報（構造・センサー等）

        # --- MLflow の初期設定 ---
        mlflow.set_tracking_uri(str(self.mlruns_dir))
        mlflow.set_experiment("data_viewer_training")

        # 学習はバックグラウンドスレッドで実行する（UIをブロックしないため）
        self._training_thread: Optional[threading.Thread] = None

    # =================================================================
    # データ準備
    # =================================================================

    @staticmethod
    def _build_input_matrix(records: List[Dict], selected_sensors: List[str],
                            data_path: Optional[str] = None) -> np.ndarray:
        """
        走行ログのレコード群から、ニューラルネットワークへの入力行列 X を構築する。

        各レコード（= 1フレームの走行データ）から選択されたセンサーの値を取り出し、
        1行にまとめて行列にする。

        対応するセンサー種別:
          - スカラーセンサー（超音波距離など）: そのまま1つの数値として追加
          - 配列センサー（LiDAR .npy）: ファイルを読み込み、全要素を展開して追加

        Args:
            records:          走行ログのレコードのリスト
            selected_sensors: 使用するセンサーキーのリスト（例: ['ultrasonic/FrFR', ...]）
            data_path:        .npy ファイルのベースパス

        Returns:
            X: numpy配列 [レコード数, 特徴量数]
               例: 5つの超音波センサー → shape = (1000, 5)
               例: LiDAR(360点) + 超音波5つ → shape = (1000, 365)
        """
        rows = []
        for d in records:
            row = []  # この1レコードの入力ベクトル
            for sensor in selected_sensors:
                value = d.get(sensor, 0)

                if isinstance(value, str) and value.endswith('.npy') and data_path:
                    # --- 配列センサー（LiDAR）の場合 ---
                    # .npy ファイルから距離配列を読み込み、1次元に展開して追加
                    npy_path = os.path.join(data_path, 'lidar', value)
                    arr = np.load(npy_path).flatten()  # 例: (360,) の配列
                    row.extend(arr.tolist())            # 360個の数値を行に追加
                else:
                    # --- スカラーセンサーの場合 ---
                    # 1つの数値をそのまま追加（None は 0.0 に変換）
                    row.append(float(value) if value is not None else 0.0)

            rows.append(row)

        # リストのリストを NumPy 配列に変換（float32 = 32ビット浮動小数点）
        return np.array(rows, dtype=np.float32)

    def _generate_model_name(self, input_size: int, hidden_layers: List[int], output_size: int) -> str:
        """Generate model name with plan, datetime, and architecture."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create architecture string
        hidden_str = "_".join(map(str, hidden_layers))
        if hidden_str:
            architecture = f"{input_size}_{hidden_str}_{output_size}"
        else:
            architecture = f"{input_size}_{output_size}"

        return f"nn_{date_str}_{architecture}.pth"

    # =================================================================
    # 学習の開始（公開API）
    # =================================================================

    def train_model(self, config: TrainingConfig, records: List[Dict],
                    data_path: Optional[str] = None,
                    continue_training: bool = False) -> Dict[str, Any]:
        """
        モデル学習をバックグラウンドスレッドで開始する。

        この関数自体はすぐに返り、実際の学習は _train_model_internal() で行われる。
        UIはポーリングで get_progress() を呼んで進捗を取得する。

        Args:
            config:            学習設定（エポック数・バッチサイズ等）
            records:           走行データのレコード群
            data_path:         データファイルのベースパス
            continue_training: True = 既存モデルから追加学習する
        """
        if self.progress["is_training"]:
            return {"error": "Training already in progress"}

        if not records:
            return {"error": "No data loaded"}

        if continue_training and self.current_model is None:
            return {"error": "No model loaded for continuation"}

        # スレッド開始前に進捗を初期化（競合状態を防ぐ）
        prev_epochs = 0
        if continue_training and self.current_model_info:
            prev_epochs = len(self.current_model_info.get("train_losses", []))

        self.progress.update({
            "is_training": True,
            "current_epoch": prev_epochs,
            "total_epochs": prev_epochs + config.epochs,
            "start_time": time.time(),
            "train_losses": [],
            "val_losses": [],
            "should_stop": False
        })

        # バックグラウンドスレッドで学習を開始
        # （Flask のリクエスト処理をブロックしないため）
        self._training_thread = threading.Thread(
            target=self._train_model_internal,
            args=(config, records, data_path, continue_training)
        )
        self._training_thread.start()

        return {
            "message": "Training started",
            "is_training": True,
            "total_epochs": prev_epochs + config.epochs
        }

    # =================================================================
    # 学習ループ本体（バックグラウンドスレッドで実行）
    # =================================================================

    def _train_model_internal(self, config: TrainingConfig, records: List[Dict],
                               data_path: Optional[str] = None,
                               continue_training: bool = False):
        """
        学習ループの本体。バックグラウンドスレッドで実行される。

        処理の流れ:
          1. データの前処理（フィルタリング・正規化）
          2. DataLoader の構築（ミニバッチに分割）
          3. モデルの初期化（新規 or 既存モデルの継続）
          4. エポックループ（順伝播 → 損失計算 → 逆伝播 → パラメータ更新）
          5. モデルの保存（ローカル + MLflow）
        """
        try:
            print(f"[Training] Starting training with {len(records)} records")
            print(f"[Training] Selected sensors: {config.selected_sensors}")
            print(f"[Training] Config: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")

            # --- MLflow の実行名を決定 ---
            if continue_training and self.current_model_info:
                parent_model = self.current_model_info.get("filename", "unknown").replace('.pth', '')
                run_name = f"{parent_model}_continued_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                run_name = f"new_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # MLflow に学習パラメータと結果を記録する
            with mlflow.start_run(run_name=run_name):
                # ---- ハイパーパラメータを MLflow に記録 ----
                mlflow.log_param("hidden_layers", config.hidden_layers)
                mlflow.log_param("epochs", config.epochs)
                mlflow.log_param("batch_size", config.batch_size)
                mlflow.log_param("learning_rate", config.learning_rate)
                mlflow.log_param("selected_sensors", config.selected_sensors)
                mlflow.log_param("selected_outputs", config.selected_outputs)
                # input_size は X 構築後に記録（センサー種別で変わるため）
                mlflow.log_param("output_size", len(config.selected_outputs))
                mlflow.log_param("use_dropout", config.use_dropout)
                mlflow.log_param("dropout_rate", config.dropout_rate)
                mlflow.log_param("is_continuation", continue_training)

                if continue_training and self.current_model_info:
                    mlflow.log_param("parent_model", self.current_model_info.get("filename", "unknown"))
                    mlflow.log_param("parent_run_id", self.current_model_info.get("mlflow_run_id", "unknown"))
                    mlflow.log_param("previous_epochs", len(self.current_model_info.get("train_losses", [])))

                # ============================================================
                # ステップ1: データの前処理
                # ============================================================
                filtered_data = records

                # 範囲フィルタ: 指定されたインデックス範囲のみ使用
                if config.data_range_start is not None or config.data_range_end is not None:
                    start_idx = config.data_range_start if config.data_range_start is not None else 0
                    end_idx = config.data_range_end if config.data_range_end is not None else len(records) - 1
                    filtered_data = [d for i, d in enumerate(filtered_data) if start_idx <= i <= end_idx]
                    print(f"[Training] After range filter ({start_idx}-{end_idx}): {len(filtered_data)} records")

                # ダウンサンプリング: N件に1件だけ使う（データ量削減）
                if config.downsample_rate > 1:
                    filtered_data = filtered_data[::config.downsample_rate]
                    print(f"[Training] After downsampling (1/{config.downsample_rate}): {len(filtered_data)} records")

                # 削除マークされたデータを除外
                valid_data = [d for d in filtered_data
                              if d.get('_index', d.get('_absolute_index', 0)) not in config.deleted_indexes]
                print(f"[Training] After removing deleted indexes: {len(valid_data)} records")

                # データ設定を MLflow に記録
                mlflow.log_param("data_range_start", config.data_range_start)
                mlflow.log_param("data_range_end", config.data_range_end)
                mlflow.log_param("downsample_rate", config.downsample_rate)
                mlflow.log_param("total_training_samples", len(valid_data))

                if len(valid_data) == 0:
                    print("[Training] ERROR: No valid data after filtering")
                    self.progress["is_training"] = False
                    return

                # センサーの存在チェック（警告用）
                if valid_data:
                    sample_keys = list(valid_data[0].keys())
                    print(f"[Training] Available keys in data: {sample_keys[:10]}...")
                    missing_sensors = [s for s in config.selected_sensors if s not in sample_keys]
                    if missing_sensors:
                        print(f"[Training] WARNING: Missing sensors in data: {missing_sensors}")

                # ---- 入力行列 X の構築 ----
                # X: [サンプル数, 特徴量数] の2次元配列
                # 例: 1000件のデータ、5つの超音波センサー → X.shape = (1000, 5)
                X = self._build_input_matrix(valid_data, config.selected_sensors, data_path)
                print(f"[Training] X shape: {X.shape}")
                mlflow.log_param("input_size", X.shape[1])

                # ---- 教師データ y の構築 ----
                # y: [サンプル数, 出力数] の2次元配列
                # 例: 1000件、出力が angle+throttle → y.shape = (1000, 2)
                print(f"[Training] Selected outputs: {config.selected_outputs}")
                y = np.array([
                    [d.get(output_key, 0) for output_key in config.selected_outputs]
                    for d in valid_data
                ], dtype=np.float32)
                print(f"[Training] y shape: {y.shape}")

                if len(X) == 0:
                    print("[Training] ERROR: No training data (X is empty)")
                    self.progress["is_training"] = False
                    return

                # ---- 入力データの正規化（標準化） ----
                # 各特徴量を「平均0、標準偏差1」に変換する
                # これにより学習が安定し、収束が速くなる
                # 例: 超音波の値が 0~400cm → 平均200, std=100 なら → (値-200)/100
                X_mean, X_std = X.mean(axis=0), X.std(axis=0)
                X_normalized = (X - X_mean) / (X_std + 1e-8)  # 1e-8 はゼロ除算防止

                # ============================================================
                # ステップ2: PyTorch の DataLoader を構築
                # ============================================================

                # NumPy配列 → PyTorchテンソルに変換
                X_tensor = torch.FloatTensor(X_normalized)
                y_tensor = torch.FloatTensor(y)

                # TensorDataset: X と y をペアにしたデータセット
                dataset = TensorDataset(X_tensor, y_tensor)

                # データを 訓練用(80%) と 検証用(20%) にランダム分割
                # 訓練用: パラメータの更新に使う
                # 検証用: 学習に使わず、汎化性能（未知データへの精度）を評価する
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

                # DataLoader: データセットをミニバッチに分割してイテレーションする
                # shuffle=True: エポックごとにデータ順をランダム化（学習の偏り防止）
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

                # ============================================================
                # ステップ3: モデルとオプティマイザの初期化
                # ============================================================
                input_size = X.shape[1]
                output_size = len(config.selected_outputs)

                if continue_training and self.current_model is not None:
                    # 既存モデルから継続学習: 学習済みの重みをそのまま使う
                    model = self.current_model
                    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                else:
                    # 新規モデルを作成（重みはランダム初期化）
                    model = NeuralNetwork(
                        input_size=input_size,
                        output_size=output_size,
                        hidden_layers=config.hidden_layers,
                        use_dropout=config.use_dropout,
                        dropout_rate=config.dropout_rate
                    )
                    # Adam: 適応的学習率の最適化アルゴリズム
                    # SGD より収束が速く、ハイパーパラメータの調整が楽
                    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

                # MSELoss: 平均二乗誤差（Mean Squared Error）
                # 回帰問題の標準的な損失関数
                # 予測値と正解値の差の二乗の平均 = (predicted - actual)^2 の平均
                criterion = nn.MSELoss()

                # 継続学習の場合、前回の損失履歴を引き継ぐ
                if continue_training and self.current_model_info:
                    prev_train_losses = self.current_model_info.get("train_losses", [])
                    prev_val_losses = self.current_model_info.get("val_losses", [])
                    prev_epochs = len(prev_train_losses)
                    self.progress["train_losses"] = prev_train_losses.copy()
                    self.progress["val_losses"] = prev_val_losses.copy()
                else:
                    prev_train_losses = []
                    prev_val_losses = []
                    prev_epochs = 0

                # ============================================================
                # ステップ4: 学習ループ（ここが学習の核心部分）
                # ============================================================
                train_losses = prev_train_losses.copy()
                val_losses = prev_val_losses.copy()

                print(f"[Training] Starting training loop for {config.epochs} epochs")
                for epoch in range(config.epochs):
                    # UIから停止リクエストがあれば中断
                    if self.progress["should_stop"]:
                        print(f"[Training] Stopped at epoch {epoch}")
                        break

                    self.progress["current_epoch"] = prev_epochs + epoch + 1

                    # ---- 訓練フェーズ ----
                    model.train()  # Dropout を有効化（学習モード）
                    train_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        if self.progress["should_stop"]:
                            break

                        # (1) 勾配をリセット（前回の計算結果をクリア）
                        optimizer.zero_grad()

                        # (2) 順伝播: 入力データをモデルに通して予測値を得る
                        outputs = model(batch_X)

                        # (3) 損失計算: 予測値と正解値の誤差を計算
                        loss = criterion(outputs, batch_y)

                        # (4) 逆伝播: 損失から各パラメータの勾配を計算
                        #     （自動微分: PyTorchが計算グラフを辿って偏微分を求める）
                        loss.backward()

                        # (5) パラメータ更新: 勾配の方向にパラメータを少し動かす
                        #     weight = weight - learning_rate * gradient
                        optimizer.step()

                        train_loss += loss.item()  # .item() でテンソル→Python数値に変換

                    if self.progress["should_stop"]:
                        break

                    # ---- 検証フェーズ ----
                    model.eval()   # Dropout を無効化（評価モード）
                    val_loss = 0.0
                    with torch.no_grad():  # 勾配計算を無効化（メモリ節約・高速化）
                        for batch_X, batch_y in val_loader:
                            if self.progress["should_stop"]:
                                break
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()

                    if self.progress["should_stop"]:
                        break

                    # エポック全体の平均損失を計算
                    avg_train_loss = train_loss / len(train_loader)
                    avg_val_loss = val_loss / len(val_loader)

                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)

                    # 進捗を更新（UIがポーリングで読み取る）
                    self.progress["train_losses"] = train_losses.copy()
                    self.progress["val_losses"] = val_losses.copy()

                    # 10エポックごと、または最初と最後にログ出力
                    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
                        print(f"[Training] Epoch {prev_epochs + epoch + 1}/{prev_epochs + config.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                    # 損失を MLflow に記録（学習曲線の可視化用）
                    mlflow.log_metric("train_loss", avg_train_loss, step=prev_epochs + epoch)
                    mlflow.log_metric("val_loss", avg_val_loss, step=prev_epochs + epoch)

                print(f"[Training] Training loop completed. Total epochs: {len(train_losses)}")

                # ============================================================
                # ステップ5: 学習済みモデルの保存
                # ============================================================

                # ファイル名を生成（例: nn_20240101_120000_5_64_32_2.pth）
                model_name = self._generate_model_name(input_size, config.hidden_layers, output_size)

                # メモリ上のモデル情報を更新
                self.current_model = model
                self.current_model_info = {
                    "filename": model_name,
                    "architecture": {
                        "input_size": input_size,
                        "hidden_layers": config.hidden_layers,
                        "output_size": output_size
                    },
                    "selected_sensors": config.selected_sensors,
                    "selected_outputs": config.selected_outputs,
                    "use_dropout": config.use_dropout,
                    "dropout_rate": config.dropout_rate,
                    "normalization_params": {
                        "X_mean": X_mean.tolist(),   # 推論時にも同じ正規化を適用するため保存
                        "X_std": X_std.tolist()
                    },
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }

                # MLflow にモデルを保存
                input_example = X_normalized[:1]
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name=model_name.replace('.pth', '')
                )

                # ローカルに .pth ファイルとして保存
                # state_dict = モデルの全パラメータ（重みとバイアス）の辞書
                model_path = self.models_dir / model_name
                model_data = {
                    'model_state_dict': model.state_dict(),
                    'input_size': input_size,
                    'output_size': output_size,
                    'hidden_layers': config.hidden_layers,
                    'use_dropout': config.use_dropout,
                    'dropout_rate': config.dropout_rate,
                    'selected_sensors': config.selected_sensors,
                    'selected_outputs': config.selected_outputs,
                    'normalization_params': {
                        'X_mean': X_mean.tolist(),
                        'X_std': X_std.tolist()
                    },
                    'mlflow_run_id': mlflow.active_run().info.run_id,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'is_continued': continue_training,
                    'total_epochs': len(train_losses)
                }

                if continue_training and self.current_model_info:
                    model_data['parent_model'] = self.current_model_info.get("filename", "unknown")
                    model_data['parent_run_id'] = self.current_model_info.get("mlflow_run_id", "unknown")
                    model_data['continuation_history'] = {
                        'started_from_epoch': prev_epochs,
                        'added_epochs': config.epochs
                    }

                torch.save(model_data, model_path)

                # 正規化パラメータも MLflow に記録
                normalization_params = {
                    "X_mean": X_mean.tolist(),
                    "X_std": X_std.tolist()
                }
                mlflow.log_dict(normalization_params, "normalization_params.json")

        except Exception as e:
            import traceback
            print(f"[Training] ERROR: {e}")
            print(f"[Training] Traceback:\n{traceback.format_exc()}")
        finally:
            print("[Training] Training finished (finally block)")
            self.progress["is_training"] = False

    def stop_training(self) -> Dict[str, Any]:
        """Request training to stop."""
        self.progress["should_stop"] = True
        return {"message": "Training stop requested"}

    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return self.progress.copy()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models."""
        models = []

        for model_file in self.models_dir.glob("*.pth"):
            try:
                model_data = torch.load(model_file, map_location="cpu")
                models.append({
                    "filename": model_file.name,
                    "path": str(model_file),
                    "architecture": {
                        "input_size": model_data.get("input_size", 0),
                        "hidden_layers": model_data.get("hidden_layers", []),
                        "output_size": model_data.get("output_size", 0)
                    },
                    "selected_sensors": model_data.get("selected_sensors", []),
                    "selected_outputs": model_data.get("selected_outputs", ["user/angle", "user/throttle"]),
                    "use_dropout": model_data.get("use_dropout", True),
                    "dropout_rate": model_data.get("dropout_rate", 0.2),
                    "total_epochs": model_data.get("total_epochs", len(model_data.get("train_losses", []))),
                    "is_continued": model_data.get("is_continued", False),
                    "parent_model": model_data.get("parent_model", None)
                })
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
                continue

        # Sort by date and time (newest first)
        # Format: nn_YYYYMMDD_HHMMSS_architecture.pth
        def extract_datetime(filename):
            try:
                parts = filename.replace('.pth', '').split('_')
                if len(parts) >= 3:
                    return (parts[1], parts[2])  # (YYYYMMDD, HHMMSS)
            except (ValueError, IndexError):
                pass
            return ("0", "0")

        models.sort(key=lambda x: extract_datetime(x["filename"]), reverse=True)
        return models

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a model from file."""
        model_file = Path(model_path)
        if not model_file.exists():
            return {"error": "Model file not found"}

        try:
            model_data = torch.load(model_file, map_location="cpu")

            # Rebuild model
            model = NeuralNetwork(
                input_size=model_data["input_size"],
                output_size=model_data["output_size"],
                hidden_layers=model_data["hidden_layers"],
                use_dropout=model_data.get("use_dropout", True),
                dropout_rate=model_data.get("dropout_rate", 0.2)
            )
            model.load_state_dict(model_data["model_state_dict"])

            self.current_model = model
            self.current_model_info = {
                "filename": model_file.name,
                "architecture": {
                    "input_size": model_data["input_size"],
                    "hidden_layers": model_data["hidden_layers"],
                    "output_size": model_data["output_size"]
                },
                "selected_sensors": model_data.get("selected_sensors", []),
                "selected_outputs": model_data.get("selected_outputs", ["user/angle", "user/throttle"]),
                "use_dropout": model_data.get("use_dropout", True),
                "dropout_rate": model_data.get("dropout_rate", 0.2),
                "normalization_params": model_data.get("normalization_params", {}),
                "mlflow_run_id": model_data.get("mlflow_run_id", None),
                "train_losses": model_data.get("train_losses", []),
                "val_losses": model_data.get("val_losses", [])
            }

            return {
                "message": "Model loaded successfully",
                "model_info": self.current_model_info,
                "train_losses": model_data.get("train_losses", []),
                "val_losses": model_data.get("val_losses", []),
                "mlflow_run_id": model_data.get("mlflow_run_id", None),
                "is_continued": model_data.get("is_continued", False),
                "parent_model": model_data.get("parent_model", None),
                "total_epochs": model_data.get("total_epochs", len(model_data.get("train_losses", [])))
            }

        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

    def delete_model(self, model_path: str) -> Dict[str, Any]:
        """Move a model file to trash."""
        model_file = Path(model_path)
        if not model_file.exists():
            return {"error": "Model file not found"}

        try:
            filename = model_file.name

            # If the deleted model is currently loaded, clear it
            if self.current_model_info and self.current_model_info.get("filename") == filename:
                self.current_model = None
                self.current_model_info = None

            # Move to trash instead of permanent delete
            send2trash(str(model_file))

            return {
                "message": f"Model '{filename}' moved to trash",
                "deleted_file": filename
            }

        except Exception as e:
            return {"error": f"Failed to delete model: {str(e)}"}

    def predict(self, records: List[Dict], deleted_indexes: List[int] = None,
                data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run prediction with current model on ALL records (including deleted ones for proper index alignment)."""
        if self.current_model is None or self.current_model_info is None:
            return {"error": "No model loaded for prediction"}

        if not records:
            return {"error": "No data loaded for prediction"}

        # Note: deleted_indexes is kept for reference but we predict on ALL data
        # to maintain proper index alignment in the chart
        deleted_indexes = deleted_indexes or []

        try:
            # Get sensors used during training
            selected_sensors = self.current_model_info.get("selected_sensors", [])
            if not selected_sensors:
                return {"error": "Model sensors information not available"}

            # Get outputs used during training
            selected_outputs = self.current_model_info.get("selected_outputs", ["user/angle", "user/throttle"])

            # Prepare input data for ALL records (not filtered)
            X = self._build_input_matrix(records, selected_sensors, data_path)

            # Get normalization params
            norm_params = self.current_model_info.get("normalization_params", {})
            X_mean = np.array(norm_params.get("X_mean", [0] * X.shape[1]))
            X_std = np.array(norm_params.get("X_std", [1] * X.shape[1]))

            # Normalize
            X_normalized = (X - X_mean) / (X_std + 1e-8)

            # Convert to tensor and predict
            X_tensor = torch.FloatTensor(X_normalized)

            self.current_model.eval()
            with torch.no_grad():
                predictions = self.current_model(X_tensor)
                predictions_np = predictions.cpu().numpy()

            # Format results for ALL records with dynamic outputs
            prediction_results = []
            for i, data in enumerate(records):
                result = {
                    "_index": data.get("_index", data.get("_absolute_index", 0)),
                    "_display_index": i,  # Add display index for direct chart mapping
                    "is_deleted": data.get("_absolute_index", data.get("_index", 0)) in deleted_indexes
                }

                # Add predicted and actual values for each output
                for j, output_key in enumerate(selected_outputs):
                    # Create friendly key names (e.g., "user/angle" -> "predicted_angle", "actual_angle")
                    key_name = output_key.split('/')[-1]  # Get the part after '/'
                    result[f"predicted_{key_name}"] = float(predictions_np[i][j])
                    result[f"actual_{key_name}"] = data.get(output_key, 0)

                prediction_results.append(result)

            return {
                "message": "Prediction completed successfully",
                "model_name": self.current_model_info.get("filename", "unknown"),
                "total_predictions": len(prediction_results),
                "selected_sensors": selected_sensors,
                "selected_outputs": selected_outputs,
                "predictions": prediction_results
            }

        except Exception as e:
            return {"error": f"Failed to predict: {str(e)}"}

    def get_current_model_info(self) -> Optional[Dict]:
        """Get current loaded model info."""
        return self.current_model_info
