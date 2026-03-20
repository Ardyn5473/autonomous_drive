"""
Neural Network module for data_viewer training functionality.
Provides PyTorch model class and training configuration.

ニューラルネットワークのモデル定義と学習設定を提供するモジュール。
センサー値（超音波・LiDARなど）を入力として、操舵角(angle)とスロットル(throttle)を
予測する回帰モデルを構築する。
"""

import torch
import torch.nn as nn          # nn = Neural Network の略。層やモデルの部品が入っている
from typing import List
from dataclasses import dataclass, field


class NeuralNetwork(nn.Module):
    """
    設定可能なニューラルネットワーク（センサー → 操縦値の回帰モデル）

    ネットワーク構造:
      入力層 → [隠れ層 → ReLU → Dropout] × N → 出力層

    例: input_size=5, hidden_layers=[64, 32], output_size=2 の場合
      [5] → [64] → ReLU → Dropout → [32] → ReLU → Dropout → [2]
      (5つのセンサー値から、angle と throttle の2値を予測)
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int],
                 use_dropout: bool = True, dropout_rate: float = 0.2):
        """
        Args:
            input_size:    入力の次元数（= 選択したセンサーの合計要素数）
            output_size:   出力の次元数（通常 2: angle, throttle）
            hidden_layers: 各隠れ層のニューロン数のリスト (例: [64, 32])
            use_dropout:   Dropout を使うか（過学習防止のためランダムにニューロンを無効化）
            dropout_rate:  Dropout の割合（0.2 = 20%のニューロンを無効化）
        """
        # nn.Module の初期化（PyTorch のお約束）
        super(NeuralNetwork, self).__init__()

        # --- 層を順番に組み立てる ---
        layers = []
        prev_size = input_size  # 前の層の出力サイズ（最初は入力サイズ）

        for hidden_size in hidden_layers:
            # Linear: 全結合層（prev_size個の入力 → hidden_size個の出力）
            # 各ニューロンが「重み × 入力 + バイアス」の計算を行う
            layers.append(nn.Linear(prev_size, hidden_size))

            # ReLU: 活性化関数（負の値を0にする）
            # これがないと、どれだけ層を重ねても線形な計算にしかならない
            layers.append(nn.ReLU())

            if use_dropout:
                # Dropout: 学習時にランダムにニューロンを無効化する
                # 特定のニューロンに依存しすぎる（過学習）のを防ぐ正則化手法
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size  # 次の層の入力サイズを更新

        # 最後の出力層（隠れ層の最後 → 出力サイズ）
        # 回帰問題なので活性化関数は付けない（生の数値をそのまま出力）
        layers.append(nn.Linear(prev_size, output_size))

        # Sequential: 層を順番に実行するコンテナ
        # forward() で入力を渡すと、layers[0] → layers[1] → ... と順に処理される
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        順伝播（フォワードパス）: 入力 x をネットワークに通して出力を返す。

        Args:
            x: 入力テンソル [バッチサイズ, 入力次元数]
        Returns:
            出力テンソル [バッチサイズ, 出力次元数]

        ※ PyTorch が自動的に backward()（逆伝播）を計算してくれるので、
          forward() だけ定義すれば学習が行える。
        """
        return self.model(x)


@dataclass
class TrainingConfig:
    """
    学習の設定パラメータをまとめたクラス。

    @dataclass を使うと、__init__ や __repr__ が自動生成される。
    UI のフォームから送られた JSON を from_dict() でこのクラスに変換する。
    """

    # --- ネットワーク構造 ---
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    #   隠れ層のニューロン数。[64, 32] = 第1隠れ層64個、第2隠れ層32個

    # --- 学習ハイパーパラメータ ---
    epochs: int = 100
    #   エポック数 = データ全体を何回繰り返し学習するか

    batch_size: int = 32
    #   バッチサイズ = 1回のパラメータ更新で使うデータ数
    #   大きい → 学習が安定するが遅い、小さい → ノイズが多いが速い

    learning_rate: float = 0.001
    #   学習率 = パラメータを1回の更新でどれくらい動かすか
    #   大きすぎると発散、小さすぎると収束が遅い

    # --- データフィルタリング ---
    deleted_indexes: List[int] = field(default_factory=list)
    #   削除済みデータのインデックス（学習から除外する）

    selected_sensors: List[str] = field(default_factory=lambda: [
        'ultrasonic/RrLH',     # 左後方の超音波センサー
        'ultrasonic/FrLH',     # 左前方の超音波センサー
        'ultrasonic/FrFR',     # 正面の超音波センサー
        'ultrasonic/FrRH',     # 右前方の超音波センサー
        'ultrasonic/RrRH'      # 右後方の超音波センサー
    ])
    #   入力に使うセンサーキーの一覧

    selected_outputs: List[str] = field(default_factory=lambda: [
        'user/angle',          # ステアリング角度 (-1.0 ~ 1.0)
        'user/throttle'        # スロットル量     (-1.0 ~ 1.0)
    ])
    #   出力（予測対象）のキー一覧

    # --- 正則化 ---
    use_dropout: bool = True
    dropout_rate: float = 0.2

    # --- データ範囲指定（None = 全データ使用） ---
    data_range_start: int = None  # 開始インデックス（この値を含む）
    data_range_end: int = None    # 終了インデックス（この値を含む）

    # --- ダウンサンプリング: N件ごとに1件使う（1 = 全件使用） ---
    downsample_rate: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """
        辞書（= UIから送られたJSON）を TrainingConfig に変換する。
        存在しないキーにはデフォルト値を使用する。
        """
        return cls(
            hidden_layers=data.get('hidden_layers', [64, 32]),
            epochs=data.get('epochs', 100),
            batch_size=data.get('batch_size', 32),
            learning_rate=data.get('learning_rate', 0.001),
            deleted_indexes=data.get('deleted_indexes', []),
            selected_sensors=data.get('selected_sensors', [
                'ultrasonic/RrLH',
                'ultrasonic/FrLH',
                'ultrasonic/FrFR',
                'ultrasonic/FrRH',
                'ultrasonic/RrRH'
            ]),
            selected_outputs=data.get('selected_outputs', [
                'user/angle',
                'user/throttle'
            ]),
            use_dropout=data.get('use_dropout', True),
            dropout_rate=data.get('dropout_rate', 0.2),
            data_range_start=data.get('data_range_start', None),
            data_range_end=data.get('data_range_end', None),
            downsample_rate=data.get('downsample_rate', 1)
        )
