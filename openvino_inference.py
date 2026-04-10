# openvino_inference.py
# coding:utf-8
"""
OpenVINO推論ラッパー
PyTorchモデルと同じインターフェース（__call__）でOpenVINO推論を実行する。
plannerやrun.pyから透過的に使用可能。

使い方:
  1. convert_to_openvino.py で .pth → .xml に変換
  2. config.py で INFERENCE_ENGINE = "openvino" に設定
  3. MODEL_NAME を変換後の .xml ファイルに設定
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OpenVINOModel:
    """
    OpenVINOのIRモデル(.xml/.bin)を読み込み、
    PyTorchモデルと同じように呼び出せるラッパークラス。

    planner.py 側で model(input_tensor) のように呼ばれることを想定。
    """

    def __init__(self, model_path, device_name="CPU"):
        """
        Args:
            model_path: .xml ファイルのパス（同じディレクトリに .bin も必要）
            device_name: 推論デバイス ("CPU", "GPU", "MYRIAD" など)
        """
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError(
                "OpenVINOがインストールされていません。\n"
                "  pip install openvino\n"
                "でインストールしてください。"
            )

        self.model_path = model_path
        self.device_name = device_name
        self._plan = None  # 後から設定される

        # OpenVINOランタイムの初期化
        core = Core()
        logger.info(f"OpenVINO利用可能デバイス: {core.available_devices}")

        # モデル読み込み
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OpenVINOモデルが見つかりません: {model_path}")

        logger.info(f"OpenVINOモデル読み込み中: {model_path}")
        ov_model = core.read_model(model=model_path)

        # コンパイル（最適化）
        self.compiled_model = core.compile_model(ov_model, device_name)
        self.infer_request = self.compiled_model.create_infer_request()

        # 入出力情報の取得
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = self.input_layer.shape
        self.output_shape = self.output_layer.shape

        logger.info(f"OpenVINOモデルロード完了: device={device_name}, "
                    f"input_shape={list(self.input_shape)}, "
                    f"output_shape={list(self.output_shape)}")

    def __call__(self, input_data):
        """
        PyTorchモデルと同じインターフェースで推論実行。

        Args:
            input_data: numpy.ndarray または torch.Tensor
                        shape は (batch, channels, height, width) など

        Returns:
            OpenVINOInferenceResult: .data でnumpy配列を返すオブジェクト
                                     planner側の互換性のため
        """
        # torch.Tensor → numpy 変換
        if hasattr(input_data, 'numpy'):
            input_np = input_data.detach().cpu().numpy()
        elif hasattr(input_data, 'cpu'):
            input_np = input_data.cpu().numpy()
        else:
            input_np = np.array(input_data, dtype=np.float32)

        # float32に統一
        if input_np.dtype != np.float32:
            input_np = input_np.astype(np.float32)

        # 推論実行
        result = self.infer_request.infer({self.input_layer: input_np})
        output = result[self.output_layer]

        return OpenVINOInferenceResult(output)

    def eval(self):
        """PyTorch互換: 推論モード切替（OpenVINOでは不要だがインターフェース維持）"""
        pass

    def half(self):
        """PyTorch互換: FP16変換（OpenVINOでは変換時に適用済み）"""
        logger.info("OpenVINOモデルはFP16変換をスキップ（変換時に適用済み）")
        return self

    def to(self, device):
        """PyTorch互換: デバイス移動（OpenVINOでは不要）"""
        return self

    def parameters(self):
        """PyTorch互換: パラメータ取得（空リスト）"""
        return []


class OpenVINOInferenceResult:
    """
    OpenVINOの出力をPyTorch Tensorのように扱えるラッパー。
    planner.py で result.data や result[0] のようにアクセスされることを想定。
    """

    def __init__(self, data):
        self.data = data
        self._array = data

    def __getitem__(self, idx):
        return OpenVINOInferenceResult(self._array[idx])

    def __len__(self):
        return len(self._array)

    def item(self):
        """スカラー値を返す"""
        return float(self._array.flat[0])

    def numpy(self):
        return self._array

    def cpu(self):
        return self

    def detach(self):
        return self

    def tolist(self):
        return self._array.tolist()

    @property
    def shape(self):
        return self._array.shape

    def __float__(self):
        return float(self._array.flat[0])

    def __repr__(self):
        return f"OpenVINOInferenceResult(shape={self._array.shape})"


def load_openvino_model(model_path, device_name="CPU"):
    """
    OpenVINOモデルをロードするヘルパー関数。

    Args:
        model_path: .xml ファイルのパス
        device_name: 推論デバイス名

    Returns:
        OpenVINOModel インスタンス
    """
    model = OpenVINOModel(model_path, device_name=device_name)
    return model
