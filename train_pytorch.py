#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch学習スクリプト（config.py設定による自動実行）

config.pyのPLANとSAVE_FORMATに基づいて自動的に学習を実行
- 画像モデル: donkeycar, resnet18, mobilevit_xxs, edgenext_xxsmall
- 超音波センサーモデル: nn
- データ形式: csv, donkeycar形式に対応
"""

import os
import sys
import time
import datetime
import pandas as pd
import json
import struct

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.optim as optim

# config.py から各種設定を読み込み
import config

# annotation_training_d2j のモジュールをインポート
try:
    import sys
    import os
    
    # サブモジュールのパスをsys.pathに追加
    submodule_path = os.path.join(os.path.dirname(__file__), 'annotation_training_d2j')
    if submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)
    
    import model_catalog
    import model_info
    import model_training
    print("Successfully imported model modules from annotation_training_d2j")
    MODEL_CATALOG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")
    print("Some advanced model features may not be available.")
    MODEL_CATALOG_AVAILABLE = False

########################################
# DonkeyCar形式データ読み込み関数
########################################
def load_donkeycar_data(data_dir, auto_select=False, plan="donkeycar"):
    """
    DonkeyCar形式のデータを読み込む
    Args:
        data_dir: データディレクトリのパス (例: 'data/data_20250830_125646')
        auto_select: Trueの場合、最初の画像カラムを自動選択
        plan: 使用するモデルプラン（"donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall", "nn"）
    Returns:
        images_or_None: 画像ファイルパスのリスト（画像モデルの場合）またはNone（nnの場合）
        angles: ステアリング角度のリスト
        throttles: スロットル値のリスト
        ultrasonic_data: 超音波センサーデータの辞書（nnプランの場合）またはNone（画像モデルの場合）
    """
    # manifest.jsonを読み込み
    manifest_path = os.path.join(data_dir, 'manifest.json')
    try:
        with open(manifest_path, 'r') as f:
            manifest = [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        print(f"データ確認エラー: manifest.jsonが見つかりません: {manifest_path}")
        sys.exit(1)
    except Exception as e:
        print(f"データ確認エラー: manifest.json読み込み中にエラーが発生しました: {e}")
        sys.exit(1)
    
    # カラム情報を取得
    column_names = manifest[0]  # 1行目がカラム名
    column_types = manifest[1]  # 2行目がデータ型
    
    # プランに応じて必要なカラムを特定
    selected_image_column = None
    image_based_plans = ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]
    
    if plan in image_based_plans:
        # 画像ベースモデルの場合: 画像カラムを探す
        image_columns = []
        for i, col_name in enumerate(column_names):
            if 'image_array' in col_name:
                image_columns.append(col_name)
        
        if not image_columns:
            print("警告: 画像カラムが見つかりません。超音波センサーデータを使用します。")
            plan = "nn"  # nnプランに切り替え
        else:
            # 複数の画像カラムがある場合の処理
            if len(image_columns) > 1:
                print("\n複数の画像カラムが見つかりました:")
                for i, col in enumerate(image_columns):
                    print(f"{i+1}. {col}")
                
                if auto_select:
                    # 自動選択モード: config.MODEL_INPUT_IMAGEで指定されたカラムを優先
                    if hasattr(config, 'MODEL_INPUT_IMAGE') and config.MODEL_INPUT_IMAGE:
                        # config.MODEL_INPUT_IMAGEに一致するカラムを探す
                        matching_column = None
                        for col in image_columns:
                            if config.MODEL_INPUT_IMAGE in col or col == config.MODEL_INPUT_IMAGE:
                                matching_column = col
                                break
                        
                        if matching_column:
                            selected_image_column = matching_column
                            print(f"\nconfig.MODEL_INPUT_IMAGEに基づいて自動選択: {selected_image_column}")
                        else:
                            # 一致するカラムが見つからない場合は最初のカラムを選択
                            selected_image_column = image_columns[0]
                            print(f"\n警告: config.MODEL_INPUT_IMAGE '{config.MODEL_INPUT_IMAGE}' に一致するカラムが見つかりません。")
                            print(f"デフォルトで最初のカラムを選択: {selected_image_column}")
                    else:
                        # MODEL_INPUT_IMAGEが設定されていない場合は最初のカラムを選択
                        selected_image_column = image_columns[0]
                        print(f"\n自動選択: {selected_image_column}")
                else:
                    # 対話モード: ユーザーに選択させる
                    while True:
                        choice = input(f"\n使用する画像カラムを選択してください (1-{len(image_columns)}, デフォルト: 1): ").strip()
                        if choice == "":
                            selected_image_column = image_columns[0]
                            break
                        elif choice.isdigit() and 1 <= int(choice) <= len(image_columns):
                            selected_image_column = image_columns[int(choice) - 1]
                            break
                        else:
                            print("無効な選択です。もう一度選択してください。")
            else:
                selected_image_column = image_columns[0]
            
            print(f"選択された画像カラム: {selected_image_column}")
    
    if plan == "nn":
        # 超音波センサーモデルの場合
        print("超音波センサーデータを読み込みます")
    
    # カタログファイルを特定
    catalog_info = manifest[4]  # 5行目がカタログ情報
    catalog_paths = catalog_info['paths']
    current_index = catalog_info['current_index']
    deleted_indexes = catalog_info.get('deleted_indexes', [])
    
    images = []
    angles = []
    throttles = []
    ultrasonic_data = {sensor: [] for sensor in config.ULTRASONIC_SENSOR_LIST} if plan == "nn" else None
    
    # 各カタログファイルを読み込み
    for catalog_file in catalog_paths:
        catalog_path = os.path.join(data_dir, catalog_file)
        
        # カタログファイルを行ごとに読み込み（JSON Lines形式）
        with open(catalog_path, 'r') as f:
            record_idx = 0
            for line in f:
                # 削除されたインデックスはスキップ
                if record_idx in deleted_indexes:
                    record_idx += 1
                    continue
                
                # current_indexまで読んだら終了
                if record_idx >= current_index:
                    break
                
                try:
                    # JSONとしてパース
                    record = json.loads(line.strip())
                    
                    # ステアリングとスロットルは常に取得
                    angle = record.get('user/angle', 0.0)
                    throttle = record.get('user/throttle', 0.0)
                    
                    # プランに応じてデータを取得
                    if plan in image_based_plans and selected_image_column:
                        # 画像ベースモデルの場合
                        if selected_image_column in record:
                            img_filename = record[selected_image_column]
                            img_path = os.path.join(data_dir, 'images', img_filename)
                            
                            images.append(img_path)
                            angles.append(angle)
                            throttles.append(throttle)
                    
                    elif plan == "nn":
                        # 超音波センサー/LiDARモデルの場合
                        angles.append(angle)
                        throttles.append(throttle)

                        for sensor in config.ULTRASONIC_SENSOR_LIST:
                            # ultrasonic/* または lidar/* キーを探す
                            sensor_key_ultrasonic = f"ultrasonic/{sensor}"
                            sensor_key_lidar = f"lidar/{sensor}"

                            # ultrasonic/* を優先、なければ lidar/* を使用
                            if sensor_key_ultrasonic in record:
                                value = record.get(sensor_key_ultrasonic, 0)
                            elif sensor_key_lidar in record:
                                value = record.get(sensor_key_lidar, 0)
                            else:
                                value = 0

                            ultrasonic_data[sensor].append(value)
                    
                except json.JSONDecodeError as e:
                    print(f"JSONデコードエラー: record {record_idx}, エラー: {e}")
                
                record_idx += 1
    
    if plan == "nn":
        print(f"DonkeyCar形式データ読み込み完了: {len(angles)}レコード（超音波センサー）")
        return None, angles, throttles, ultrasonic_data
    else:
        print(f"DonkeyCar形式データ読み込み完了: {len(images)}レコード（画像）")
        return images, angles, throttles, None

########################################
# Model Catalog統合関数
########################################
def get_model_from_catalog(plan_name, input_dim=None, for_training=False):
    """
    model_catalogからモデルを取得する（推論エンジンに応じてモデルを選択）
    
    Args:
        plan_name: PLANの名前（例: 'donkeycar', 'resnet18', 'mobilevit_xxs', 'edgenext_xxsmall'など）
        input_dim: 入力次元（センサーデータ用）
        for_training: Trueの場合、学習用にPyTorchモデルを強制的に使用
    
    Returns:
        PyTorchモデル（または最適化されたモデル）
    """
    if not MODEL_CATALOG_AVAILABLE:
        raise ImportError("model_catalog is not available")
    
    # plan_nameに対応するモデルクラス名を探す
    model_class_map = {
        'donkeycar': 'DonkeyModel',
        'resnet18': 'ResNet18Model',
        'mobilevit_xxs': 'MobileViTXXSModel',
        'edgenext_xxsmall': 'EdgeNextXXSmallModel'
    }
    
    if plan_name in model_class_map:
        model_class_name = model_class_map[plan_name]
        
        # 学習時は常にPyTorchモデルを使用
        if for_training:
            model_path = config.MODEL_PATH
            print(f"Training mode: Using PyTorch model path: {model_path}")
        else:
            # 推論エンジンに応じてモデルパスを動的生成
            if config.INFERENCE_ENGINE == "tensorrt":
                # TensorRTパスを動的生成（.trtまたは元のファイル）
                model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME.replace('.pth', '.trt'))
                print(f"Using TensorRT model path: {model_path}")
            elif config.INFERENCE_ENGINE == "openvino":
                # OpenVINOパスを動的生成（.xml）
                model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME.replace('.pth', '.xml'))
                print(f"Using OpenVINO model path: {model_path}")
            else:
                model_path = config.MODEL_PATH
                print(f"Using PyTorch model path: {model_path}")
        
        # model_catalogから直接モデルクラスを取得
        if hasattr(model_catalog, model_class_name):
            model_class = getattr(model_catalog, model_class_name)
            
            # 学習時は常にPyTorchモデルを使用
            if for_training:
                model = model_class()
            else:
                # 推論エンジン固有のモデル初期化
                if config.INFERENCE_ENGINE == "tensorrt":
                    model = _create_tensorrt_model(model_class, model_path)
                elif config.INFERENCE_ENGINE == "openvino":
                    model = _create_openvino_model(model_class, model_path)
                else:
                    model = model_class()
                
            if for_training:
                print(f"Loaded {model_class_name} from model_catalog (training mode: PyTorch)")
            else:
                print(f"Loaded {model_class_name} from model_catalog (engine: {config.INFERENCE_ENGINE})")
            
            # モデル情報を表示
            if hasattr(model, 'get_info'):
                info = model.get_info()
                print(f"Model info: {info}")
            
            return model
        else:
            print(f"Model class {model_class_name} not found in model_catalog")
    
    # カタログにない場合は従来のモデルを使用
    print(f"Model {plan_name} not found in catalog, falling back to default")
    return None

def _create_tensorrt_model(model_class, model_path):
    """TensorRT最適化モデルを作成"""
    try:
        # 実際のファイルパスを確認（拡張子なしの場合も考慮）
        actual_model_path = model_path
        if not os.path.exists(model_path):
            # .trt拡張子を削除してみる
            if model_path.endswith('.trt'):
                actual_model_path = model_path[:-4]
                if not os.path.exists(actual_model_path):
                    # 元のMODEL_NAMEパスも試す
                    actual_model_path = config.MODEL_PATH
        
        # TensorRTエンジンファイルが存在するかチェック
        if os.path.exists(actual_model_path):
            print(f"Loading TensorRT model from {actual_model_path}")
            
            # TensorRTエンジンファイル（.trt, .engine）の場合
            if actual_model_path.endswith(('.trt', '.engine')):
                return TensorRTModel(actual_model_path)
            
            # torch2trt形式のTensorRTモデル（_trt.pthファイル）の場合
            elif '_trt.pth' in actual_model_path:
                try:
                    from torch2trt import TRTModule
                    trt_module = TRTModule()
                    trt_module.load_state_dict(torch.load(actual_model_path, map_location='cpu'))
                    print(f"Loaded torch2trt TensorRT model from {actual_model_path}")
                    return trt_module
                except ImportError:
                    print("torch2trt not available, loading as PyTorch model")
                    # PyTorchモデルとして読み込み処理に続く
            
            # PyTorchモデル（拡張子なしも含む）の場合は通常のPyTorchモデルとして読み込み
            else:
                try:
                    # PyTorchモデルをロード
                    pytorch_model = model_class()
                    # チェックポイントから状態をロード
                    checkpoint = torch.load(actual_model_path, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        pytorch_model.load_state_dict(checkpoint)
                    pytorch_model.eval()
                    
                    # torch2trtが利用可能な場合はTensorRTに変換を試みる
                    try:
                        import torch2trt
                    except ImportError:
                        print("torch2trt not available, using PyTorch model directly")
                        return pytorch_model
                    
                    # TensorRTに変換
                    dummy_input = torch.randn(1, 3, 224, 224)
                    if torch.cuda.is_available():
                        pytorch_model = pytorch_model.cuda()
                        dummy_input = dummy_input.cuda()
                    
                    trt_model = torch2trt.torch2trt(pytorch_model, [dummy_input])
                    print("PyTorch model converted to TensorRT successfully")
                    return trt_model
                except Exception as e:
                    print(f"TensorRT conversion failed: {e}")
                    print("Falling back to PyTorch model")
                    return pytorch_model
        else:
            print(f"Model not found at {model_path}, using PyTorch model")
            return model_class()
    except Exception as e:
        print(f"Error loading TensorRT model: {e}")
        print("Falling back to PyTorch model")
        try:
            model = model_class()
            # 実際のモデルファイルが存在する場合は読み込み
            if os.path.exists(config.MODEL_PATH):
                checkpoint = torch.load(config.MODEL_PATH, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            return model
        except Exception as e2:
            print(f"Failed to load PyTorch model: {e2}")
            return model_class()

class TensorRTModel:
    """TensorRT推論用のラッパークラス"""
    
    def __init__(self, engine_path):
        """
        TensorRTエンジンを初期化
        
        Args:
            engine_path (str): TensorRTエンジンファイルのパス
        """
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        self._load_engine()
        self._allocate_buffers()
        print(f"TensorRT model initialized: {engine_path}")
    
    def _load_engine(self):
        """TensorRTエンジンをロード"""
        import tensorrt as trt
        
        # ファイルがTorchScript（.trt）かネイティブTensorRTエンジン（.engine）かをチェック
        try:
            # まずTorchScriptファイルかどうか確認
            with open(self.engine_path, 'rb') as f:
                # TorchScriptファイルはZIPアーカイブで、先頭に'PK'がある
                magic = f.read(2)
                f.seek(0)
                
                if magic == b'PK':
                    # TorchScriptファイルの場合
                    print(f"Loading TorchScript TensorRT model: {self.engine_path}")
                    self.torch_model = torch.jit.load(self.engine_path, map_location='cpu')
                    if torch.cuda.is_available():
                        self.torch_model = self.torch_model.cuda()
                    self.torch_model.eval()
                    self.is_torchscript = True
                    return
                else:
                    # ネイティブTensorRTエンジンファイルの場合
                    engine_data = f.read()
            
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
            self.context = self.engine.create_execution_context()
            self.is_torchscript = False
            
        except Exception as e:
            print(f"Failed to load as TensorRT engine: {e}")
            print("Falling back to TorchScript loading...")
            # フォールバックとしてTorchScriptとして読み込み
            try:
                self.torch_model = torch.jit.load(self.engine_path, map_location='cpu')
                if torch.cuda.is_available():
                    self.torch_model = self.torch_model.cuda()
                self.torch_model.eval()
                self.is_torchscript = True
                print(f"Successfully loaded as TorchScript model: {self.engine_path}")
            except Exception as torch_error:
                raise RuntimeError(f"Failed to load both as TensorRT engine and TorchScript: {e}, {torch_error}")
    
    def _allocate_buffers(self):
        """GPU/CPUバッファを割り当て"""
        # TorchScriptモデルの場合はバッファ割り当ては不要
        if hasattr(self, 'is_torchscript') and self.is_torchscript:
            return
            
        import tensorrt as trt
        import pycuda.driver as cuda
        import numpy as np
        
        for binding in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = self.context.get_tensor_shape(binding)
            size = trt.volume(shape)
            
            # CPUとGPUメモリを割り当て
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def __call__(self, x):
        """
        PyTorchモデルと同じインターフェースで推論実行
        
        Args:
            x (torch.Tensor): 入力テンソル
            
        Returns:
            torch.Tensor: 推論結果テンソル
        """
        # TorchScriptモデルの場合は直接推論
        if hasattr(self, 'is_torchscript') and self.is_torchscript:
            return self.torch_model(x)
        
        # ネイティブTensorRTエンジンの場合
        import numpy as np
        import pycuda.driver as cuda
        
        # PyTorchテンソルからNumPy配列に変換
        if isinstance(x, torch.Tensor):
            input_np = x.cpu().numpy()
        else:
            input_np = np.array(x)
        
        # 入力データをGPUメモリにコピー
        np.copyto(self.inputs[0]['host'], input_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 推論実行
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 結果をCPUメモリにコピー
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 結果をTorchテンソルに変換して返す
        output_np = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return torch.from_numpy(output_np.copy())
    
    def eval(self):
        """PyTorchモデルとの互換性のため"""
        return self
    
    def to(self, device):
        """PyTorchモデルとの互換性のため"""
        # TorchScriptモデルの場合はデバイス移動を適用
        if hasattr(self, 'is_torchscript') and self.is_torchscript:
            self.torch_model = self.torch_model.to(device)
        # ネイティブTensorRTエンジンは常にGPU
        return self
    
    def cuda(self):
        """PyTorchモデルとの互換性のため"""
        # TorchScriptモデルの場合はGPUに移動
        if hasattr(self, 'is_torchscript') and self.is_torchscript:
            self.torch_model = self.torch_model.cuda()
        # ネイティブTensorRTエンジンは常にGPU
        return self

def _create_openvino_model(model_class, model_path):
    """OpenVINO最適化モデルを作成"""
    try:
        # 実際のファイルパスを確認（拡張子なしの場合も考慮）
        actual_model_path = model_path
        if not os.path.exists(model_path):
            # .xml拡張子を削除してみる
            if model_path.endswith('.xml'):
                actual_model_path = model_path[:-4]
                if not os.path.exists(actual_model_path):
                    # 元のMODEL_NAMEパスも試す
                    actual_model_path = config.MODEL_PATH
        
        # OpenVINOモデルファイルが存在するかチェック
        if os.path.exists(actual_model_path):
            print(f"Loading OpenVINO model from {actual_model_path}")
            
            # OpenVINO固有のファイル（.xml/.bin）の場合
            if actual_model_path.endswith('.xml'):
                try:
                    from openvino.runtime import Core
                    # OpenVINOラッパーを作成
                    model = model_class()
                    # OpenVINO固有の初期化があれば実行
                    if hasattr(model, 'load_openvino_model'):
                        model.load_openvino_model(actual_model_path)
                    print("OpenVINO model loaded successfully")
                    return model
                except ImportError:
                    print("OpenVINO not available, loading as PyTorch model")
                    return model_class()
            
            # PyTorchモデル（拡張子なしも含む）の場合は通常のPyTorchモデルとして読み込み
            else:
                try:
                    pytorch_model = model_class()
                    # チェックポイントから状態をロード
                    checkpoint = torch.load(actual_model_path, map_location='cpu', weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        pytorch_model.load_state_dict(checkpoint)
                    pytorch_model.eval()
                    
                    # OpenVINOに変換を試みる（将来的な拡張用）
                    try:
                        from openvino.runtime import Core
                        # OpenVINO変換コードをここに追加可能
                        print("OpenVINO conversion not implemented, using PyTorch model")
                        return pytorch_model
                    except ImportError:
                        print("OpenVINO not available, using PyTorch model directly")
                        return pytorch_model
                        
                except Exception as e:
                    print(f"Error loading PyTorch model for OpenVINO: {e}")
                    return model_class()
        else:
            print(f"Model not found at {model_path}, using PyTorch model")
            return model_class()
    except Exception as e:
        print(f"Error loading OpenVINO model: {e}")
        print("Falling back to PyTorch model")
        try:
            model = model_class()
            # 実際のモデルファイルが存在する場合は読み込み
            if os.path.exists(config.MODEL_PATH):
                checkpoint = torch.load(config.MODEL_PATH, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            return model
        except Exception as e2:
            print(f"Failed to load PyTorch model: {e2}")
            return model_class()

########################################
# CSVファイル関連関数
########################################
def find_csv_files(folder: str) -> list[str]:
    """指定フォルダ内の全てのCSVファイルをソートしてリストで返す"""
    csv_files = [file for file in os.listdir(folder) if file.endswith(".csv")]
    csv_files.sort()
    return csv_files

def combine_csv_files(csv_files: list[str], folder: str) -> pd.DataFrame:
    """複数のCSVファイルを結合する。列構造が違う場合はsys.exit()で中断。"""
    dataframes = []
    columns_list = []
    for csv_file in csv_files:
        csv_path = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_path)
        dataframes.append(df)
        columns_list.append(df.columns)

        # 列構造が異なる場合は結合不可とする
        if len(columns_list) > 1:
            if not all(columns_list[0] == columns_list[-1]):
                print(f"{csv_path} の列構造が他ファイルと異なるため結合できません。")
                sys.exit()
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def choose_csv_file(csv_files: list[str], folder: str) -> tuple[pd.DataFrame, str]:
    """
    CSVファイルのリストからユーザーに選ばせてDataFrameを返す。
    何も入力しない場合は最新(csv_files[-1])を選択。
    """
    csv_file = input("ファイル名を入力してください (何も入力しないと最新を選択): ").strip()
    if csv_file == "":
        csv_file = csv_files[-1]
        print(f"\n最新のファイルを選択: {csv_file}")
        time.sleep(0.5)

    csv_path = os.path.join(folder, csv_file)
    df = pd.read_csv(csv_path)
    return df, csv_file


########################################
# データロード/前処理
########################################
def load_data(folder=None, data_format="csv", plan="donkeycar"):
    """
    1. CSVファイルまたはDonkeyCar形式データを読み込み
    2. DataFrame化
    3. x_tensor, y_tensor, csv_file, ts を返す
    """
    if folder is None:
        folder = config.RECORDS_DIRECTORY
    
    if data_format == "donkeycar":
        # DonkeyCar形式のデータを読み込み（プランに応じて処理）
        if plan == "nn":
            # 超音波センサーNNの場合
            _, angles, throttles, ultrasonic_data = load_donkeycar_data(folder, plan=plan)
            
            # DataFrameに変換
            df = pd.DataFrame({
                'steering': angles,
                'throttle': throttles
            })
            
            # 超音波センサーデータを追加
            for sensor in config.ULTRASONIC_SENSOR_LIST:
                df[sensor] = ultrasonic_data[sensor]
        else:
            # 画像ベースモデルの場合
            images, angles, throttles, _ = load_donkeycar_data(folder, plan=plan)
            
            # DataFrameに変換
            df = pd.DataFrame({
                'image_file': images,
                'steering': angles,
                'throttle': throttles
            })
            
            # 超音波センサーデータは0で初期化（画像モデルでは使用しない）
            for sensor in config.ULTRASONIC_SENSOR_LIST:
                df[sensor] = 0
        
        csv_file = os.path.basename(folder)
        print("\n入力データのサイズ:", df.shape)
        
        x_tensor, y_tensor = preprocess_data(df)
        ts = None
        
        return x_tensor, y_tensor, csv_file, ts
    
    # 従来のCSV読み込み処理
    csv_files = find_csv_files(folder)
    if not csv_files:
        print("CSVファイルが見つかりません。")
        sys.exit()

    print("検出されたCSVファイル一覧:", csv_files)

    if len(csv_files) > 1:
        answer = input("複数のCSVファイルがあります。ファイルを結合しますか？ (y/n): ").strip().lower()
        if answer == "y":
            df = combine_csv_files(csv_files, folder)
            csv_file = "merged.csv"
        else:
            df, csv_file = choose_csv_file(csv_files, folder)
    else:
        csv_file = csv_files[0]
        csv_path = os.path.join(folder, csv_file)
        df = pd.read_csv(csv_path)

    print("\n入力データの先頭3行:\n", df.head(3), "\nデータサイズ:", df.shape)

    x_tensor, y_tensor = preprocess_data(df)
    ts = df['timestamp'] if 'timestamp' in df.columns else None

    return x_tensor, y_tensor, csv_file, ts


def preprocess_data(df: pd.DataFrame):
    """
    CSV例: 
      timestamp, mode, steering, throttle, FrLH, Fr, FrRH, image_file
    入力データ x → config.ULTRASONIC_SENSOR_LIST (例: ["FrLH","Fr","FrRH"])
    出力データ y → ["steering","throttle"]
    """
    # X: 超音波センサ列
    x = df[config.ULTRASONIC_SENSOR_LIST].copy()
    x_tensor = torch.tensor(x.values, dtype=torch.float32)
    x_tensor = normalize_ultrasonics(x_tensor)

    # Y: 操舵 & スロットル
    y = df[["steering", "throttle"]].copy()
    if config.MODEL_TYPE == "categorical":
        # カテゴリ分類用の変換があれば実行
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    else:
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        # 利用しない、後方互換のため残す
        # y_tensor = normalize_motor(y_tensor)
        # y_tensor[:, 0] = steering_shifter_to_01(y_tensor[:, 0])

    return x_tensor, y_tensor


########################################
# 正規化 / デノーマライズ
########################################
def normalize_ultrasonics(x_tensor, scale=None):
    if scale is None:
        scale = config.NORMALIZE_RANGE  # 2000 mm → 1.0 など
    return x_tensor / scale

# 利用しない後方互換のため残す
def normalize_motor(y_tensor, scale=100):
    return y_tensor / scale

# 利用しない後方互換のため残す
def steering_shifter_to_01(y_tensor):
    return (y_tensor + 1) / 2

# 利用しない後方互換のため残す
def steering_shifter_to_m11(y_tensor):
    return (y_tensor - 0.5) * 2


########################################
# データセットクラス
########################################
class CustomDataset(torch.utils.data.Dataset):
    """数値入力 (NN用)"""
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CustomImageDataset(torch.utils.data.Dataset):
    """画像入力 (CNN用)"""
    def __init__(self, image_files, y_tensor, transform=None):
        self.image_files = image_files
        self.y_tensor = y_tensor
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # 画像が存在しない場合などの対策が必要ならtry-except
        from PIL import Image  # import は先頭でもOK
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.y_tensor[idx]
        return image, labels


########################################
# モデル定義
########################################
class NeuralNetwork(nn.Module):
    """全結合NN (数値入力)"""
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        if config.MODEL_TYPE == "categorical":
            # 分類ならカテゴリー数だけ出力
            layers.append(nn.Linear(hidden_dim, 3))  # 例: 3カテゴリ
        else:
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if config.MODEL_TYPE == "categorical":
            x = F.log_softmax(x, dim=1)
        return x

    def predict(self, model, x_tensor):
        model.eval()
        # デバイスを自動検出
        device = next(model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            pred = model(x_tensor)
            if config.MODEL_TYPE == "categorical":
                pred = torch.argmax(pred, dim=1)
            else:
                pass
                # 利用しない、後方互換のため残す
                # 0~1→-1~1
                # pred[:, 0] = steering_shifter_to_m11(pred[:, 0])
        # clamp
        pred = torch.clamp(pred, -1, 1)
        return pred


class ConvolutionalNeuralNetwork(nn.Module):
    """CNN (画像入力)"""
    def __init__(self, input_dim=(160, 120, 3), output_dim=2):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # Conv層
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        self.layer1 = nn.Sequential(
            self.conv1, self.relu, self.dropout,
            self.conv2, self.relu, self.dropout,
            self.conv3, self.relu, self.dropout,
            self.conv4, self.relu, self.dropout,
            self.conv5, self.relu, self.dropout,
        )

        n_size = self._get_conv_output(input_dim)
        self.fc1 = nn.Linear(n_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        self.layer2 = nn.Sequential(
            self.fc1, self.relu, self.dropout,
            self.fc2, self.relu, self.dropout,
            self.fc3
        )

    def _get_conv_output(self, shape):
        # shape: (W, H, Depth)
        dummy_input = torch.rand(1, 3, shape[1], shape[0])  # (N, C, H, W)
        output_feat = self.layer1(dummy_input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        return x
            
    def predict(self, model, x):
        """
        モデルの予測を行う。
        :param model: PyTorch モデル
        :param x: 入力データ (torch.Tensor)
        :return: モデルの出力
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, but got {type(x)}")

        model.eval()  # モデルを推論モードに切り替え
        # デバイスを自動検出
        device = next(model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            pred = model(x)  # モデルに入力
        pred = torch.clamp(pred, -1, 1) #正規化
        return pred


########################################
# トレーニング / 保存 / ロード
########################################
def train_model(model, dataloader, criterion, optimizer, model_name, start_epoch=0, epochs=None, device=None, val_dataloader=None):
    if epochs is None:
        epochs = config.EPOCHS
    
    # デバイスの設定
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルをデバイスに移動
    model = model.to(device)
    model.train()
    
    loss_history = []
    steering_loss_history = []
    throttle_loss_history = []
    val_loss_history = []  # 検証用損失履歴
    
    # Early Stopping用変数
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopped = False
    stopped_epoch = 0
    best_model_state = None
    
    import time
    training_start_time = time.time()
    
    print(f"学習開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用デバイス: {device}")
    if device.type == 'cuda':
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"利用可能メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-" * 80)

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start_time = time.time()
        
        # トレーニング
        model.train()
        epoch_loss = 0.0
        epoch_steering_loss = 0.0
        epoch_throttle_loss = 0.0
        batch_count = 0
        
        for inputs, targets in dataloader:
            # データをデバイスに移動
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if config.MODEL_TYPE == "categorical":
                targets = targets[:, 0]  # steeringだけ分類のケースなどは適宜調整
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
            else:
                # 回帰の場合、steering と throttle の個別 loss を計算
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
                
                # 個別 loss 計算（steering: index 0, throttle: index 1）
                if outputs.shape[1] >= 2 and targets.shape[1] >= 2:
                    steering_loss = criterion(outputs[:, 0], targets[:, 0])
                    throttle_loss = criterion(outputs[:, 1], targets[:, 1])
                    epoch_steering_loss += steering_loss.item()
                    epoch_throttle_loss += throttle_loss.item()
            
            loss.backward()
            optimizer.step()
            batch_count += 1

        # エポック平均の計算
        avg_loss = epoch_loss / batch_count
        avg_steering_loss = epoch_steering_loss / batch_count if batch_count > 0 else 0.0
        avg_throttle_loss = epoch_throttle_loss / batch_count if batch_count > 0 else 0.0
        
        loss_history.append(avg_loss)
        steering_loss_history.append(avg_steering_loss)
        throttle_loss_history.append(avg_throttle_loss)
        
        # 検証
        val_loss = 0.0
        if val_dataloader is not None:
            model.eval()
            val_batch_count = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    val_outputs = model(val_inputs)
                    if config.MODEL_TYPE == "categorical":
                        val_targets = val_targets[:, 0]
                    val_batch_loss = criterion(val_outputs, val_targets)
                    val_loss += val_batch_loss.item()
                    val_batch_count += 1
            
            val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
            val_loss_history.append(val_loss)
            
            # Early Stoppingチェック
            if config.USE_EARLY_STOPPING:
                if val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    early_stopping_counter += 1
                    
                if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly Stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    early_stopped = True
                    stopped_epoch = epoch + 1
                    break
        
        # 経過時間計算
        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        
        # 進捗表示
        if config.MODEL_TYPE == "categorical":
            progress_msg = f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}"
            if val_dataloader is not None:
                progress_msg += f" | Val: {val_loss:.6f}"
            progress_msg += f" | 時間: {epoch_duration:.1f}s | 総経過: {total_elapsed:.1f}s"
            print(progress_msg)
        else:
            progress_msg = (f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Total: {avg_loss:.6f} | "
                          f"Steering: {avg_steering_loss:.6f} | "
                          f"Throttle: {avg_throttle_loss:.6f}")
            if val_dataloader is not None:
                progress_msg += f" | Val: {val_loss:.6f}"
            progress_msg += f" | 時間: {epoch_duration:.1f}s | 総経過: {total_elapsed:.1f}s"
            print(progress_msg)
            
        if config.USE_EARLY_STOPPING and val_dataloader is not None and early_stopping_counter > 0:
            print(f"Early stopping counter: {early_stopping_counter}/{config.EARLY_STOPPING_PATIENCE}")
    
    # Early Stoppingで停止した場合、最良のモデルを復元
    if early_stopped and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("最良の検証損失を持つモデル状態を復元しました")

    total_training_time = time.time() - training_start_time
    print("-" * 80)
    print(f"トレーニング完了！総学習時間: {total_training_time:.1f}秒")
    print(f"完了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Loss可視化
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    if config.MODEL_TYPE == "categorical":
        # カテゴリカル分類の場合
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(loss_history, label='Train Loss', linewidth=2)
        if val_loss_history:
            ax.plot(val_loss_history, label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        loss_path = os.path.join(config.MODEL_DIR, f"{model_name}_loss.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # 回帰の場合は個別Lossも表示
        num_plots = 2 if val_loss_history else 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        # 総合Loss
        axes[0].plot(loss_history, label='Total Train Loss', linewidth=2, color='blue')
        if val_loss_history:
            axes[0].plot(val_loss_history, label='Total Validation Loss', linewidth=2, color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 個別Loss
        if len(axes) > 1 and steering_loss_history and throttle_loss_history:
            axes[1].plot(steering_loss_history, label='Steering Loss', linewidth=2, color='red')
            axes[1].plot(throttle_loss_history, label='Throttle Loss', linewidth=2, color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Individual Loss')
            axes[1].set_title('Individual Training Loss History')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_path = os.path.join(config.MODEL_DIR, f"{model_name}_loss.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Loss履歴を保存しました:", loss_path)
    
    # 学習完了後のモデル変換確認は、main()関数でモデル保存後に実行
    # ここでは実行しない

    return epoch + 1


def save_model(model, optimizer, folder, model_name, epoch):
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, model_name)
    # CPUに移動してから保存（互換性のため）
    model_cpu = model.cpu()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    # 元のデバイスに戻す
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"モデルを保存しました: {model_path}")
    return model_path


def load_model(model, model_path=None, optimizer=None, folder='.', device=None):
    # モデルファイルの存在確認
    if os.path.exists(model_path):
        # TensorRTモデル（.trtファイル）の場合はtorch.jit.loadを使用
        if model_path.endswith('.trt'):
            try:
                trt_model = torch.jit.load(model_path, map_location='cpu')
                # TensorRTモデルは直接返すか、元のモデルに状態をコピー
                print(f"TensorRTモデルを読み込みました: {model_path}")
                return trt_model
            except Exception as e:
                print(f"TensorRTモデル読み込みエラー: {e}")
                raise e
        else:
            # PyTorchモデル（.pthファイル）の場合は従来通り
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # オプティマイザの状態をGPUに移動
                if device is not None and device.type == 'cuda':
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
            
            print("オプティマイザも読み込みました。")
        print(f"モデルを読み込みました: {model_path}")
        return checkpoint.get('epoch', 0)
    else:
        raise FileNotFoundError(f"configで指定したモデルファイルが見つかりません: {model_path}")

        model_files = [f for f in os.listdir(folder) if f.startswith('model_')]
        if model_files:
            print("利用可能なモデル:", model_files)
            model_name = input("読み込むモデル名を入力してください: ")
            path = os.path.join(folder, model_name)
            checkpoint = torch.load(path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # オプティマイザの状態をGPUに移動
                if device is not None and device.type == 'cuda':
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                
                print("オプティマイザも読み込みました。")
            print(f"モデルを読み込みました: {model_name}")
            return checkpoint.get('epoch', 0)
        else:
            print(model_path)
            print("利用可能なモデルが見つかりませんでした。")
            return 0


########################################
# モデル変換関数
########################################

def should_offer_model_conversion():
    """
    モデル変換を提案するかどうかを判定
    デバイスタイプに基づいて変換可能な場合のみTrue
    """
    # デバイス検出を実行（config.pyにはデフォルト値のみなので）
    try:
        from device_detection import detect_device
        device_info = detect_device()
        device_type = device_info.device_type
        
        # JetsonならTensorRT、RaspberryPiならOpenVINO
        return device_type.startswith('JETSON') or device_type.startswith('RPI')
    except ImportError:
        return False


def offer_model_conversion(model_name, model, saved_model_path=None):
    """
    学習完了後にモデル変換を提案する
    """
    # デバイス検出
    try:
        from device_detection import detect_device
        device_info = detect_device()
        device_type = device_info.device_type
        platform_name = device_info.platform_name
        
        print("\n" + "="*60)
        print("🚀 学習完了！モデル変換オプション")
        print("="*60)
        print(f"検出されたプラットフォーム: {platform_name}")
        
        # デバイスに応じた推奨変換方式
        if device_type.startswith('JETSON'):
            conversion_type = "TensorRT"
            print("💡 Jetsonデバイスが検出されました")
            print("   TensorRTへの変換により推論速度が大幅に向上します")
        elif device_type.startswith('RPI'):
            conversion_type = "OpenVINO"
            print("💡 Raspberry Piデバイスが検出されました")
            print("   OpenVINOへの変換により推論速度が向上します")
        else:
            print("⚠️  デバイス固有の最適化は利用できません")
            return
        
        print(f"\n推奨変換方式: {conversion_type}")
        print(f"モデル名: {model_name}")
        
        # 変換確認
        response = input(f"\n{conversion_type}形式に変換しますか？ (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print(f"\n{conversion_type}変換を開始します...")
            success = convert_model(model_name, model, conversion_type.lower(), device_type, saved_model_path)
            
            if success:
                print(f"✅ {conversion_type}変換が完了しました！")
                print("推論実行時にINFERENCE_ENGINEを変更してください：")
                if conversion_type == "TensorRT":
                    print('   config.py: INFERENCE_ENGINE = "tensorrt"')
                else:
                    print('   config.py: INFERENCE_ENGINE = "openvino"')
            else:
                print(f"❌ {conversion_type}変換に失敗しました")
        else:
            print("モデル変換をスキップしました")
        
        print("="*60)
        
    except ImportError as e:
        print(f"⚠️  デバイス検出モジュールが見つかりません: {e}")


def convert_model(model_name, model, conversion_type, device_type, saved_model_path=None):
    """
    モデルを指定された形式に変換する
    
    Args:
        model_name: モデル名
        model: 学習済みモデル
        conversion_type: 変換タイプ ("tensorrt" or "openvino")
        device_type: デバイスタイプ
        saved_model_path: 実際に保存されたモデルファイルパス
    
    Returns:
        bool: 変換成功時True
    """
    if saved_model_path and os.path.exists(saved_model_path):
        model_path = saved_model_path
    else:
        model_path = os.path.join(config.MODEL_DIR, model_name)
    
    try:
        if conversion_type == "tensorrt":
            return convert_to_tensorrt(model, model_path)
        elif conversion_type == "openvino":
            return convert_to_openvino(model, model_path)
        else:
            print(f"❌ サポートされていない変換タイプ: {conversion_type}")
            return False
            
    except Exception as e:
        print(f"❌ 変換中にエラーが発生しました: {e}")
        return False


def convert_to_tensorrt(model, model_path):
    """TensorRT変換を実行（annotation_training_d2j/tools/torch2trt_converter.pyを使用）"""
    try:
        # torch2trt_converter.pyが存在するか確認
        converter_path = os.path.join("annotation_training_d2j", "tools", "torch2trt_converter.py")
        
        if not os.path.exists(converter_path):
            print(f"❌ TensorRT変換スクリプトが見つかりません: {converter_path}")
            return False
        
        # torch2trtの利用可能性をチェック
        try:
            from torch2trt import torch2trt
            print("✅ torch2trtが利用可能です")
        except ImportError:
            print("❌ torch2trtライブラリが利用できません")
            print("   torch2trtのインストール方法:")
            print("   git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install")
            return False
        
        # モデルタイプを特定
        plan_name = None
        for name in ['donkeycar', 'resnet18', 'mobilevit_xxs', 'edgenext_xxsmall']:
            if name in model_path.lower():
                plan_name = name
                break
        
        if plan_name is None:
            print("❌ モデルタイプを特定できませんでした")
            return False
        
        # TensorRT変換されたモデルの保存先（torch2trt形式）
        if model_path.endswith('.pth'):
            trt_save_path = model_path.replace('.pth', '_trt.pth')
        else:
            trt_save_path = model_path + '_trt.pth'
        
        print(f"🔄 torch2trt_converter.pyを使用してTensorRT変換を実行...")
        print(f"モデルタイプ: {plan_name}")
        print(f"入力サイズ: {config.IMAGE_H}x{config.IMAGE_W}")
        print(f"元モデル: {model_path}")
        print(f"変換後: {trt_save_path}")
        
        # torch2trt_converter.pyから変換関数をインポートして直接使用
        sys.path.insert(0, os.path.join("annotation_training_d2j", "tools"))
        from torch2trt_converter import convert_pytorch_to_tensorrt, load_model_weights
        
        # model_catalogからモデルを取得
        sys.path.insert(0, "annotation_training_d2j")
        from model_catalog import get_model
        
        # モデルを作成
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_model = get_model(plan_name, pretrained=False, input_size=(config.IMAGE_H, config.IMAGE_W))
        
        # モデルファイルが存在するか確認
        if not os.path.exists(model_path):
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return False
        
        # 重みを読み込む
        new_model = load_model_weights(new_model, model_path, device)
        new_model = new_model.to(device)
        new_model.eval()
        
        # TensorRT変換実行
        model_trt = convert_pytorch_to_tensorrt(
            new_model,
            input_size=(config.IMAGE_H, config.IMAGE_W),
            batch_size=1,
            fp16_mode=True,  # FP16モードで高速化
            save_path=trt_save_path,
            device=device
        )
        
        if model_trt is not None:
            print("✅ TensorRT変換が完了しました")
            print(f"保存先: {trt_save_path}")
            return True
        else:
            print("❌ TensorRT変換に失敗しました")
            return False
            
    except ImportError as e:
        print(f"❌ 必要なモジュールをインポートできません: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorRT変換実行エラー: {e}")
        return False


def convert_to_openvino(model, model_path):
    """OpenVINO変換を実行"""
    try:
        import sys
        
        # annotation_training_d2j/tools/pytorch_to_openvino.py を使用
        converter_path = os.path.join("annotation_training_d2j", "tools", "pytorch_to_openvino.py")
        
        if not os.path.exists(converter_path):
            print(f"❌ OpenVINO変換スクリプトが見つかりません: {converter_path}")
            return False
        
        sys.path.append(os.path.join("annotation_training_d2j", "tools"))
        
        try:
            from pytorch_to_openvino import convert_pytorch_to_openvino
            
            # モデルタイプ特定
            plan_name = None
            for name in ['donkeycar', 'resnet18', 'mobilevit_xxs', 'edgenext_xxsmall']:
                if name in model_path.lower():
                    plan_name = name
                    break
            
            if plan_name is None:
                print("❌ モデルタイプを特定できませんでした")
                return False
            
            # OpenVINO変換実行
            openvino_save_path = model_path.replace('.pth', '_openvino')
            
            result_path = convert_pytorch_to_openvino(
                model_path=model_path,
                model_type=plan_name,
                output_path=openvino_save_path,
                precision='FP32',
                compress_to_fp16=True
            )
            
            return result_path is not None
            
        except ImportError as e:
            print(f"❌ OpenVINO変換ライブラリが見つかりません: {e}")
            print("OpenVINOをインストールしてください: pip install openvino-dev")
            return False
            
    except Exception as e:
        print(f"❌ OpenVINO変換エラー: {e}")
        return False


def export_to_onnx_for_tensorrt(model, model_path, input_size):
    """
    TensorRT変換の準備としてONNXファイルを出力
    """
    try:
        print("ONNXファイルを生成しています...")
        
        # デバイス設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # ONNX出力パス
        onnx_path = model_path.replace('.pth', '.onnx')
        
        # ダミー入力作成
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # ONNXエクスポート
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNXファイルを出力しました: {onnx_path}")
        print("\n次のステップでTensorRTエンジンを作成できます:")
        print(f"   trtexec --onnx={onnx_path} --saveEngine={model_path.replace('.pth', '.trt')} --fp16")
        print("   または：")
        print(f"   /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={model_path.replace('.pth', '.trt')} --fp16")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX出力エラー: {e}")
        return False


########################################
# テスト / 推論デモ
########################################
def test_model(model, model_path, dataset, sample_num=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n保存したモデルをロードします。")
    load_model(model, model_path, None, config.MODEL_DIR)
    model = model.to(device)
    print("使用モデル:\n", model)
    print(f"推論デバイス: {device}")

    print(f"\n推論デモ: ランダムに {sample_num} サンプルを取り出して予測します。")
    testloader = DataLoader(dataset, batch_size=1, shuffle=True)

    x_cat = torch.tensor([])
    y_cat = torch.tensor([])
    yh_cat = torch.tensor([])

    loader_iter = iter(testloader)
    for _ in range(sample_num):
        x_batch, y_batch = next(loader_iter)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        x_cat = torch.cat([x_cat, x_batch.cpu()])
        y_cat = torch.cat([y_cat, y_batch.cpu()])

        # 予測
        yh_batch = model.predict(model, x_batch)
        yh_cat = torch.cat([yh_cat, yh_batch.cpu()])

    print("\n入力データ:\n", x_cat)
    print("正解データ:\n", y_cat)
    print("予測結果:\n", yh_cat)


########################################
# config.py設定による自動学習機能
########################################
def get_data_directory():
    """データディレクトリを選択（複数ある場合はユーザー選択）"""
    if config.SAVE_FORMAT.lower() == "donkeycar":
        # DonkeyCar形式の場合、dataディレクトリから選択
        data_dir = "data"
        if os.path.exists(data_dir):
            data_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
            if data_folders:
                data_folders.sort()
                
                if len(data_folders) == 1:
                    # 1つしかない場合はそれを選択
                    selected_folder = data_folders[0]
                    print(f"データフォルダを自動選択: {selected_folder}")
                    return os.path.join(data_dir, selected_folder)
                else:
                    # 複数ある場合はユーザーに選択させる
                    print("\n複数のデータフォルダが見つかりました:")
                    for i, folder in enumerate(data_folders, 1):
                        print(f"{i}. {folder}")
                    
                    print(f"\nデフォルト: {data_folders[-1]} (最新)")
                    
                    while True:
                        choice = input(f"使用するデータフォルダを選択してください (1-{len(data_folders)}, デフォルト: {len(data_folders)}): ").strip()
                        
                        if choice == "":
                            # デフォルト（最新）を選択
                            selected_folder = data_folders[-1]
                            print(f"最新のフォルダを選択: {selected_folder}")
                            break
                        elif choice.isdigit() and 1 <= int(choice) <= len(data_folders):
                            selected_folder = data_folders[int(choice) - 1]
                            print(f"選択されたフォルダ: {selected_folder}")
                            break
                        else:
                            print(f"無効な選択です。1-{len(data_folders)}の数字を入力してください。")
                    
                    return os.path.join(data_dir, selected_folder)
            else:
                print("エラー: dataディレクトリにデータセットが見つかりません。")
                sys.exit(1)
        else:
            print("エラー: dataディレクトリが存在しません。")
            sys.exit(1)
    else:
        # CSV形式の場合、recordsディレクトリ
        if os.path.exists(config.RECORDS_DIRECTORY):
            return config.RECORDS_DIRECTORY
        else:
            print(f"エラー: {config.RECORDS_DIRECTORY} ディレクトリが存在しません。")
            sys.exit(1)

def load_data_auto(folder, data_format, plan):
    """自動選択モードでデータを読み込む"""
    if data_format == "donkeycar":
        # DonkeyCar形式のデータを読み込み（プランに応じて処理）
        if plan == "nn":
            # 超音波センサーNNの場合
            _, angles, throttles, ultrasonic_data = load_donkeycar_data(folder, auto_select=True, plan=plan)
            
            # DataFrameに変換
            df = pd.DataFrame({
                'steering': angles,
                'throttle': throttles
            })
            
            # 超音波センサーデータを追加
            for sensor in config.ULTRASONIC_SENSOR_LIST:
                df[sensor] = ultrasonic_data[sensor]
        else:
            # 画像ベースモデルの場合
            images, angles, throttles, _ = load_donkeycar_data(folder, auto_select=True, plan=plan)
            
            # DataFrameに変換
            df = pd.DataFrame({
                'image_file': images,
                'steering': angles,
                'throttle': throttles
            })
            
            # 超音波センサーデータは0で初期化（画像モデルでは使用しない）
            for sensor in config.ULTRASONIC_SENSOR_LIST:
                df[sensor] = 0
        
        csv_file = os.path.basename(folder)
        print("\n入力データのサイズ:", df.shape)
        
        x_tensor, y_tensor = preprocess_data(df)
        ts = None
        
        return x_tensor, y_tensor, csv_file, ts
    else:
        # CSV形式の場合は通常のload_dataを呼び出し
        return load_data(folder, data_format, plan)

def validate_plan():
    """プランの妥当性を検証"""
    learning_plans = ["nn", "donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]
    if config.PLAN not in learning_plans:
        print(f"警告: config.PLAN '{config.PLAN}' は学習対応プランではありません。")
        print(f"学習対応プラン: {learning_plans}")
        print("'donkeycar'に変更して続行します。")
        return "donkeycar"
    return config.PLAN

def show_training_config(data_folder, plan, data_format):
    """学習設定を表示して確認"""
    print("=" * 80)
    print("自動学習設定")
    print("=" * 80)
    
    print(f"\n【設定情報】")
    print(f"データ形式: {data_format}")
    print(f"学習プラン: {plan}")
    print(f"データディレクトリ: {data_folder}")
    print(f"学習エポック数: {config.EPOCHS}")
    print(f"バッチサイズ: {config.BATCH_SIZE}")
    print(f"モデル保存先: {config.MODEL_DIR}")
    
    if plan == "nn":
        print(f"\n【超音波センサーNN設定】")
        print(f"使用センサー: {config.ULTRASONIC_SENSOR_LIST}")
        print(f"隠れ層ノード数: {config.HIDDEN_DIM}")
        print(f"隠れ層数: {config.NUM_HIDDEN_LAYERS}")
    else:
        print(f"\n【画像モデル設定】")
        print(f"画像サイズ: {config.IMAGE_W} x {config.IMAGE_H}")
        print(f"カラーチャンネル: {config.IMAGE_DEPTH}")
        print(f"推論エンジン: {config.INFERENCE_ENGINE}")
    
    # データの確認
    if data_format == "donkeycar":
        try:
            if plan == "nn":
                _, angles, throttles, ultrasonic_data = load_donkeycar_data(data_folder, auto_select=True, plan=plan)
                data_count = len(angles)
                print(f"\n【データ統計】")
                print(f"総レコード数: {data_count}")
                print(f"ステアリング範囲: [{min(angles):.3f}, {max(angles):.3f}]")
                print(f"スロットル範囲: [{min(throttles):.3f}, {max(throttles):.3f}]")
            else:
                images, angles, throttles, _ = load_donkeycar_data(data_folder, auto_select=True, plan=plan)
                data_count = len(images)
                print(f"\n【データ統計】")
                print(f"総レコード数: {data_count}")
                print(f"画像ファイル例: {os.path.basename(images[0]) if images else 'N/A'}")
                print(f"ステアリング範囲: [{min(angles):.3f}, {max(angles):.3f}]")
                print(f"スロットル範囲: [{min(throttles):.3f}, {max(throttles):.3f}]")
        except Exception as e:
            print(f"\n【データ確認エラー】: {e}")
            print("データ読み込みでエラーが発生しました。")
    
    print("\n" + "=" * 80)
    
    # ユーザー確認
    while True:
        response = input("\nこの設定で学習を開始しますか？ (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no', '']:
            print("学習を中止しました。")
            sys.exit()
        else:
            print("y(yes) または n(no) で回答してください。")

def main():
    """config.pyの設定に基づく自動学習"""
    
    print("config.pyの設定に基づく自動学習を開始します...\n")
    print(f"SAVE_FORMAT: {config.SAVE_FORMAT}\n")
    
    # CUDAの利用可能性をチェック
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CPUモードで学習を実行します")
    
    # 1. 設定の検証と取得
    plan = validate_plan()
    data_format = config.SAVE_FORMAT.lower()
    data_folder = get_data_directory()
    
    # 2. 学習設定を表示して確認
    show_training_config(data_folder, plan, data_format)
    
    print("\n学習を開始しています...")
    
    # 3. データをロード（自動選択モードで）
    x_tensor, y_tensor, csv_file, ts = load_data_auto(data_folder, data_format, plan)
    
    if data_format == "donkeycar":
        if plan == "nn":
            print("超音波センサーデータを使用してNNモデルを学習します")
        else:
            print("画像データを使用して画像ベースモデルを学習します")

    # 4. データセット作成 & モデル定義
    if plan in ["donkeycar", "resnet18", "mobilevit_xxs", "edgenext_xxsmall"]:
        # 画像用 Dataset
        if data_format == "donkeycar" and plan != "nn":
            # DonkeyCar形式の画像データ
            images, angles, throttles, _ = load_donkeycar_data(data_folder, auto_select=True, plan=plan)
            image_files = images
        else:
            # CSV形式の画像データ
            df_path = os.path.join(data_folder, csv_file) if data_format == "csv" else None
            if df_path and os.path.exists(df_path):
                df = pd.read_csv(df_path)
                image_files = df['image_file'].tolist()
            else:
                print("画像ファイルリストが取得できません")
                sys.exit()
        
        # データオーグメンテーションの設定
        if config.USE_DATA_AUGMENTATION:
            augmentation_transforms = []
            
            # 水平反転
            if config.AUG_USE_FLIP:
                augmentation_transforms.append(
                    transforms.RandomHorizontalFlip(p=config.AUG_FLIP_PROB)
                )
            
            # 色調整
            if config.AUG_USE_COLOR:
                augmentation_transforms.append(
                    transforms.ColorJitter(
                        brightness=config.AUG_BRIGHTNESS,
                        contrast=config.AUG_CONTRAST,
                        saturation=config.AUG_SATURATION
                    )
                )
            
            # 幾何変換
            if config.AUG_USE_GEOMETRY:
                augmentation_transforms.append(
                    transforms.RandomAffine(
                        degrees=config.AUG_ROTATION_DEGREES,
                        translate=(config.AUG_TRANSLATE_RATIO, config.AUG_TRANSLATE_RATIO)
                    )
                )
            
            # ベース変換（リサイズ、テンソル化）
            base_transforms = [
                transforms.Resize((config.IMAGE_H, config.IMAGE_W)),
                transforms.ToTensor()
            ]
            
            # ランダムイレース（ToTensor後に適用）
            if config.AUG_USE_ERASE:
                base_transforms.append(
                    transforms.RandomErasing(
                        p=config.AUG_ERASE_PROB,
                        scale=(config.AUG_ERASE_MIN_RATIO, config.AUG_ERASE_MAX_RATIO),
                        ratio=(0.3, 3.3),
                        value=0
                    )
                )
            
            transform = transforms.Compose(augmentation_transforms + base_transforms)
            print("データオーグメンテーションを有効化しました")
        else:
            transform = transforms.Compose([
                transforms.Resize((config.IMAGE_H, config.IMAGE_W)),
                transforms.ToTensor(),
            ])
        dataset = CustomImageDataset(image_files=image_files, y_tensor=y_tensor, transform=transform)
        
        # データをトレーニング・検証用に分割（Early Stopping用）
        val_dataloader = None
        if config.USE_EARLY_STOPPING and len(dataset) > 10:  # データセットが十分な大きさの場合のみ
            from torch.utils.data import random_split
            val_ratio = 0.2  # 20%を検証用に
            val_size = int(len(dataset) * val_ratio)
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            print(f"データセット分割: トレーニング {train_size}, 検証 {val_size}")
        else:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # model_catalogからモデルを取得
        model = get_model_from_catalog(plan, for_training=True)
        if model is None:
            print(f"Failed to create model for {plan}, falling back to ConvolutionalNeuralNetwork")
            input_dim = (config.IMAGE_W, config.IMAGE_H, config.IMAGE_DEPTH)
            output_dim = y_tensor.shape[1]
            model = ConvolutionalNeuralNetwork(input_dim, output_dim)

    else:
        # 数値用 Dataset (nn) - 超音波センサー
        dataset = CustomDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = None

        input_dim = x_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        model = NeuralNetwork(input_dim, output_dim, config.HIDDEN_DIM, config.NUM_HIDDEN_LAYERS)

    print(f"\n使用モデル: {type(model).__name__}")
    
    # 継続学習の確認とモデル読み込み
    pretrained_model_path = None
    if hasattr(config, 'MODEL_NAME') and config.MODEL_NAME:
        pretrained_model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
        
        if os.path.exists(pretrained_model_path):
            print(f"\n既存のモデルが見つかりました: {config.MODEL_NAME}")
            print(f"パス: {pretrained_model_path}")
            
            while True:
                response = input("\nこのモデルから継続学習を開始しますか？ (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    try:
                        # モデルを読み込み
                        checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        
                        # エポック情報があれば取得
                        if 'epoch' in checkpoint:
                            start_epoch = checkpoint['epoch']
                            print(f"前回の学習エポック: {start_epoch}")
                        
                        print("✓ 事前学習済みモデルを読み込みました")
                        print("継続学習を開始します...")
                        break
                    except Exception as e:
                        print(f"エラー: モデルの読み込みに失敗しました - {e}")
                        print("新しいモデルで学習を開始します...")
                        start_epoch = 0
                        break
                elif response in ['n', 'no', '']:
                    print("新しいモデルで学習を開始します...")
                    start_epoch = 0
                    break
                else:
                    print("y(yes) または n(no) で回答してください。")
        else:
            print(f"\n指定されたモデルファイルが見つかりません: {pretrained_model_path}")
            print("新しいモデルで学習を開始します...")
            start_epoch = 0
    else:
        start_epoch = 0

    # 5. 損失関数 & Optimizer
    if config.MODEL_TYPE == "categorical":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # オプティマイザの状態も復元（継続学習の場合）
    if pretrained_model_path and os.path.exists(pretrained_model_path) and start_epoch > 0:
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # オプティマイザの状態をGPUに移動
                if device.type == 'cuda':
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                
                print("✓ オプティマイザの状態も復元しました")
        except Exception as e:
            print(f"警告: オプティマイザの状態復元に失敗: {e}")

    # 6. 学習設定
    epochs = config.EPOCHS
    
    # 7. モデル名を作成
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if start_epoch > 0:
        model_name = f"{plan}_{date_str}_{csv_file}_continue_epoch{start_epoch}".replace(".csv","")
    else:
        model_name = f"{plan}_{date_str}_{csv_file}".replace(".csv","")
    
    print(f"\n学習エポック数: {epochs}")
    print(f"開始エポック: {start_epoch}")
    print(f"モデル名: {model_name}")
    if start_epoch > 0:
        print(f"継続学習: {config.MODEL_NAME} から再開")

    # 8. トレーニング
    print(f"\n=== 学習開始 ===")
    end_epoch = train_model(model, dataloader, criterion, optimizer, model_name, start_epoch, epochs, device, val_dataloader)

    # 9. モデル保存
    model_path = save_model(model, optimizer, config.MODEL_DIR, model_name, end_epoch)

    # 9.5. 学習完了後のモデル変換確認（モデル保存後に実行）
    if should_offer_model_conversion():
        offer_model_conversion(model_name, model, model_path)

    # 10. 推論デモ（可能な場合）
    if plan == "nn" and hasattr(model, 'predict'):
        try:
            test_model(model, model_path, dataset, sample_num=3, device=device)
        except Exception as e:
            print(f"推論デモでエラーが発生しました: {e}")
    else:
        print("推論デモはスキップします")
    
    print(f"\n=== 学習完了 ===")
    print(f"保存されたモデル: {model_path}")
    print(f"モデルサイズ: {os.path.getsize(model_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()