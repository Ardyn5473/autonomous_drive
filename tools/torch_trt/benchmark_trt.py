#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT推論速度ベンチマークツール

Usage:
    python tools/benchmark_trt.py --model donkey
    python tools/benchmark_trt.py --model resnet18
    python tools/benchmark_trt.py --model mobilevit_xxs  
    python tools/benchmark_trt.py --model edgenext_xxsmall
    python tools/benchmark_trt.py --all
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch_tensorrt
import timm
from torchvision import models

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from train_pytorch import get_model_from_catalog, load_model

def create_test_input(batch_size=1, channels=3, height=224, width=224):
    """テスト用の入力データを作成"""
    return torch.randn(batch_size, channels, height, width).float()

def benchmark_model(model, input_tensor, num_runs=50, warmup_runs=10, model_name="Model"):
    """モデルの推論速度をベンチマーク"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()
    
    print(f"{model_name} ウォームアップ中... ({warmup_runs}回)")
    # ウォームアップ
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # GPU同期
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"{model_name} ベンチマーク実行中... ({num_runs}回)")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if (i + 1) % 10 == 0:
                print(f"  進行状況: {i+1}/{num_runs}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'all_times': times,
        'last_output': output
    }

def convert_to_tensorrt(pytorch_model, input_tensor, precision="fp32"):
    """PyTorchモデルをTensorRTに変換"""
    print(f"TensorRT変換開始... (精度: {precision})")
    
    # モデルをevalモードに設定
    pytorch_model.eval()
    
    # 変換設定
    if precision == "fp16":
        enabled_precisions = {torch.float, torch.half}
    elif precision == "int8":
        enabled_precisions = {torch.float, torch.half, torch.int8}
    else:
        enabled_precisions = {torch.float}
    
    try:
        # torch_tensorrtを使ってモデル変換
        trt_model = torch_tensorrt.compile(
            pytorch_model,
            inputs=[input_tensor],
            enabled_precisions=enabled_precisions,
            workspace_size=1 << 28,  # 256MB
            min_block_size=1,
            use_python_runtime=True,
        )
        print("✅ TensorRT変換完了")
        return trt_model
    except Exception as e:
        print(f"❌ TensorRT変換エラー: {e}")
        return None

def get_pretrained_model(model_name):
    """事前学習済みモデルを取得"""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model
    elif model_name == "mobilevit_xxs":
        model = timm.create_model('mobilevit_xxs', pretrained=True)
        # 出力層を調整（回帰用に2次元出力に変更）
        if hasattr(model, 'classifier'):
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        elif hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                model.head.fc = torch.nn.Linear(model.head.fc.in_features, 2)
            else:
                model.head = torch.nn.Linear(model.head.in_features, 2)
        else:
            # ByobNet の場合、最後のlayerを確認
            last_layer = list(model.modules())[-1]
            if isinstance(last_layer, torch.nn.Linear):
                in_features = last_layer.in_features
                for name, module in model.named_modules():
                    if module is last_layer:
                        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                        layer_name = name.rsplit('.', 1)[1] if '.' in name else name
                        if parent_name:
                            parent = model.get_submodule(parent_name)
                            setattr(parent, layer_name, torch.nn.Linear(in_features, 2))
                        else:
                            setattr(model, layer_name, torch.nn.Linear(in_features, 2))
                        break
        return model
    elif model_name == "edgenext_xxsmall":
        model = timm.create_model('edgenext_xx_small', pretrained=True)
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, 2)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

def benchmark_single_model(model_name, num_runs=50):
    """単一モデルのベンチマーク"""
    print(f"\n{'='*60}")
    print(f"🔧 {model_name.upper()} モデルベンチマーク開始")
    print(f"{'='*60}")
    
    try:
        # テスト用入力データ作成
        test_input = create_test_input()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input_gpu = test_input.to(device)
        
        # モデル取得
        if model_name == "donkey":
            # 既存のdonkeyモデル
            original_engine = config.INFERENCE_ENGINE
            config.INFERENCE_ENGINE = 'pytorch'
            
            pytorch_model = get_model_from_catalog(config.PLAN)
            if pytorch_model is None:
                print("❌ Donkeyモデルの作成に失敗")
                return None
                
            model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
            if os.path.exists(model_path):
                load_model(pytorch_model, model_path=model_path)
            else:
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return None
                
            config.INFERENCE_ENGINE = original_engine
        else:
            # 事前学習済みモデル
            print(f"📥 {model_name} 事前学習済みモデル取得中...")
            pytorch_model = get_pretrained_model(model_name)
        
        pytorch_model = pytorch_model.to(device)
        print(f"✅ {model_name} モデル準備完了")
        
        # PyTorchベンチマーク実行
        print(f"\n⏱️  {model_name} PyTorchベンチマーク実行中...")
        pytorch_results = benchmark_model(pytorch_model, test_input_gpu, num_runs=num_runs, model_name=f"{model_name} PyTorch")
        
        # TensorRT変換とベンチマーク
        print(f"\n🔄 {model_name} TensorRT変換中...")
        trt_model = convert_to_tensorrt(pytorch_model, test_input_gpu, precision="fp32")
        
        tensorrt_results = None
        if trt_model is not None:
            print(f"\n⏱️  {model_name} TensorRTベンチマーク実行中...")
            tensorrt_results = benchmark_model(trt_model, test_input_gpu, num_runs=num_runs, model_name=f"{model_name} TensorRT")
        else:
            print(f"❌ {model_name} TensorRTモデルが利用できません")
        
        # 推論結果の一致性確認
        if tensorrt_results is not None:
            print(f"\n🔍 {model_name} 推論結果の一致性確認...")
            pytorch_output = pytorch_results['last_output']
            tensorrt_output = tensorrt_results['last_output']
            
            if pytorch_output.is_cuda:
                pytorch_output = pytorch_output.cpu()
            if tensorrt_output.is_cuda:
                tensorrt_output = tensorrt_output.cpu()
            
            diff = torch.abs(pytorch_output - tensorrt_output)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            
            print(f"最大差異: {max_diff:.6f}")
            print(f"平均差異: {mean_diff:.6f}")
            
            if max_diff < 1e-3:
                print("✅ 推論結果は良好に一致しています")
            elif max_diff < 1e-2:
                print("⚠️  推論結果に軽微な差異がありますが許容範囲内です")
            else:
                print("❌ 推論結果に大きな差異があります")
        
        return {
            'model_name': model_name,
            'pytorch_results': pytorch_results,
            'tensorrt_results': tensorrt_results
        }
        
    except Exception as e:
        print(f"❌ {model_name} ベンチマークエラー: {e}")
        return {
            'model_name': model_name,
            'pytorch_results': None,
            'tensorrt_results': None,
            'error': str(e)
        }
    finally:
        # GPU メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def print_benchmark_results(results_list):
    """ベンチマーク結果を表示"""
    print("\n" + "="*80)
    print("📊 TensorRT ベンチマーク結果")
    print("="*80)
    
    print(f"{'モデル名':<18} {'エンジン':<10} {'平均時間(ms)':<12} {'FPS':<8} {'速度向上':<8}")
    print("-"*80)
    
    for result in results_list:
        model_name = result['model_name']
        pytorch_results = result.get('pytorch_results')
        tensorrt_results = result.get('tensorrt_results')
        
        if pytorch_results:
            print(f"{model_name:<18} {'PyTorch':<10} {pytorch_results['mean_time']*1000:.2f}{'':<8} {pytorch_results['fps']:.1f}{'':<4} {'1.00x':<8}")
            
            if tensorrt_results:
                speedup = pytorch_results['mean_time'] / tensorrt_results['mean_time']
                print(f"{'':<18} {'TensorRT':<10} {tensorrt_results['mean_time']*1000:.2f}{'':<8} {tensorrt_results['fps']:.1f}{'':<4} {speedup:.2f}x")
            else:
                print(f"{'':<18} {'TensorRT':<10} {'変換失敗':<12} {'-':<8} {'-':<8}")
        else:
            print(f"{model_name:<18} {'ERROR':<10} {'-':<12} {'-':<8} {'-':<8}")
        
        print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='TensorRT推論ベンチマークツール')
    parser.add_argument('--model', choices=['donkey', 'resnet18', 'mobilevit_xxs', 'edgenext_xxsmall'], 
                       help='ベンチマークするモデル名')
    parser.add_argument('--all', action='store_true', help='全てのモデルをベンチマーク')
    parser.add_argument('--runs', type=int, default=50, help='測定回数（デフォルト: 50）')
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        parser.print_help()
        return
    
    print("🚀 TensorRT推論ベンチマーク開始")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch_tensorrt version: {torch_tensorrt.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-" * 50)
    
    results = []
    
    if args.all:
        models_to_test = ['donkey', 'resnet18', 'mobilevit_xxs', 'edgenext_xxsmall']
    else:
        models_to_test = [args.model]
    
    for model_name in models_to_test:
        result = benchmark_single_model(model_name, args.runs)
        if result:
            results.append(result)
    
    # 結果表示
    if results:
        print_benchmark_results(results)
        
        # メモリ情報表示
        if torch.cuda.is_available():
            print(f"\n💾 最終GPU メモリ使用量:")
            print(f"   割り当て済み: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"   キャッシュ済み: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()