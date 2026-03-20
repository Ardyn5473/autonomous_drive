#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRTモデルのテストツール

Usage:
    python tools/test_trt_model.py models/donkey_converted.trt
    python tools/test_trt_model.py models/resnet18_fp16.trt --input-size 1 3 224 224
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

def test_trt_model(model_path, input_size=[1, 3, 224, 224], num_runs=10):
    """TensorRTモデルをテスト"""
    
    print("🚀 TensorRTモデルテスト開始")
    print(f"📁 モデルパス: {model_path}")
    print(f"📊 入力サイズ: {input_size}")
    print("-" * 50)
    
    # ファイル存在確認
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return False
    
    # ファイルサイズ表示
    file_size_bytes = os.path.getsize(model_path)
    if file_size_bytes > 1024 * 1024:
        file_size = file_size_bytes / 1024 / 1024
        print(f"📁 ファイルサイズ: {file_size:.1f} MB")
    elif file_size_bytes > 1024:
        file_size = file_size_bytes / 1024
        print(f"📁 ファイルサイズ: {file_size:.1f} KB")
    else:
        print(f"📁 ファイルサイズ: {file_size_bytes} bytes")
    
    try:
        # デバイス設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 デバイス: {device}")
        
        # モデル読み込み
        print("📥 TensorRTモデル読み込み中...")
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("✅ モデル読み込み完了")
        
        # テスト用入力データ作成
        batch_size, channels, height, width = input_size
        test_input = torch.randn(batch_size, channels, height, width).to(device)
        print(f"📊 テスト入力作成: {test_input.shape}")
        
        # 推論テスト
        print("🔍 推論テスト実行中...")
        with torch.no_grad():
            # ウォームアップ
            for _ in range(3):
                _ = model(test_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 推論時間測定
            times = []
            for i in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                output = model(test_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                if (i + 1) % 5 == 0:
                    print(f"  進行状況: {i+1}/{num_runs}")
        
        # 結果計算
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / mean_time
        
        # 推論結果表示
        print(f"📊 出力形状: {output.shape}")
        print(f"📊 出力例（最初の5値）: {output.flatten()[:5].cpu().numpy()}")
        
        print("\n" + "="*50)
        print("📊 性能結果")
        print("="*50)
        print(f"平均推論時間: {mean_time*1000:.2f} ms")
        print(f"標準偏差:     {std_time*1000:.2f} ms")
        print(f"最小時間:     {min_time*1000:.2f} ms")
        print(f"最大時間:     {max_time*1000:.2f} ms")
        print(f"平均FPS:      {fps:.1f}")
        print("="*50)
        
        # メモリ使用量表示（CUDA使用時）
        if torch.cuda.is_available():
            print(f"\n💾 GPU メモリ使用量:")
            print(f"   割り当て済み: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"   キャッシュ済み: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        print("\n✅ テスト完了！モデルは正常に動作しています")
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # GPU メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='TensorRTモデルテストツール')
    parser.add_argument('model_path', type=str, help='TensorRTモデルファイルのパス')
    parser.add_argument('--input-size', nargs=4, type=int, default=[1, 3, 224, 224],
                       help='入力テンソルのサイズ [batch, channels, height, width]（デフォルト: 1 3 224 224）')
    parser.add_argument('--runs', type=int, default=10, help='推論回数（デフォルト: 10）')
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    success = test_trt_model(args.model_path, args.input_size, args.runs)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()