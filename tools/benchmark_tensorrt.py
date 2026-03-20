#!/usr/bin/env python3
# benchmark_tensorrt.py
# TensorRT vs PyTorchの推論速度比較

import sys
import os
import time
import argparse
import glob
import numpy as np
import torch

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

import config
from model_inference import load_model_with_engine

# torch2trt_converter.pyから関数をインポート
sys.path.insert(0, os.path.join(project_root, 'tools'))
from torch2trt_converter import find_pytorch_models, infer_model_type_from_filename, get_available_model_types

def benchmark_model(model_path, model_type, inference_engine, num_iterations=100, **kwargs):
    """モデルの推論速度をベンチマーク"""
    print(f"\n{'='*60}")
    print(f"ベンチマーク: {model_type} ({inference_engine})")
    print(f"{'='*60}")

    # モデル読み込み
    model = load_model_with_engine(
        model_path=model_path,
        model_type=model_type,
        inference_engine=inference_engine,
        **kwargs
    )

    if model is None:
        print("❌ モデル読み込み失敗")
        return None

    print(f"✅ モデル読み込み成功")

    # ウォームアップ（最初の推論は遅い）
    test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    if hasattr(model, 'run') and model_type == "driving":
        # 自動運転モデルのmodel_catalogのrunメソッド
        for _ in range(5):
            _ = model.run(test_input)
    else:
        # 位置推論モデルまたは通常のPyTorch/TensorRTモデル
        test_tensor = torch.from_numpy(test_input).float().permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()

        with torch.no_grad():
            for _ in range(5):
                _ = model(test_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"ウォームアップ完了")

    # ベンチマーク
    times = []
    for i in range(num_iterations):
        test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        start = time.perf_counter()

        if hasattr(model, 'run') and model_type == "driving":
            _ = model.run(test_input)
        else:
            test_tensor = torch.from_numpy(test_input).float().permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                test_tensor = test_tensor.cuda()

            with torch.no_grad():
                _ = model(test_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ミリ秒

    # 統計情報
    times = np.array(times)
    print(f"\n📊 推論時間統計 ({num_iterations}回の推論)")
    print(f"   平均: {times.mean():.2f} ms")
    print(f"   中央値: {np.median(times):.2f} ms")
    print(f"   最小: {times.min():.2f} ms")
    print(f"   最大: {times.max():.2f} ms")
    print(f"   標準偏差: {times.std():.2f} ms")
    print(f"   FPS: {1000.0 / times.mean():.1f}")

    return times.mean()


def main():
    parser = argparse.ArgumentParser(
        description='TensorRT vs PyTorch 推論速度ベンチマーク',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
    # 対話的にモデルを選択
    python tools/benchmark_tensorrt.py

    # 特定のモデルを指定
    python tools/benchmark_tensorrt.py --model models/donkeycar_20251101_144257.pth

    # modelsフォルダの全モデルをベンチマーク
    python tools/benchmark_tensorrt.py --all

    # 測定回数を変更
    python tools/benchmark_tensorrt.py --iterations 200
        """
    )
    parser.add_argument('--model', type=str, help='ベンチマークするモデルのパス')
    parser.add_argument('--all', action='store_true', help='modelsフォルダの全モデルをベンチマーク')
    parser.add_argument('--iterations', type=int, default=100, help='測定回数（デフォルト: 100）')
    parser.add_argument('--models-dir', type=str, default='models', help='モデルディレクトリ（デフォルト: models）')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TensorRT vs PyTorch 推論速度ベンチマーク")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 利用可能なモデルタイプを取得
    available_models = get_available_model_types()

    # ベンチマーク対象モデルを決定
    models_to_benchmark = []

    if args.all:
        # 全モデルをベンチマーク
        pth_files = find_pytorch_models(args.models_dir)
        if not pth_files:
            print(f"❌ {args.models_dir} 内にモデルが見つかりません")
            return
        models_to_benchmark = pth_files
        print(f"\n📋 {len(models_to_benchmark)}個のモデルをベンチマークします")

    elif args.model:
        # 指定されたモデルをベンチマーク
        if not os.path.exists(args.model):
            print(f"❌ モデルが見つかりません: {args.model}")
            return
        models_to_benchmark = [args.model]

    else:
        # 対話的にモデルを選択
        pth_files = find_pytorch_models(args.models_dir)
        if not pth_files:
            print(f"❌ {args.models_dir} 内にモデルが見つかりません")
            return

        print("\n=== ベンチマーク可能なモデル ===")
        for i, model_path in enumerate(pth_files):
            model_type, category = infer_model_type_from_filename(model_path, available_models)
            print(f"{i+1}. {model_path:60s} ({model_type}, {category})")

        while True:
            try:
                choice = input("\nベンチマークするモデルの番号を入力してください（aで全て、qで終了）: ")
                if choice.lower() == 'q':
                    return
                elif choice.lower() == 'a':
                    models_to_benchmark = pth_files
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(pth_files):
                        models_to_benchmark = [pth_files[idx]]
                        break
                    else:
                        print("有効な番号を入力してください。")
            except ValueError:
                print("数字、'a'、または 'q' を入力してください。")

    # ベンチマーク実行
    all_results = {}

    for model_path in models_to_benchmark:
        # モデルタイプを推測
        model_type, category = infer_model_type_from_filename(model_path, available_models)

        print("\n\n" + "#"*60)
        print(f"# {os.path.basename(model_path)}")
        print(f"# タイプ: {model_type} ({category})")
        print("#"*60)

        # モデル固有のパラメータを設定
        kwargs = {}
        if category == "driving":
            # 自動運転モデルの場合、planパラメータを設定
            kwargs['plan'] = model_type
        elif category in ["position", "waypoint"]:
            # 位置推論モデルの場合、model_nameパラメータを設定
            kwargs['model_name'] = model_type

        # PyTorchベンチマーク
        pytorch_time = benchmark_model(
            model_path, category, "pytorch", num_iterations=args.iterations, **kwargs
        )

        # TensorRTベンチマーク
        tensorrt_time = benchmark_model(
            model_path, category, "tensorrt", num_iterations=args.iterations, **kwargs
        )

        if pytorch_time and tensorrt_time:
            all_results[model_path] = {
                'pytorch': pytorch_time,
                'tensorrt': tensorrt_time,
                'speedup': pytorch_time / tensorrt_time,
                'model_type': model_type,
                'category': category
            }

    # 結果サマリー
    if all_results:
        print("\n\n" + "="*60)
        print("ベンチマーク結果サマリー")
        print("="*60)

        for model_path, result in all_results.items():
            print(f"\n{os.path.basename(model_path)}:")
            print(f"  モデルタイプ: {result['model_type']} ({result['category']})")
            print(f"  PyTorch:      {result['pytorch']:.2f} ms  ({1000.0/result['pytorch']:.1f} FPS)")
            print(f"  TensorRT:     {result['tensorrt']:.2f} ms  ({1000.0/result['tensorrt']:.1f} FPS)")
            print(f"  高速化率:     {result['speedup']:.2f}x")

        print("\n" + "="*60)
        print("🎉 ベンチマーク完了")
        print("="*60)
    else:
        print("\n❌ ベンチマーク結果がありません")


if __name__ == "__main__":
    main()
