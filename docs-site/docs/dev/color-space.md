# 画像処理における色空間の注意事項

本プロジェクトでは複数のライブラリを使用して画像を扱うため、色空間（RGB/BGR）の違いによる問題が発生する可能性があります。以下の統一ルールに従って開発してください。

## 各ライブラリの色空間

| ライブラリ | 読み込み関数 | 色空間 | 備考 |
|-----------|-------------|--------|------|
| **OpenCV** | `cv2.imread()` | **BGR** | 一般的なRGBとは逆 |
| **OpenCV** | `cv2.VideoCapture.read()` | **BGR** | カメラからの取得もBGR |
| **PIL/Pillow** | `Image.open()` | **RGB** | 標準的なRGB順序 |
| **matplotlib** | `plt.imread()` | **RGB** | 表示用途 |
| **PyTorch/TensorFlow** | - | **RGB** | 学習フレームワークはRGB前提 |

## 統一ルール：すべてRGB形式で処理

```python
# ❌ 間違った例：OpenCVで読んでそのまま使用
image_bgr = cv2.imread('image.jpg')  # BGR形式
model.run(image_bgr)  # BGRをRGBと誤認識 → 色がおかしくなる

# ✅ 正しい例：BGR→RGB変換
image_bgr = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
model.run(image_rgb)  # RGB形式で処理

# ✅ 正しい例：PILで読み込み（変換不要）
from PIL import Image
image_rgb = Image.open('image.jpg').convert('RGB')
model.run(np.array(image_rgb))  # 既にRGB形式
```

## 本プロジェクトでの実装

### 1. カメラ取得 (`camera.py`)

```python
# Jetsonカメラ：BGR→RGB変換を実装済み
frame_bgr = cap.read()
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
return ret, frame_rgb
```

### 2. 学習データ読み込み (`train_pytorch.py`)

```python
# PILでRGB読み込み
image = Image.open(img_path).convert("RGB")
```

### 3. 推論時 (`planner.py`)

```python
# camera.pyからRGB形式で受け取り、そのまま使用
image_rgb = camera.get_frame()
prediction = model(image_rgb)
```

## よくある問題

### 問題1: 学習と推論で色が違う

**症状:** 学習時は正常だが、推論時に色がおかしい

**原因:** 学習データはPIL(RGB)、推論時はOpenCV(BGR)で読み込んでいる

**解決:**
```python
# 推論時もRGBに変換
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

### 問題2: 画像保存時に色が変わる

**症状:** OpenCVで保存した画像の色がおかしい

**原因:** RGB画像をそのままcv2.imwrite()で保存

**解決:**
```python
# RGB→BGRに戻してから保存
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.jpg', image_bgr)
```

### 問題3: matplotlibで表示がおかしい

**症状:** OpenCVで読んだ画像をmatplotlibで表示すると色がおかしい

**原因:** BGR画像をRGBとして表示

**解決:**
```python
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
```

## チェックリスト

開発時は以下を確認してください：

- [ ] カメラ取得後にRGB変換しているか
- [ ] 学習データの読み込みはPILまたはRGB変換済みか
- [ ] 推論時の入力形式は学習時と同じか
- [ ] 画像保存時にBGR変換しているか（OpenCV使用時）
- [ ] 表示時の色空間は正しいか
