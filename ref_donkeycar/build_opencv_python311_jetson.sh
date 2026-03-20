#!/bin/bash
# build_opencv_jetson.sh

# === 設定 ===
OPENCV_VERSION="latest"
VENV_PATH="$HOME/env"
PYTHON_VERSION="3.11"
PYTHON_SITE_PACKAGES="$VENV_PATH/lib/python$PYTHON_VERSION/site-packages"

echo "PyPI 版 OpenCV をアンインストール..."
$VENV_PATH/bin/pip uninstall -y opencv-python opencv-python-headless || true

# === 依存ライブラリのインストール ===
echo "依存ライブラリをインストール中..."
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# === OpenCV の取得 ===
cd $HOME
if [ "$OPENCV_VERSION" = "latest" ]; then
    git clone https://github.com/opencv/opencv.git
else
    git clone -b "$OPENCV_VERSION" https://github.com/opencv/opencv.git
fi

cd opencv
mkdir -p build && cd build

# === CMake 設定 ===
echo "CMake コンフィグを生成中..."
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_GSTREAMER=ON \
      -D WITH_FFMPEG=ON \
      -D BUILD_opencv_python3=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      ..

# === ビルド・インストール ===
echo "OpenCV をビルド中..."
make -j$(nproc)
sudo make install
sudo ldconfig

# === Python バインディングを仮想環境にリンク ===
echo "Python バインディングを仮想環境にリンク..."
SO_PATH=$(find . -name "cv2.cpython-${PYTHON_VERSION/./}-*.so" | head -n 1)

if [ -n "$SO_PATH" ]; then
    ln -sf "$PWD/$SO_PATH" "$PYTHON_SITE_PACKAGES/cv2.so"
    echo "Linked: $PYTHON_SITE_PACKAGES/cv2.so"
else
    echo "エラー: cv2.so が見つかりませんでした"
fi

# === カメラデーモンの再起動 ===
echo "nvargus-daemon を再起動します..."
sudo systemctl restart nvargus-daemon
echo "完了: OpenCV ビルド & カメラ初期化 OK"
