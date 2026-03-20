#!/usr/bin/env python3
# coding:utf-8
"""
PWMコントローラークラス
プロポからのPWM信号をI2C経由で読み取り、joystickと同じインターフェースで操作値を提供

使用方法:
    from pwm_controller import PWMController
    controller = PWMController()
    controller.poll()  # PWM信号を読み取って更新
    steering = controller.steering  # -1.0 ~ 1.0
    throttle = controller.throttle  # -1.0 ~ 1.0
"""

import config
# Platform-aware import: smbus2 for Raspberry Pi, smbus for Jetson
try:
    import smbus2 as smbus
except ImportError:
    import smbus
import time
import numpy as np


class PWMController:
    """
    PWM信号を読み取ってジョイスティックと同じインターフェースで操作値を提供するクラス
    """

    def __init__(self):
        """初期化"""
        self.HAVE_CONTROLLER = False

        # joystickと同じインターフェースの属性
        self.steering = 0.0
        self.throttle = 0.0
        self.mode = ["user", "auto_str", "auto"]
        self.recording = False
        self.is_braking = False

        # PWM設定
        self.i2c_bus = config.PWM_I2C_BUS
        self.i2c_addr = config.PWM_I2C_ADDRESS
        self.i2c = None

        # キャリブレーション値を読み込み
        self.ch1_left = config.PWM_CH1_LEFT_RAW
        self.ch1_center = config.PWM_CH1_CENTER_RAW
        self.ch1_right = config.PWM_CH1_RIGHT_RAW

        self.ch2_forward = config.PWM_CH2_FORWARD_RAW
        self.ch2_neutral = config.PWM_CH2_NEUTRAL_RAW
        self.ch2_reverse = config.PWM_CH2_REVERSE_RAW

        # デッドゾーン設定（ニュートラル付近の遊び）
        self.deadzone = 0.05  # ±5%

        # 前回の値（スムージング用）
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.smoothing = 0.3  # 0.0=即座に変化、1.0=変化なし

        # I2C接続試行
        try:
            self.i2c = smbus.SMBus(self.i2c_bus)
            # テスト読み取り
            self.i2c.read_i2c_block_data(self.i2c_addr, 0x01, 12)
            self.HAVE_CONTROLLER = True
            print(f"PWMコントローラー接続成功 (I2C: bus={self.i2c_bus}, addr=0x{self.i2c_addr:02X})")
            print(f"キャリブレーション値:")
            print(f"  CH1 (ステアリング): LEFT={self.ch1_left}, CENTER={self.ch1_center}, RIGHT={self.ch1_right}")
            print(f"  CH2 (スロットル):   FORWARD={self.ch2_forward}, NEUTRAL={self.ch2_neutral}, REVERSE={self.ch2_reverse}")
        except Exception as e:
            self.HAVE_CONTROLLER = False
            print(f"PWMコントローラー接続失敗: {e}")
            print("キーボード操作に切り替えます")

    def read_raw_values(self):
        """
        PWM生値を読み取る

        Returns:
            tuple: (raw1, raw2) - CH1とCH2の生値、エラー時は(None, None)
        """
        if not self.HAVE_CONTROLLER:
            return None, None

        try:
            # 12バイトのデータブロックを読み取る
            data = self.i2c.read_i2c_block_data(self.i2c_addr, 0x01, 12)

            # 4バイトずつ32ビット値に変換
            raw1 = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3]
            raw2 = data[4] << 24 | data[5] << 16 | data[6] << 8 | data[7]

            return raw1, raw2
        except Exception as e:
            print(f"PWM読み取りエラー: {e}")
            return None, None

    def raw_to_normalized(self, raw_value, min_val, center_val, max_val):
        """
        RAW値を-1.0~1.0の範囲に正規化

        Args:
            raw_value: 生のPWM値
            min_val: 最小値（左または前進）
            center_val: 中央値（ニュートラル）
            max_val: 最大値（右または後退）

        Returns:
            float: -1.0 ~ 1.0の正規化された値
        """
        if raw_value < center_val:
            # 中央より小さい側（左/前進）
            if min_val == center_val:
                return 0.0
            normalized = (raw_value - center_val) / (center_val - min_val)
        else:
            # 中央より大きい側（右/後退）
            if max_val == center_val:
                return 0.0
            normalized = (raw_value - center_val) / (max_val - center_val)

        # -1.0 ~ 1.0 にクリップ
        normalized = max(-1.0, min(1.0, normalized))

        # デッドゾーン適用
        if abs(normalized) < self.deadzone:
            normalized = 0.0

        return normalized

    def apply_smoothing(self, new_value, prev_value):
        """
        値のスムージング（急激な変化を抑える）

        Args:
            new_value: 新しい値
            prev_value: 前回の値

        Returns:
            float: スムージング適用後の値
        """
        return prev_value * self.smoothing + new_value * (1 - self.smoothing)

    def poll(self):
        """
        PWM信号を読み取って操作値を更新
        joystick.poll()と同じインターフェース
        """
        if not self.HAVE_CONTROLLER:
            return

        # PWM生値を読み取り
        raw1, raw2 = self.read_raw_values()

        if raw1 is None or raw2 is None:
            return

        # 正規化
        raw_steering = self.raw_to_normalized(raw1, self.ch1_left, self.ch1_center, self.ch1_right)
        raw_throttle = self.raw_to_normalized(raw2, self.ch2_forward, self.ch2_neutral, self.ch2_reverse)

        # スムージング適用
        self.steering = self.apply_smoothing(raw_steering, self.prev_steering)
        self.throttle = self.apply_smoothing(raw_throttle, self.prev_throttle)

        # 前回値を更新
        self.prev_steering = self.steering
        self.prev_throttle = self.throttle

        # ブレーキ判定（スロットルが後退側に大きく倒れている場合）
        self.is_braking = self.throttle < -0.8

    def calibrate(self, interval=0.05):
        """
        PWM信号のキャリブレーションモード
        プロポを操作して最大値・最小値を記録

        Args:
            interval: 読み取り間隔（秒）
        """
        if not self.HAVE_CONTROLLER:
            print("エラー: コントローラーが接続されていません")
            return

        print("\n" + "=" * 80)
        print("PWMキャリブレーションモード")
        print("=" * 80)
        print("\n指示に従ってプロポを操作してください：")
        print("  1. ステアリングを左右に最大まで動かす")
        print("  2. スロットルを前進・後退に最大まで動かす")
        print("  3. 各位置で1〜2秒間保持してください")
        print("\nCtrl+Cで終了すると、測定した最大値・最小値が表示されます")
        print("-" * 80)

        # 最大値・最小値の初期化
        ch1_min = float('inf')
        ch1_max = float('-inf')
        ch1_center = None
        ch2_min = float('inf')
        ch2_max = float('-inf')
        ch2_neutral = None

        # 初期値取得用のカウンター
        stable_count = 0
        required_stable = 10  # 10回連続で安定した値を取得

        print(f"\n{'時刻':<12} {'CH1(RAW)':>12} {'CH2(RAW)':>12} {'状態':<30}")
        print("-" * 80)

        try:
            while True:
                raw1, raw2 = self.read_raw_values()

                if raw1 is not None and raw2 is not None:
                    # 最大値・最小値の更新
                    ch1_min = min(ch1_min, raw1)
                    ch1_max = max(ch1_max, raw1)
                    ch2_min = min(ch2_min, raw2)
                    ch2_max = max(ch2_max, raw2)

                    # 中央値・ニュートラル値の推定（初期10回の平均）
                    if stable_count < required_stable:
                        if ch1_center is None:
                            ch1_center = raw1
                            ch2_neutral = raw2
                        else:
                            ch1_center = (ch1_center * stable_count + raw1) / (stable_count + 1)
                            ch2_neutral = (ch2_neutral * stable_count + raw2) / (stable_count + 1)
                        stable_count += 1

                    # 状態判定
                    status = []
                    if abs(raw1 - ch1_min) < 50:
                        status.append("ステア:左MAX")
                    elif abs(raw1 - ch1_max) < 50:
                        status.append("ステア:右MAX")
                    elif ch1_center and abs(raw1 - ch1_center) < 50:
                        status.append("ステア:中央")

                    if abs(raw2 - ch2_min) < 50:
                        status.append("スロ:前進MAX")
                    elif abs(raw2 - ch2_max) < 50:
                        status.append("スロ:後退MAX")
                    elif ch2_neutral and abs(raw2 - ch2_neutral) < 50:
                        status.append("スロ:中立")

                    status_str = " / ".join(status) if status else "操作中..."

                    timestamp = time.strftime("%H:%M:%S")
                    print(f"{timestamp:<12} {raw1:>12} {raw2:>12} {status_str:<30}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("キャリブレーション結果")
            print("=" * 80)
            print("\n【CH1: ステアリング】")
            print(f"  左最大   (LEFT):   {ch1_min:>10} RAW")
            print(f"  中央     (CENTER): {int(ch1_center):>10} RAW" if ch1_center else "  中央値: 未測定")
            print(f"  右最大   (RIGHT):  {ch1_max:>10} RAW")

            print("\n【CH2: スロットル】")
            print(f"  前進最大 (FORWARD): {ch2_min:>10} RAW")
            print(f"  中立     (NEUTRAL): {int(ch2_neutral):>10} RAW" if ch2_neutral else "  中立値: 未測定")
            print(f"  後退最大 (REVERSE): {ch2_max:>10} RAW")

            print("\n" + "=" * 80)
            print("config.pyに設定する値:")
            print("=" * 80)
            print(f"PWM_CH1_LEFT_RAW = {ch1_min}")
            print(f"PWM_CH1_CENTER_RAW = {int(ch1_center) if ch1_center else 0}")
            print(f"PWM_CH1_RIGHT_RAW = {ch1_max}")
            print(f"PWM_CH2_FORWARD_RAW = {ch2_min}")
            print(f"PWM_CH2_NEUTRAL_RAW = {int(ch2_neutral) if ch2_neutral else 0}")
            print(f"PWM_CH2_REVERSE_RAW = {ch2_max}")
            print("=" * 80)

    def monitor(self, interval=0.05):
        """
        PWM信号を連続監視

        Args:
            interval: 読み取り間隔（秒）
        """
        if not self.HAVE_CONTROLLER:
            print("エラー: コントローラーが接続されていません")
            return

        print("\nPWM信号監視開始 (Ctrl+Cで終了)")
        print("-" * 80)
        print(f"{'時刻':<12} {'Steering':>10} {'Throttle':>10} {'RAW1':>12} {'RAW2':>12}")
        print("-" * 80)

        try:
            while True:
                self.poll()

                raw1, raw2 = self.read_raw_values()
                timestamp = time.strftime("%H:%M:%S")

                if raw1 is not None and raw2 is not None:
                    print(f"{timestamp:<12} {self.steering:>10.3f} {self.throttle:>10.3f} {raw1:>12} {raw2:>12}")
                else:
                    print(f"{timestamp:<12} {'---':>10} {'---':>10} {'---':>12} {'---':>12}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n監視を終了しました")

    def close(self):
        """I2C接続を閉じる"""
        if self.i2c:
            try:
                self.i2c.close()
                print("PWMコントローラー接続を閉じました")
            except:
                pass


# テスト用のメイン関数
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("PWMコントローラー")
    print("=" * 80)

    # コマンドライン引数でモード選択
    mode = "calibrate"  # デフォルトはキャリブレーションモード
    if len(sys.argv) > 1:
        if sys.argv[1] in ["monitor", "m"]:
            mode = "monitor"
        elif sys.argv[1] in ["calibrate", "c", "calib"]:
            mode = "calibrate"
        elif sys.argv[1] in ["test", "t"]:
            mode = "test"
        else:
            print(f"\n使用方法: {sys.argv[0]} [calibrate|monitor|test]")
            print("  calibrate (c): キャリブレーションモード（デフォルト）")
            print("  monitor (m):   監視モード（正規化された値＋RAW値）")
            print("  test (t):      簡易テストモード（正規化された値のみ）")
            sys.exit(1)

    controller = PWMController()

    if not controller.HAVE_CONTROLLER:
        print("エラー: PWMコントローラーに接続できませんでした")
        print("\n確認事項:")
        print("  1. I2Cデバイスが接続されているか")
        print("  2. I2Cバス番号が正しいか (i2cdetect -y -r 7)")
        print("  3. 適切な権限があるか (sudo usermod -aG i2c $USER)")
        sys.exit(1)

    try:
        if mode == "calibrate":
            controller.calibrate(interval=0.1)
        elif mode == "monitor":
            controller.monitor(interval=0.1)
        elif mode == "test":
            # 簡易テストモード（正規化された値のみ表示）
            print("\nPWMコントローラー簡易テスト")
            print("\nプロポを操作してください（Ctrl+Cで終了）")
            print("-" * 80)
            print(f"{'時刻':<12} {'Steering':>10} {'Throttle':>10} {'Braking':<8}")
            print("-" * 80)

            while True:
                controller.poll()

                timestamp = time.strftime("%H:%M:%S")
                braking_str = "BRAKE" if controller.is_braking else ""

                print(f"{timestamp:<12} {controller.steering:>10.3f} {controller.throttle:>10.3f} {braking_str:<8}")

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n終了")
    finally:
        controller.close()
