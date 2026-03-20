# モーター確認

モーター（ステアリング・スロットル）の動作確認を行います。

## モーターの値を確認する

信号値の数値を入れてEnterを押す：

```bash
python motor.py
```

---

## ステアリングのPWM値を探す

1. **真ん中** - ステアリングがまっすぐになる値
2. **左最大** - 左いっぱいに切った値
3. **右最大** - 右いっぱいに切った値

## スロットルのPWM値を探す

1. **ニュートラル** - モータードライバーがピッピッピとなる値
2. **前進の最大値** - 音が変わらなくなるところ
3. **後進の最大値** - 音が変わらなくなるところ

!!! warning "注意"
    極端に大きな/小さな値を入れるとモーターを破損する恐れがあるため注意！

!!! warning "RC用ESC（モータードライバー）の仕様"
    - 多くのRC用ESCは、前進状態から直接バックできません
    - バックするには、一度ニュートラルに戻してから後進のPWM値を入力
    - ESCの取扱説明書を確認してください

---

## config.pyに値を保存する

調整した値を`config.py`に保存します：

```python
## ステアリングのPWM値
STEERING_CENTER_PWM = 370
STEERING_WIDTH_PWM = 80
STEERING_RIGHT_PWM = STEERING_CENTER_PWM + STEERING_WIDTH_PWM
STEERING_LEFT_PWM = STEERING_CENTER_PWM - STEERING_WIDTH_PWM

## スロットルのPWM値
THROTTLE_STOPPED_PWM = 370
THROTTLE_FORWARD_PWM = 500
THROTTLE_REVERSE_PWM = 300
```

---

## トラブルシューティング

### モーターが動かない
1. 電源が入っているか確認
2. PWM値が適切な範囲内か確認
3. ESCのキャリブレーションが必要な場合がある

### ステアリングが逆に動く
`STEERING_WIDTH_PWM`の符号を反転：
```python
STEERING_WIDTH_PWM = -80
```

### スロットルが反応しない
1. ESCがニュートラル位置を認識しているか確認
2. `THROTTLE_STOPPED_PWM`の値を微調整
