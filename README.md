# mediapipe test

## Overview
急に使ってみたくなったので，MediaPipeをローカルで動かしてみた

ソースコードは[docs](https://google.github.io/mediapipe/)のコピペ

一応自分なりにコメントアウト書きながら処理を追ってみた

### 動かしてみた感想：
opencvだけで同じことも出来そうだけど，cascade.xmlを目的ごとに用意しないといけない．mediapipeを使えば，モジュールを呼び出すだけなので，手軽で良い．


## Usage
```
$ git clone ...
$ cd ...
$ python -m venv venv
$ source venv/Scripts/activate # 環境による
$ pip install -U pip
$ pip install -r requirements.txt
$ python main.py --mode <mode name>
$ python main.py --help
```

## References
- https://google.github.io/mediapipe/getting_started/python.html
- https://google.github.io/mediapipe/solutions/solutions.html
