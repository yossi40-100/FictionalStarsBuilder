# Fictional Stars Builder - for UE5 Celestial Vault Plugin
#  架空の星座ビルダー　yossi40-100 改造版

使い方： https://yossi40-100.com/ue5star_builder/


下記で公開されている作業のうちUE取り込み前の3パートを、ワンツール化したものです。
https://dev.epicgames.com/community/learning/tutorials/9XyB/unreal-engine-creating-custom-constellations-for-the-celestial-vault-plugin

行程１の自動化は単純です。白地に黒などの単純な画像を想定しています。  
それ以外の画像に柔軟に対応したい場合は `image2starpoint.py` を自分で改造してください。


複数の星座を合体させる機能はありません。2つ目以降のCSVからヘッダーを消して合体させてください。


# インストール

1. Pythonのインストール
Pythonをインストールします See https://www.python.org/downloads/ (PATHを通します)

その際、Tkinterオプションを有効にしてください

2. このリポジトリをクローンします

3. cmd.exeなどの端末でツール動作に必要なライブラリをインストールします

```
>> pip install opencv-python matplotlib numpy pillow
```

# 使い方

1. GUIを起動します

```
>> python ./FictionalStarsBuilder.py
```

2. 右上の “Open Picture” ボタンから、以前に作成したファイルを開きます

3. 左のメニューにて星の抽出設定を調整します

4. 表示モードをfinalImageに変更します

5. 左のメニュー下部にて星座の天球上の配置を調整します

 - Center RA/Center DEC 星座の中心位置
 - Size RA/Size DEC 星座の大きさ
 - Rotation Angle 星座の角度

6. 右上の “Export CSV” ボタンから、計算結果を出力します

7. UnrealEngineに「StarInputData」型のDataTableアセットとして取り込みます
  オプションのチェックはすべてつけて、KeyFieldは「ID」とします

8. CelestialVaultDaySequenceアクターの詳細欄でFictional Star Catalogに登録します


## 表示モード

- original オリジナル - 入力画像を表示します
- Dot ドット - 入力画像からエッジ抽出された画像を表示します
- greyscale グレースケール - グレースケールされたDot画像を表示します
- Threshold しきい値 - Min Threshold操作後の画像を表示します
- finalImage 最終系 - 最終的に出力される天球上に配置される星の位置を表示します


# 改造
ご自由に改変してください。

元画像から特徴量の抽出は `image2starpoint.py` に分離してあります。
星の色は紫系でサイズごとに固定されています。
もとの手順のようなグラデーションマスクにはなっていません。


# ライセンス
MIT
サポートなしで現状のまま提供されます。