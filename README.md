# 目的
東京の観光地の画像を入力し、その観光地の名前を出力する分類器を作る。
今回は、浅草・明治神宮・スカイツリー・東京タワーの4箇所の画像を学習し分類できるようにした。

# 実行確認環境
  Python 3.5.2

# データセットのディレクトリ構造 (train, test同様) 
* \<dataset-dir\>
  * asakusa
    * \<subdir-1\>
      * a.jpg
      * b.jpg
      * c.jpg
      * ...
    * \<subdir-2\>
      * a2.jpg
      * ...
    * ...
  * meiji
    * ...
  * sky_tree
    * ...
  * tokyo_tower
    * ...

# プログラムたち
* 前処理のプログラム
  * image_augmentor/main.py (外部プログラム)    
    * https://github.com/codebox/image_augmentor よりダウンロード
    * 画像の反転・回転・ノイズ付与などが可能。加工した画像を、元の画像が保存されているディレクトリと同じディレクトリに保存している。
    * 実行    
      $ python image_augmentor/main.py \<dataset-dir\> [\<command-1\> \<command-2\> ...]
  * data_augment.py
    * 画像の反転・random cropなどが可能。出力は224×224の画像
    * 実行    
      $ python data_augment.py -d \<dataset-dir\> -o \<output-dir\> -C {True, False} -c \<random-crop-times\> -H {True, False} -R {True, False}
      * -d, --dataset : データセットディレクトリ (デフォルト : 'image/test')
      * -o, --output : 出力先ディレクトリ (デフォルト : image_train_aug_cv)
      * -C, --docrop : ランダムにクロップした画像を出力するかどうか。許容入力：{True, False}　（デフォルト : True）
      * -c, --randomcroptimes : ランダムにクロップする回数（-C: Trueの時）。許容入力：{True, False} （デフォルト : 10）
      * -H, --dofliph : 左右反転した画像を出力するかどうか。許容入力：{True, False}　（デフォルト: False）
      * -C, --docrop : ランダムにクロップした画像を出力するかどうか。許容入力：{True, False}
      * -R, --doresize : 元画像を224×224にリサイズしただけの画像を出力するかどうか。許容入力:{True, False}（デフォルト: True）
    * 入力データセットのスタイル
      * 画像のみを含む任意のディレクトリ（画像データはディレクトリ2層目または2層目より深く存在することを想定）
    * 出力データセットのスタイル
      * 元の画像のファイル名、直上ディレクトリのディレクトリ名に、行った処理の識別末尾をつけたファイルを、出力ディレクトリ直下のディレクトリに保存
      * 末尾識別子の種類
        * 左右反転: _fh224
        * クロッピング: _cr224
        * リサイズ: _rs224     
        (例): 左右反転した画像の場合:     
        \<dataset-dir\>/subdir1/subdir11/image1.jpg → \<output-dir\>/subdir1/subdir11_fh224/image1_fh224.jpg

# trainプログラム
* train_sight_new.py
* 画像を入力して学習するプログラム。エポックごとに学習したモデル・スナップショットと、各エポックごとの学習データ・バリデーションデータのaccuracy, lossを指定した出力ディレクトリに出力している。
* 実行    
  $ python train_sight_new.py -e \<epoch-num\> -g \<gpu-id\> -o \<result-dir\> -d [\<dataset-dir1\> \<dataset-dir2\> ...] -t \<train-data-num\> -b \<batchsize\> -r \<resume-snapshot\>
  * -e, --epoch: エポック数
  * -g, --gpu: GPU ID (default: -1(GPU不使用))
  * -o, --output: 学習結果出力ディレクトリ
  * -o, --out: 学習中のモデル、スナップショットの保存ディレクトリ
  * -d, --dataset: 学習データセットディレクトリ。複数指定可。階層構造は上記のデータセットのディレクトリ構造に準じる。
  * -t, --traindatanum: trainデータとして用いる数。（入力データセットのうち、trainデータ以外はvalidationデータとして使われる。）
  * -b, --batchsize: バッチサイズ　(default: 100)
  * -r, --resume: 途中から再開するスナップショット。

# testプログラム
* test_sight_new_one.py
  * モデルを指定し、ディレクトリ内のすべての画像に対して分類結果と最後の全結合層のソフトマックス関数の出力値を表示する。また、そのモデルのテストデータに対する全体・クラスごとのaccuracyも求めて表示している。
* 実行
  $ python test_sight_new_one.py -d \<dataset-dir\> -m \<model-path\> -b \<batchsize\> -g \<gpu-id\>
  * -d, --dataset: テストデータセットディレクトリ。複数指定可。階層構造は上記のデータセットのディレクトリ構造に準じる。
  * -m, --model: 学習済みモデルのパス
  * -b, --batchsize: バッチサイズ (default: 1)
  * -g, --gpu: GPU ID (default: -1)

* test_sight_new_all.py
  * ある決まったテストデータセットに対してエポックごとのモデルのaccuracyを求めている。
  * accuracyの計算ではテストデータ全体と各クラスごとのaccuracyを求めていて、その値をターミナルに表示するとともにそのモデルが保管されているディレクトリにcsvファイル(test_accuracy.csv)として保存している。
  * また、各テスト画像のモデル（エポック）ごとの判別結果をまとめたcsvファイル(test_predict.csv)を出力している。
* 実行
  $ python test_sight_new_all.py -d \<dataset-dir\> -m \<models-dir\> -b \<batchsize\> -g \<gpu-id\> 
  * -d, --dataset: テストデータセットディレクトリ。複数指定可。階層構造は上記のデータセットのディレクトリ構造に準じる。
  * -m, --modeldir: 学習済みモデルが保管されているディレクトリ。（モデルはディレクトリ直下にあることを想定。）また、各種分析結果はこのディレクトリに保存される。
  * -b, --batchsize: バッチサイズ (default: 1)
  * -g, --gpu: GPU ID (default: 1)
  * -f, --firstmodel: テスト開始エポック (default: None(最初から実行))
  * -l, --lastmodel: テスト終了エポック (default: None(最後まで実行))

# その他
* net.py
  * 学習ネットワークが記されているファイル。（今回はResNet50ToNClassを使用）
* dataset.py
  * データセット取得に使われるファイル。（今回はMyImageNewDatasetを使用）
* show_imagepath.py
  * ディレクトリ内のファイルがすべて画像ファイルかチェックし、各画像のパスをすべて表示する・
  * 実行     
    $ python -d <dataset-dir>
    * -d, --dataset: 画像が入っているディレクトリ