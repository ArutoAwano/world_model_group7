# LPM Exploration Experiments

以下の2つのノートブックは、LPM (Learning Progress Motivation) 手法の検証と性能比較のために使用されます。

## 1. `mnist_curiosity.ipynb`
**目的:** MNISTデータセットを用いて、ノイズ（Noisy TV問題）に対する好奇心に基づく探索手法の挙動を検証・比較するための実験ノートブックです。

**詳細:**
*   **比較対象:**
    *   **MSE Baseline:** 予測誤差に基づく単純なベースライン。
    *   **AMA (Adversarial Motivation Architecture):** 敵対的な動機づけを用いる手法。
    *   **Uncertainty Predictor:** 不確実性を予測して報酬とする手法。
*   **処理:** MNISTデータを読み込み、各手法でトレーニング（約600イテレーション）を行い、5つの異なるシードで安定性を検証します。

## 2. `plot_atari.ipynb`
**目的:** Atariゲーム環境（SpaceInvaders, MsPacman）における学習曲線をプロットし、LPMと他の多数のベースライン手法を比較するための可視化ツールです。

**詳細:**
*   **データ集計:** `monitor.csv` ファイルから累積報酬を集計。
*   **比較対象:**
    *   **LPM (LP):** 提案手法 (`LPM(ours)` として表示)。
    *   **既存手法:** AMA, MSE, RND, IDF, Ensemble, EDT, EME。
    *   **Random:** ランダムな行動。
*   **プロット:** 以下の4環境での性能を比較プロットし、`comparison_plots` に保存します。
    1.  Space Invader
    2.  Space Invader (Noiseあり)
    3.  Ms PacMan
    4.  Ms PacMan (Noiseあり)

## 3. Craftium 環境での実験

`train_lpm.py` を使用して、Craftium（Minetestベースの3D環境）で LPM エージェントを学習させることができます。

### 実行方法

Docker 環境内で以下のコマンドを実行してください。

```bash
docker compose exec lpm-dreamer xvfb-run -a python3 train_lpm.py --env-name <環境名> --steps 50000 --wandb
```

### 利用可能な環境 (タスク)

*   `Craftium/OpenWorld-v0`: デフォルト。広大な世界を自由に探索します。
*   `Craftium/ChopTree-v0`: 木を切るタスク。斧を持った状態で開始します。
*   `Craftium/Speleo-v0`: 洞窟探検タスク。
*   `Craftium/SpidersAttack-v0`: クモと戦うタスク。剣を持った状態で開始します。
*   `Craftium/ProcDungeons-v0`: ダンジョン攻略タスク。剣を持った状態で開始します。

**例: 木を切るタスクの実行**

```bash
docker compose exec lpm-dreamer xvfb-run -a python3 train_lpm.py --env-name Craftium/ChopTree-v0 --steps 50000 --wandb

## 4. Atari & MNIST での実験 (新規追加)

`train_lpm.py` が拡張され、Atari と MNIST 環境でも動作するようになりました。

### 必要なライブラリのインストール (Atari用)

Atari環境を実行するには、追加のライブラリが必要です。

```bash
pip install "gymnasium[atari]" "gymnasium[accept-rom-license]" ale-py shimmy
```

### MNIST 環境の実行

MNISTデータセットを環境として使用し、ランダムな画像遷移に対する好奇心を検証します。

```bash
python3 LPM_exploration/train_lpm.py --env-name MNIST --steps 10000
```

*   **Noisy TV (ノイズあり) の場合:** `--noisy-tv` を付けます（確率10%でランダムなCIFAR画像が表示されます）。
    ```bash
    python3 LPM_exploration/train_lpm.py --env-name MNIST --steps 10000 --noisy-tv
    ```

### Atari 環境の実行

例: Breakout (ブロック崩し)

```bash
python3 LPM_exploration/train_lpm.py --env-name ALE/Breakout-v5 --steps 50000
```

*   **注意:** `SB3_AVAILABLE` (stable-baselines3) がインストールされている場合、自動的に推奨されるAtariラッパー（NoopReset, MaxAndSkipなど）が適用されます。

## 5. Atari 実装の改善と可視化 (Update: 2026/01/06)

Atari環境 (Breakout等) での LPM の動作検証および可視化機能を強化しました。

### 実行方法

Docker環境内で以下のコマンドを実行します。

```bash
# Breakout (ブロック崩し) の場合
docker run --gpus all -it --rm -v $(pwd):/workspace worldmodel-competition3 \
  python train_lpm.py --env-name ALE/Breakout-v5 --steps 50000 --wandb
```

### 変更点と機能

#### 1. 依存関係の更新 (Dockerfile)
*   `gymnasium[atari]`, `gymnasium[accept-rom-license]`, `ale-py` を追加。
*   `gymnasium>=1.0.0` を強制インストールし、`AutoresetMode` のインポートエラーを解消。

#### 2. 環境構築の改善 (train_lpm.py)
*   **Atari Wrapper:** `AtariPreprocessing` を導入し、自動的に以下の処理を行うように変更しました。
    *   NoopReset (開始時のランダム待機)
    *   FrameSkip (4フレームスキップ)
    *   Resize (64x64 にリサイズし、Dreamerの入力形式に適合)
    *   **注意:** `RenderWrapper` は Atari 環境ではスキップされ、`AtariPreprocessing` の出力が直接使用されます。

#### 3. 可視化機能の強化 (eval_view)
学習中の評価（Evaluation）時に、**「実際の環境 (Real)」** と **「世界モデルの夢 (Reconstructed)」** を比較する動画を生成します。

*   **保存場所:** `/workspace/eval_view/video/` および `/workspace/eval_view/images/`
*   **内容:** 左側が実際のゲーム画面、右側がRSSMが内部状態から再構成した予測画像です。
*   **目的:** 世界モデルが正しく環境を認識・予測できているかを目視確認するため。

### 改善・設定変更のポイント

今後、パラメータ調整や環境変更を行う場合の主な修正箇所です。

1.  **環境の変更:**
    *   `train_lpm.py` の `make_env_simple` 関数内。
    *   `ALE/Breakout-v5` 以外のアタリゲームを使う場合は、`--env-name ALE/Pong-v5` のように引数で指定可能ですが、`shimmy` や `ale-py` のサポート状況に依存します。

2.  **LPMパラメータの調整:**
    *   `train_lpm.py` の `Config` クラス内。
    *   `self.lpm_eta`: 予測誤差に対する報酬のスケール (デフォルト 1.0)。
    *   `self.lpm_lr`: ErrorPredictorの学習率。

3.  **可視化の頻度:**
    *   `train_lpm.py` の `Config` クラス内。
    *   `self.eval_interval`: 評価を実行する間隔 (ステップ数)。デフォルトは 800 ですが、頻繁に見たい場合は短くしてください。

4.  **モデル構造:**
    *   `student_code.py` 内の `RSSM` クラス。
    *   現在 `transition_hidden` は `state_dim * num_classes + action_dim` を入力としていますが、アクションのエンコーディング方法（One-hot vs Scalar）を変更する場合はここを確認してください。

## 6. Atari ベンチマークについて (解説)

なぜ Craftium ではなく Atari (Breakout) なのか？ その背景と重要性について。

### 強化学習における \"Atari\" とは
Atari 2600 (Arcade Learning Environment: ALE) は、AI研究における**「共通一次試験」**のような存在です。
2013年にDeepMindが「DQN」で人間超えのスコアを叩き出して以来、新しいAI手法の性能を測る絶対的な物差しとなっています。

### Craftium との比較 (LPM検証の観点)

| 特徴 | Atari (今回) | Craftium (前回) |
| :--- | :--- | :--- |
| **画面情報** | **シンプルで抽象的** | **複雑でノイズが多い** |
| **動き** | 動き=ゲームの進行 (ボールの変化など) | 動き≠意味 (草の揺れ、影のチラつき) |
| **予測難易度** | ルール学習が必要 | 画素の変化に惑わされやすい |
| **LPMへの影響** | **「学習の進捗」を正しく測れる** | **「予測不能なノイズ」を面白がってしまう** |

今回は、シンプルでノイズの少ない Atari 環境に戻ることで、**「LPMが正しく機能する (予測誤差の減少を報酬として感じる)」** ことを確認します。これが成功すれば、Craftiumでの失敗が「アルゴリズムではなく環境の複雑さ(ノイズ)が原因だった」と証明できます。

## 7. 利用可能な Atari 環境ID 一覧

本実装では、`--env-name` 引数を変更することで以下の様々なゲームを実行可能です。
※ `gymnasium[atari]` に含まれる標準的な環境です。

### 推奨環境 (検証済み・動作確認容易)

*   **Breakout (ブロック崩し)**
    *   ID: `ALE/Breakout-v5`
    *   特徴: シンプルな因果関係。LPMの基本動作確認に最適。

*   **Pong (ポン)**
    *   ID: `ALE/Pong-v5`
    *   特徴: 敵AIとの対戦。報酬が明確で学習が早い。Breakoutの次に試すのに推奨。

### 応用環境 (難易度高)

*   **Space Invaders (インベーダー)**
    *   ID: `ALE/SpaceInvaders-v5`
    *   特徴: 多数の敵、遮蔽物。世界モデルの予測能力が試される。

*   **Seaquest (海底大戦争)**
    *   ID: `ALE/Seaquest-v5`
    *   特徴: 定期的な酸素補給が必要。少し長期的な計画性が求められる。

*   **Montezuma's Revenge (モンテズマの復讐)**
    *   ID: `ALE/MontezumaRevenge-v5`
    *   特徴: **好奇心系AIの最終ボス**。
    *   即死トラップが多く、通常の報酬(Extrinsic Reward)だけでは0点しか取れない。
    *   「新しい部屋に行けた」という**内発的報酬(LPM)**だけが頼りの最難関タスク。

### その他の環境について
基本的には `ALE/<GameName>-v5` の形式で指定可能です（例: `ALE/MsPacman-v5`, `ALE/Qbert-v5`）。
動作しない場合は、以下の古い形式も試行可能です（ただし画像処理等のWrapper調整が必要になる場合があります）：
*   `<GameName>NoFrameskip-v4` (例: `BreakoutNoFrameskip-v4`)
