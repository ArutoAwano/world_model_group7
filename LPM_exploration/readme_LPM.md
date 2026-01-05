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
