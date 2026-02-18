# World Model Competition Group 7 - Experiment Instructions

本プロジェクトは `experiments/` フォルダ内に構成されています。
全ての実験はDockerコンテナ内で行います。

Experiment1でDreamerV3の動作確認，Experiment2でNoisyTVWrapperの動作確認を行った． \
本実験ではExperiment6の再構成誤差を内的報酬とする実験をベースラインとして，Reward modelを分離させていないExperiment3と分離させたExperiment4.5を比較する． \
他のExperimentについては最終課題に取り組む中で行った試行錯誤の名残です． \
最後にlogdirフォルダ内のメトリクスを用いて学習曲線をグラフ化し，結果を確認します． \

**共通事項:**
*   カレントディレクトリ: プロジェクトルート (`world_model_group7`)
*   GPU有効化 (`--gpus all`)
*   ログ保存先: `~/logdir` (ホスト側) -> `/logdir` (コンテナ側)
*   **WandB設定**: プロジェクトルートに `.env` ファイルを作成し、`WANDB_API_KEY=APIキー` を記述してください。

---

## 1. Experiment 1: Origin (Reference)
オリジナルのDreamerV3 (JAX) による基準性能の確認。

**設定:** `configs: atari100k`
**タスク:** `atari100k_bank_heist`
**ステップ数:** 400,000

```bash
# Build
docker build -t dreamerv3-origin -f experiments/dreamerV3_origin/Dockerfile experiments/dreamerV3_origin

# Run
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_origin:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-origin \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_origin_bankheist_400k \
    --configs atari100k \
    --task atari100k_bank_heist \
    --run.steps 400000
```

---

## 2. Experiment 2: Noisy-TV (Baseline)
行動空間依存のNoisy-TV環境下でのDreamerV3の性能確認 (LPMなし)。
`torch` 依存が含まれるため、専用のDockerイメージを使用します。

**設定:** `configs: atari100k_noisy`
**タスク:** `atari100k_bank_heist`
**ステップ数:** 400,000

```bash
# Build
docker build -t dreamerv3-noisytv -f experiments/dreamerV3_NoisyTvWrapper/Dockerfile experiments/dreamerV3_NoisyTvWrapper

# Run
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_NoisyTvWrapper:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-noisytv \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_noisytv_bankheist_400k \
    --configs atari100k_noisy \
    --task atari100k_bank_heist \
    --run.steps 400000
```

---

### 動画ログについて
`logdir/dreamerv3_.../scope/` 配下に保存される動画ファイルのファイル名（例: `00000000000000840120-...mp4`）の **長い数字部分はステップ数** （ゼロ埋め）を表しています。
例: `840120` -> 840,120 ステップ目

---

## 3. Experiment 3: LPM Integrated
内発的報酬(LPM)を外発的報酬に直接加算して統合された報酬モデルで学習する実験環境です。

**設定:** `configs: atari100k_lpm`
**タスク:** `atari100k_bank_heist`
**ステップ数:** 400,000

```bash
# Build
docker build -t dreamerv3-simple-lpm -f experiments/dreamerV3_Integrated/Dockerfile experiments/dreamerV3_Integrated

# Run
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_Integrated:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-simple-lpm \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_simple_lpm_bankheist_400k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --run.steps 400000 \
    --lpm_warmup 200000
```

---

## 4. Experiment 4: LPM 


**設定:** `configs: atari100k_lpm`
**タスク:** `atari100k_bank_heist`
**ステップ数:** 400,000

```bash
# Build
docker build -t dreamerv3-lpm -f experiments/dreamerV3_LPM/Dockerfile experiments/dreamerV3_LPM

# Run
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_LPM:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-lpm \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_lpm_bankheist_100k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --run.steps 400000 \
    --lpm_warmup 200000
```
---

<details>
<summary><strong>Tips: 行動空間の定義と確認方法 (Action Space Details)</strong></summary>

Noisy-TV環境では、通常のAtariの行動に加え、ノイズ画像を表示するための特殊な行動が追加されています。
定義は `experiments/dreamerV3_LPM/embodied/envs/atari.py` で確認できます。

*   **Standard Actions (Indices 0-17):**
    *   通常のゲーム操作 ('NOOP', 'FIRE', 'UP' ... 'DOWNLEFTFIRE')。
    *   ソースコード: `ACTION_MEANING` タプル (Line 18-21付近)。
*   **Noisy Actions (Indices 18-19):**
    *   選択すると画面がランダムなCIFAR画像に切り替わる行動。
    *   ソースコード: `__init__` メソッド内の `self.num_noisy_actions = 2` (Line 50-56付近)。
*   **Total Action Space:**
    *   合計 20アクション。
    *   ソースコード: `act_space` プロパティ (Line 95付近)。

</details>

---

## 4.5. Experiment 4.5: LPM separated
LPM の報酬モデルの分離

**設定:** `configs: atari100k_lpm`
**タスク:** `atari100k_bank_heist`
**ステップ数:** 400,000

```bash
# Build
docker build -t dreamerv3-lpm-separated -f experiments/dreamerV3_LPM_separated/Dockerfile experiments/dreamerV3_LPM_separated

# Run
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_LPM_separated:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-lpm-separated \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_lpm_separated_bankheist_400k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --run.steps 400000 \
    --lpm_warmup 200000 \
    --env.atari100k.noisy True
```

---

## 5. Experiment 5: LPM Scheduling (Proposed)

LPMのスケジューリング（Focus Phase / Discovery Phase）を実装した実験環境です。

**主な特徴:**
*   `train_lpm_modified.py` のロジックに基づく、報酬応答型の `eta_base` 減衰機能を搭載。
*   `student_code.py` と `craftium_repo` を同梱した独立環境。

**実行方法:**

### 5-1. Build
```bash
docker build -t dreamerv3-lpm-scheduling -f experiments/dreamerV3_LPM_scheduling/Dockerfile experiments/dreamerV3_LPM_scheduling
```

### 5-2. Run
```bash
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_LPM_scheduling:/workspace \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  dreamerv3-lpm-scheduling \
  python3 main.py \
    --wandb \
    --wandb-project "Dreamer-LPM-Scheduling" \
    --enable-lpm \
    --lpm-eta 1.0 \
    --lpm-focus-eta 0.1 \
    --lpm-decay-rate 0.995 \
    --steps 400000
```

**補足:**
*   ソースコードは `/workspace/main.py` (元 `train_lpm_modified.py`) として配置されます。
*   ログは wandb および `/logdir` (ホストの `~/logdir`) に保存されます。

---

---

## 6. Experiment 6: Reconstruction Error baseline
Experiment 3 (LPM) の構成をベースに、内発的報酬をLPMから再構成誤差に置き換えた実験環境です。
LPMとの比較としてのbaselineです．

**設定:** `configs: atari100k_lpm`
**タスク:** `atari100k_bank_heist`

**実行方法:**

### 6-1. Build
```bash
docker build -t dreamerv3-recon -f experiments/dreamerV3_Recon/Dockerfile experiments/dreamerV3_Recon
```

### 6-2. Run
`--use_lpm True` で内発的報酬モジュールを有効化し、`--intrinsic_type recon` でロジックを切り替えます。

```bash
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_Recon:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-recon \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_recon_single_bankheist_400k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --intrinsic_type recon \
    --use_lpm True \
    --intr_reward_scale 1.0 \
    --run.steps 400000 \
    --lpm_warmup 200000 \
    --env.atari100k.noisy True
```

---

## 7. Experiment 7: critic_separated Critic (Proposed)
外発的報酬と内発的報酬のCriticを分離し、両者のバランスを調整可能にした実験環境です。
内発的報酬の種類をLPM とReconstruction Errorから選択できます。
動作環境は `Experiment 3: LPM` と互換性があります。

**設定:** `configs: atari100k_lpm` (ベース) + 追加オプション
**タスク:** `atari100k_bank_heist` (例)

**実行方法:**

### 7-1. Build (LPM環境があればスキップ可)
```bash
docker build -t dreamerv3-critic_separated -f experiments/dreamerV3_critic_separated/Dockerfile experiments/dreamerV3_critic_separated
# または dreamerv3-lpm をそのまま利用可能
```

### 7-2. Run (Case 1: LPM Mode)
LPMによる内発的報酬を使用する場合。

```bash
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_critic_separated:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-critic_separated \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_critic_separated_lpm_bankheist_400k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --intrinsic_type lpm \
    --intr_val_scale 1.0 \
    --run.steps 400000 \
    --lpm_warmup 200000 \
    --env.atari100k.noisy True
```

### 7-3. Run (Case 2: Reconstruction Mode)
再構成誤差 (Reconstruction Error) による内発的報酬を使用する場合。
```bash
docker run -it --gpus all \
  -v $(pwd)/experiments/dreamerV3_critic_separated:/app \
  -v $(pwd)/data:/app/data \
  -v ~/logdir:/logdir \
  --env-file .env \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  dreamerv3-critic_separated \
  python3 dreamerv3/main.py \
    --logdir /logdir/dreamerv3_critic_separated_recon_bankheist_400k \
    --configs atari100k_lpm \
    --task atari100k_bank_heist \
    --intrinsic_type recon \
    --intr_val_scale 1.0 \
    --intr_reward_scale 1.0 \
    --run.steps 400000 \
    --lpm_warmup 200000 \
    --env.atari100k.noisy True
```

---

## 7. Analysis Tools (Log Visualization)
WandBを使用しない場合や、ローカルで学習推移を確認するためのスクリプトを用意しています。
`metrics.jsonl` から主要なメトリクス（Loss, Score, Value等）を抽出し、詳細なプロットを作成します。

### 単一実験のプロット
**使用方法:**
```bash
python3 plot_detailed_metrics.py --logdir <path_to_logdir>
```

**例:**
```bash
python3 plot_detailed_metrics.py --logdir ~/logdir/dreamerv3_origin_bankheist_400k
```

**出力:**
*   実行ディレクトリに `detailed_metrics.png` が生成されます。
*   Loss系（Image, Dynamics, Policy等）は対数グラフ(Log Scale)で、Scoreなどは線形グラフ(Linear Scale)でプロットされます。

### 複数実験の比較プロット
複数の実験ログを読み込み、一つのグラフにまとめて比較します。

**使用方法:**
```bash
python3 compare_experiments.py --logdirs <logdir1> <logdir2> ... --labels "<label1>" "<label2>" ...
```

**例 (Recon vs LPM 対照実験):**
```bash
python3 compare_experiments.py \
  --logdirs ~/logdir/dreamerv3_recon_single_bankheist_400k \
  ~/logdir/dreamerv3_lpm_bankheist_100k \
  ~/logdir/dreamerv3_lpm_add_Dquote_bankheist_100k \
  --labels "Recon (400k)" "LPM (100k)" "LPM Improved (100k)" \
  --output comparison_metrics.png
```

**出力ファイル:** 
分割された3つの画像ファイルが生成されます（デフォルトの場合）：
*   `comparison_metrics_1.png` (World Model & Policy)
*   `comparison_metrics_2.png` (LPM & Rewards)
**例 (LPM 各種バイエーションの比較):**
Simple LPM, NoBuffer, Recon の3つを比較する場合：
```bash
python3 compare_experiments.py \
  --logdirs ~/logdir/dreamerv3_simple_lpm_bankheist_400k \
  ~/logdir/dreamerv3_lpm_nobuffer_bankheist_400k \
  ~/logdir/dreamerv3_recon_single_bankheist_400k \
  --labels "Simple LPM" "NoBuffer" "Recon" \
  --output comparison_lpm_variants.png
```

**出力ファイル:** 
*   `comparison_lpm_variants_1.png`
*   `comparison_lpm_variants_2.png`
*   `comparison_lpm_variants_3.png`
