
# Walkthrough - LPMスイッチング戦略の改善

- [x] LPMスイッチング戦略の議論と策定 (First-Success Triggered Decay -> Reliability-Gated)
- [x] `train/train_lpm_modified.py` への実装
    - $R^2$スコアによる信頼度ゲーティングの実装
    - 直近平均報酬によるEta減衰/回復ロジックの実装
    - コンフィグパラメータの整理と誤字修正

## 実装された戦略: Reliability-Gated & Reward-Feedback

### 1. Reliability-Gated (信頼度ゲーティング) - "賢くなるまで待つ"
ErrorPredictorが未熟なうちは内発的報酬を与えず、予測精度($R^2$)の向上に伴って自動的に探索フェーズへ移行します。

```python
# R2 (信頼度) 計算
ep_variance = torch.var(eps_batch)
ep_r2 = 1.0 - ep_loss_D / ep_variance
ep_reliability = max(0.0, ep_r2) # 負の値は0(信頼なし)とする

# 報酬への適用
eta_current = eta_base * ep_reliability
```

### 2. Reward-Feedback (報酬フィードバック) - "見つけたら集中、見失ったら探索"
直近の成績をもとに、探索の強さ（`eta_base`）を動的に調整します。

```python
# 直近10エピソードの平均報酬
avg_recent_reward = np.mean(recent_ext_rewards)

if avg_recent_reward > 0:
    # 成功中: 集中モード (減衰)
    eta_base = eta_base * 0.995 + 0.1 * (1 - 0.995)
else:
    # 失敗中: 探索モード (回復)
    eta_base = 1.0
```

## 次のステップ
- 実機（Colab等）でのトレーニング実行と、WandBログによる `LPM/ep_reliability` および `LPM/eta_base` の挙動確認。
