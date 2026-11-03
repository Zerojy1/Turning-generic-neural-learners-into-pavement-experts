# Turning generic neural learners into pavement experts

EKI predicts rutting depth from the latest inspection, pavement layer and material properties, and daily load and temperature. The same interface is fitted independently with MLP, GRU, LSTM and Transformer backbones.

$$\widehat y_t = y_{t-1} + b_\theta(c_t) + \sum_j \Delta N_j\,a_\theta(c_t)\,h_\theta(T_j).$$

With history, structure and inspection interval fixed, increasing current load or temperature cannot lower the EKI prediction.

| One-step evaluation | RIOHTrack | LTPP (locally fitted) |
|---|---|---|
| Test set | 2,089 targets, 15 held-out structures | 63 targets, 26 sections in 7 held-out states |
| EKI vs direct prediction (A0) | 49.49–57.31% lower RMSE | 29.0–39.1% lower RMSE |
| EKI vs observation residual (A2) | 4.03–14.36% lower RMSE | 0.0007–0.0166 mm higher RMSE |
| EKI decreases in fixed-context perturbation checks | 0/125,340 individual-seed checks | 0/756 ensemble checks (A2: 144/756) |

On RIOHTrack, EKI had higher RMSE than A2 after a missed inspection. Cross-backbone noninferiority to the unconstrained counterpart was not established. In a separate transfer test, all four frozen RIOHTrack EKI models had higher RMSE than persistence on LTPP.

![RIOHTrack results for direct prediction, observation residuals and EKI](supplements/direct-depth/figures/01_three_level_comparison.png)

[Method and protocol](PROTOCOL.md) · [RIOHTrack results](RESULTS.md) · [LTPP results and data](supplements/ltpp/) · [Reproduction and downloads](REPRODUCTION.md)

