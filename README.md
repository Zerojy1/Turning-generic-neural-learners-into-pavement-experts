# Reusable engineering-knowledge modules for reliable neural rutting forecasting

**Pavement knowledge, shared across neural architectures.**

The engineering-knowledge interface (**EKI**) combines three reusable modules with MLP, GRU, LSTM and Transformer backbones. It gives each forecaster an explicit way to use the latest inspection, represent pavement composition and control the direction of load–temperature responses. The module equations and connections are shared across architectures; each complete model is fitted independently.

[Method](PROTOCOL.md) · [Results](RESULTS.md) · [Paper-to-code guide](MANUSCRIPT_GUIDE.md) · [Data and reproduction](REPRODUCTION.md)

## Three complementary engineering modules

| Module | Engineering knowledge | Role in the forecast |
|---|---|---|
| **Observation anchoring (OA)** | The latest inspection records the current pavement condition | Carries measured depth into the output and learns the subsequent adjustment |
| **Layer–material encoding (LME)** | Material, layer position and thickness jointly describe pavement composition | Builds a shared structural representation using thickness-weighted layer pooling |
| **Monotonic load–temperature pathway (MLT)** | Increased exposure should not lower the forecast under otherwise fixed conditions | Accumulates daily load through a nonnegative rate and a nondecreasing temperature factor |

$$\widehat y_t = y_{t-1} + b(c_t) + G_t,\qquad G_t = a(c_t)\sum_j\Delta N_j h(T_j).$$

The backbone encodes the preceding exposure history. Current-interval load and temperature enter only through MLT. With history, structure and interval duration fixed, the response is monotonic by construction. [Architecture, equations and module comparisons](PROTOCOL.md)

## Results across backbones and monitoring datasets

| Evaluation | RIOHTrack | LTPP with local fitting |
|---|---|---|
| Held-out evaluation | 2,089 targets, 15 structures | 63 targets, 26 sections in 7 states |
| RMSE reduction versus matched direct-depth neural models | **49.5–57.3%** | **29.0–39.1%** |
| EKI RMSE | **0.2850–0.2937 mm** | **1.0080–1.0835 mm** |
| EKI versus anchoring alone | **4.0–14.4% lower RMSE** | 0.0007–0.0166 mm higher RMSE |
| Decreasing responses in EKI exposure checks | **0 / 125,340** individual-fit checks | **0 / 756** ensemble checks |

All four EKI backbones improve on persistence in both evaluations. On RIOHTrack, mean coverage of nominal 95% one-step prediction intervals is **95.0–95.3%**. Accuracy, response direction and interval calibration are assessed separately, allowing the modules' complementary contributions to be examined.

The complete results include four-backbone ablations, soft monotonicity penalties, monotonic XGBoost, inspection gaps and multistep forecasts. Benefits beyond anchoring depend on the monitoring setting; local LTPP fitting preserves response consistency without an additional accuracy gain over the anchored baseline. [Full comparisons and application conditions](RESULTS.md)

## Explore the study

| To examine | Start here |
|---|---|
| Cross-backbone accuracy and prediction intervals | [RIOHTrack results](RESULTS.md) |
| OA, LME and MLT contributions | [Module comparisons and original result tables](supplements/modules/README.md) |
| Independent highway data, local fitting and frozen-weight transfer | [LTPP evaluation](supplements/ltpp/README.md) |
| A particular figure or table in the paper | [Paper-to-code guide](MANUSCRIPT_GUIDE.md) |
| Prepared inputs, models, checksums and replay commands | [Reproduction](REPRODUCTION.md) |

The study evaluates reuse of an engineering design across four neural architectures and two monitoring datasets. Learned weights are fitted for each application. [Validation protocol and scope](PROTOCOL.md#study-status-and-inferential-scope)
