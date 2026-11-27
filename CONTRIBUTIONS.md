# Individual contributions

The course requires each member's contributions to be recorded in the GitHub commit history, with a summary in the final report. **Each member commits their own work from their own GitHub account**, and adds their entries below.

| Member | GitHub | Contributions |
|---|---|---|
| Vaibhav Sahni | `vaibhav-sahni` | Repository and validation setup (fixed hold-out, folds, exact cost scorer); EDA and data fixes (histogram realignment, dropped columns, censoring and depot-shift analysis); versions v0–v9 (decision rule, calibration, per-vehicle cost model, follow-up and feature experiments, depot-shift correction, histogram-shape features); report notebook; Kaggle submissions #1–6 |
| Shivanshu Kumar Singh | `Shivanshu-Kr-Singh` | Built the probability-calibration pipeline and V7 depot-shift correction, using EM, SHAP, and regret analysis to improve cost-sensitive maintenance decisions and achieve a public score of 19,067. Tested and rejected unreliable missingness/follow-up variants, and added reusable calibration and feature-drift validation tools used in later model versions.
| _(name)_ | `Daksh183` | Fleet scoring and baselines (v0 cost scorer and v1 sensor analysis). Cost-based inspection and parts-budget rules (v3). Notebook setup and sensor histogram analysis (v0–v5 notebook builder and v8 exploration scripts). V32–V34 experiment documentation and submission packaging (validation reports, rebuild notebooks, and Kaggle CSVs). |
| Naveen Jangra | `Naveen23-hub` | V5 repair-cost model. Added a model that predicts repair costs for individual vehicles and uses those estimates to guide inspections and spare-parts budgets. V8 sensor features and validation. Added features that summarize sensor histograms, such as the share of readings in each bin. Tested their effect on repair-cost predictions across 15 validation folds. |
