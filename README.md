# C1-Tabular: Insurance Premium Prediction (Tabular ML)

This project predicts individual medical insurance charges using tabular features and multiple regression approaches, with a final focus on tuned XGBoost models.

## Project Goals
- Build reliable regression models for `charges`.
- Compare baseline vs tuned performance.
- Improve generalization and reduce overfitting.
- Prepare a deployable model export (`model.onnx`).

## Dataset and Features
The cleaned dataset includes core columns such as:
- `age`
- `bmi`
- `children`
- `gender_encoded`
- `smoker_encoded`
- region dummies (`region_northwest`, `region_southeast`, `region_southwest`)
- target: `charges`

Additional engineered features used in experiments:
- `age_squared`
- `smoker_bmi` (interaction between smoking status and BMI)

## Main Notebooks
- `eda.ipynb`  
  Exploratory data analysis and early baseline comparisons.
- `xgboost_reg.ipynb`  
  XGBoost regression directly on original `charges` scale.
- `xgboost_log_reg.ipynb`  
  XGBoost regression trained on `log1p(charges)` and evaluated on original scale via `expm1`.

## Modeling Workflow (xgboost_log_reg)
1. Load cleaned data.
2. Feature engineering (`age_squared`, `smoker_bmi`).
3. Train-test split.
4. Transform target for training: `y_train_log = log1p(y_train)`.
5. Train baseline XGBoost.
6. Tune hyperparameters with `RandomizedSearchCV`.
7. Use early stopping for training monitoring.
8. Train final tuned model.
9. Convert predictions back to original scale with `expm1`.
10. Evaluate with `R2`, `RMSE`, and `MAE`.
11. Run 5-fold cross-validation.
12. Analyze feature importance and residuals.

## Key Result Snapshot (xgboost_log_reg tuned)
On original `charges` scale:
- Test `R2`: `0.8555`
- Test `RMSE`: `4481.0091`
- Test `MAE`: `1980.3446`

Generalization signal:
- No strong overfitting indication from train-vs-validation RMSE gap in log space.

## Model Export
The final notebook includes ONNX export code:
- Output file: `model.onnx`
- Input order must follow `feature_cols` exactly.
- Model output is still in log space (`log1p` target), so convert back with:

```python
charges = np.expm1(pred_log_charges)
```

## How to Run
1. Open the project in Jupyter.
2. Run cells in order inside `xgboost_log_reg.ipynb`.
3. Re-run the final export cell to generate `model.onnx`.

## Notes
- If ONNX conversion raises feature-name errors, use the latest export cell version that remaps booster feature names to `f0, f1, ...`.
- Keep package versions consistent across training and export environments (`xgboost`, `onnxmltools`, `onnx`).
