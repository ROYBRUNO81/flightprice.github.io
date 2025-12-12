# Flight Ticket Price Prediction

This project analyzes Bangladesh flight-fare data end to end—cleaning the raw dataset, engineering predictive features, exploring price drivers, and benchmarking multiple machine-learning models to estimate total ticket cost (`Total Fare (BDT)`).

## Project Workflow

1. **Data Understanding & Cleaning**
   - Loaded the 57k-row Kaggle dataset and verified schema, missingness, and categorical levels.
   - Converted timestamp columns to `datetime`, removed obvious leakage columns (`Base Fare`, `Tax & Surcharge`), and standardized categorical dtypes.

2. **Feature Engineering**
   - Extracted calendar signals (`dep_month`, `dep_dayofweek`, `dep_hour`, `dep_is_weekend`) from the departure timestamp.
   - Converted `Stopovers` to a numeric count and calculated `route_frequency` to capture route competitiveness.
   - Split the curated features into numeric vs categorical lists to support reproducible preprocessing.

3. **Exploratory Data Analysis**
   - Visualized fare distributions, airline and class comparisons, seasonal effects, the role of stopovers, and correlations between numeric variables.
   - Key findings: fares are highly skewed, seasonality and class have the largest price gaps, and airline brand alone explains relatively little variance.

4. **Preprocessing Pipeline**
   - Built a `ColumnTransformer` pipeline with median-imputation + scaling for numeric fields and mode-imputation + one-hot encoding for categoricals.
   - Created an 80/20 train-test split (`train_test_split` with `random_state=42`) to ensure consistent evaluation.

5. **Modeling & Evaluation**
   - **Linear baselines:** Ordinary Least Squares and Ridge regression on both raw and `log1p` targets. Log-transforming the target reduced RMSE from ~53.5k to ~48.3k BDT, while Ridge offered no additional gain.
   - **Tree ensembles:** RandomForestRegressor and HistGradientBoostingRegressor fit through the same preprocessing pipeline. The best model—HistGradientBoosting on the raw target—achieved **R² ≈ 0.677** and **RMSE ≈ 46.4k BDT**, outperforming all linear variants by capturing nonlinear interactions among seasonality, class, route frequency, and stopovers.

6. **Interpretation & Insights**
   - Seasonality, cabin class, and route-level demand are the dominant drivers of fare variability.
   - Log-transforming the target benefits linear models but is unnecessary for tree ensembles, which already tolerate skew.
   - Compute constraints (Colab runtime) limited tree depth/iterations, yet performance still improved ~13% over the baseline, indicating further gains are possible with richer hyperparameter sweeps or GPU-backed runtimes.

## Data Source

Flight Price Dataset of Bangladesh — [Kaggle](https://www.kaggle.com/datasets/mahatiratusher/flight-price-dataset-of-bangladesh)

