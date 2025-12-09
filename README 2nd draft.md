# CS506 Final Project

**Members:**  
Anna LaPrade - U14515609 - alaprade@bu.edu  
Saniya Sekhon - U68321677 - saneya52@bu.edu  
Sorathorn Thongpitukthavorn - U01165313 - plum@bu.edu  
Aline Mukadi - U43727980 - alemuk@bu.edu  

## Midterm Report Video 
https://youtu.be/GTi8yNdaAHU 

## Final Report Video

<br></br>

## Overview

Air quality and pollen levels play a central role in public health, especially for individuals with asthma, allergies, or respiratory conditions. In Boston, these conditions fluctuate due to complex interactions between meteorology, seasonality, and biological cycles.

This project builds a full data-science pipeline to:

- Predict daily AQI values using past AQI and weather data.

- Predict total pollen counts using weather and temporal features.

- Understand nonlinear environmental patterns using clustering and exploratory visualizations.

The project incorporates data ingestion, cleaning, feature engineering, exploratory visualization, clustering analysis, and multiple machine-learning models, resulting in an interpretable and reproducible forecasting framework.
<br></br>

## Datasets

We combined three independently sourced datasets:

Data Type        | Source     | Description
---------------- | ---------- | -----------------------------------------------
Weather          | Open-Meteo | Hourly/daily meteorological measurements (temperature, wind, humidity, precipitation, etc.)
Pollen           | EPHT       | Daily tree, grass, weed, and total pollen counts
Pollutants + AQI | U.S. EPA   | PM2.5, O3, NO2, CO, SO2 and computed AQI

The pollen–weather merged dataset contains over 900 daily samples with consistent timestamps and unified variable naming. 

The AQI dataset (used in our air-quality forecasting model) contains 5,844 daily observations from 2009–2024.

## Data Processing

A complete preprocessing pipeline was built to standardize and enhance the environmental data.

### Column Standardization

- Removed unit markers (°C, mm, km/h)

- Converted to lowercase snake_case

- Consistent naming across all sources

### Temporal Feature Engineering

We added cyclical encodings to reflect seasonality:

- Day of year (sin, cos)

- Month (sin, cos)

- Weekday (sin, cos)

These allow models to learn repeating yearly and weekly biological cycles.

### Lag & Rolling Features

Both AQI and pollen have strong autoregressive behavior.

We generated:

- Lags: 1, 2, 3, 4, 7, 14, 21, 30, 365 days

- Rolling averages: 2, 3, 7, 14, 30 days

Weather-yesterday and weather-rolling windows
(to avoid data leakage: no same-day weather is used to predict same-day pollen)

### Spike Indicators

Pollen has discontinuous spikes. We computed:

- pollen_ratio_1d_7d = lag_1 / rolling_7
- z-score thresholding → spike or non-spike label

This enables spike-specific modeling.

### Train/Validation/Test Split (Pollen Models)

#### Chronological split:

- **Train:** 2020–2021

- **Validation:** 2022

- **Test:** 2023–2024

This reflects true forecasting (future unseen data).

# Exploratory Data Analysis
## Distributions & Missingness

Pollen and pollutants are extremely skewed, with long-tailed spike distributions.
Weather variables are smooth and seasonal.
Missing-matrix visualizations show gaps in pollen & pollutant records, all addressed via imputation.

<img src="visualizations\Plum\histogram_weather.png" width="400"/>

<img src="visualizations\Plum\histogram_pollen.png" width="400"/>

<img src="visualizations\Plum\histogram_aqi.png" width="400"/>

<img src="visualizations\Plum\missing_data_matrix_aqi.png" width="400"/>

<img src="visualizations\Plum\missing_data_matrix_pollen.png" width="400"/>

<img src="visualizations\Plum\missing_data_matrix_weather.png" width="400"/>

## Correlation & Lag Structure

Initial Pearson correlations show **weak linear relationships** between weather, AQI, and pollen. Even lag correlations reveal only mild linear patterns.

This motivated nonlinear clustering & nonlinear ML models.

<img src="visualizations\Anna\correlation_heatmap.png" width="400"/>
<img src="visualizations\Anna\strong_correlation_heatmap.png" width="400"/>
<img src="visualizations\Anna\lagged_correlation_heatmap.png" width="400"/>
<img src="visualizations\Anna\strong_lagged_correlation_heatmap.png" width="400"/>

## Seasonal Patterns

Monthly time-series reveal:

- Tree pollen peaks sharply April–May

- Grass/weed peaks later in summer

- AQI elevates during hot, stagnant summer periods

- Temperature strongly aligns with biological cycles

<img src="Lola/Plots (Monthly)/by fours/AQI_monthly_timeseries.png" width="400"/>

<img src="Lola\Plots (Monthly)\by fours\Grass_monthly_timeseries.png" width="400"/>

<img src="Lola\Plots (Monthly)\by fours\Ragweed_monthly_timeseries.png" width="400"/>

<img src="Lola\Plots (Monthly)\by fours\Total_Pollen_monthly_timeseries.png" width="400"/>

<img src="Lola\Plots (Monthly)\by fours\Tree_monthly_timeseries.png" width="400"/>

<img src="Lola\Plots (Monthly)\by fours\Weed_monthly_timeseries.png" width="400"/> 

# Clustering Analysis: Nonlinear Environmental Structure

Linear correlations underestimated the true relationships.
K-Means clustering uncovered distinct environmental regimes.

### Weather → Pollen Clusters

Using k = 17:

- **High pollen spikes:** cool, dry early spring days

- **Low pollen:** rainy or humid periods

- **Moderate pollen:** transitional weather days

### Weather → AQI Clusters (k = 14)

- **High AQI:** hot, dry, stagnant days

- **Low AQI:** windy or rainy periods

- **Moderate AQI:** mild transitional conditions

### AQI ↔ Pollen Clusters (k = 11)

States emerge such as:

- Low AQI + Low Pollen (winter/rain)

- High Pollen + Moderate AQI (late spring)

- High AQI + High Pollen (stagnant warm spring days)

### Date-Aware Clustering

- Adding month/day reveals:

- Pollen clusters concentrated April–June

- AQI clusters in July–August

- Suppression periods aligning with rainfall events

<img src="visualizations\Anna\weather_pollen_kmeans_17.png" width="400"/> 

<img src="visualizations\Anna\weather_aqi_kmeans_14.png" width="400"/> 

<img src="visualizations\Anna\aqi_pollen_kmeans_11.png" width="400"/> 

<img src="visualizations\Anna\weather_pollen_kmeans_18_dates.png.png" width="400"/> 

<img src="visualizations\Anna\weather_aqi_kmeans_14_dates.png" width="400"/> 

<img src="visualizations\Anna\aqi_pollen_kmeans_12_date.png.png" width="400"/> 

<img src="visualizations\Saniya\scripts\plots\aqi\clustering_elbow_method.png" width="400"/> 

Environmental behavior is strongly non-linear and regime-based, not linearly correlated. This justifies nonlinear ML modeling.

# AQI Prediction Model

We evaluated Random Forests and Gradient Boosting Regression using lag features plus weather.

The AQI dataset consists of **EPA AQS daily pollutant measurements** for Suffolk County, MA:

| Pollutant | Variable                |
| --------- | ----------------------- |
| PM2.5     | Fine particulate matter |
| O₃        | Ozone                   |
| CO        | Carbon monoxide         |
| NO₂       | Nitrogen dioxide        |
| SO₂       | Sulfur dioxide          |

Weather variables were collected from **Open-Meteo**, including:

- daily mean temperature  
- daily precipitation  
- maximum wind speed and gusts  

## Forecasting Setup

We split the data chronologically to mimic real forecasting and avoid leakage:
| Split      | Period                  | Size (days) | Share |
| ---------- | ----------------------- | ----------- | ----- |
| Train      | 2009-01-01 → 2022-12-30 | 5,112       | 87.5% |
| Validation | 2022-12-31 → 2023-12-30 | 365         | 6.2%  |
| Test       | 2023-12-31 → 2024-12-31 | 367         | 6.3%  |

## Feature Engineering

All features use past information

To ensure realistic forecasting, the model does not use same-day AQI or weather values. Instead, features include:

- lagged AQI summaries (e.g., AQI_lag_2, AQI_lag_3, AQI_roll3, AQI_roll7, AQI_diff_1, AQI_diff_7)

- lagged pollutant concentrations and rolling averages (lags 1, 3, 7; 3-day and 7-day rolls)

- lagged weather features and rolling averages (temperature, wind, rain)

- pollutant × weather interactions (e.g., PM25_wind, O3_temp, NO2_wind)

- temporal patterns (month, day-of-year, day-of-week + sine/cosine encodings, weekend flag)

After feature construction and dropping rows without sufficient history, we obtain:

- 53 predictive features

- ≈4,800 training examples (train + val)

## Models Evaluated

We compared several algorithms:

1. Ridge Regression (baseline linear model)

2. Lasso Regression (sparse linear baseline)

3. XGBoost Regressor

4. LightGBM Regressor

5. Random Forest Regressor

Hyperparameters for the tree-based models were tuned on the validation year, then each model was retrained on train + validation and evaluated once on the held-out test year (2024).

## Performance
Test Performance (2024)

| Model         | R²        | RMSE     | MAE      |
| ------------- | --------- | -------- | -------- |
| Ridge         | 0.30      | —        | —        |
| XGBoost       | 0.38      | 8.58     | 6.57     |
| LightGBM      | 0.39      | 8.52     | 6.53     |
| Random Forest | 0.39      | 8.51     | 6.61     |
| **Ensemble**  | **0.397** | **8.46** | **6.50** |

The ensemble explains about 40% of the variance in daily AQI while keeping typical prediction errors within about ±6–9 AQI points

## Interpretation

Daily AQI is highly stochastic and influenced by:

- meteorology

- short-term emission events

- external wildfires and regional transport

- unobserved sources not in the dataset

Given a single-city, daily-resolution dataset (~5.8k rows), achieving R² ≈ 0.4 is consistent with reported difficulty levels for AQI forecasting. The ensemble:

- captures seasonal trends and multi-day pollution episodes

- tracks most moderate pollution days

- still underpredicts rare high-AQI spikes, which are often driven by wildfire smoke or unusual synoptic events that are hard to infer from local history alone

Classification Perspective

If we convert continuous AQI predictions into categorical EPA levels:

- 0 = Good (≤ 50)

- 1 = Moderate (51–100)

- 2+ = Unhealthy ranges (aggregated)

on the test set we obtain:

- Macro F1 ≈ 0.63

- Weighted F1 ≈ 0.92

- Precision for “Good” ≈ 0.94

This suggests that, even if exact AQI is imperfect, the model is useful for high-level messaging such as whether air quality is likely to remain “Good” vs. shift into “Moderate or worse.”

<img src="visualizations\AQI\A_vs_P_over_time.png" width="400"/> 
<img src="visualizations\AQI\actual_vs_pred_scatter.png" width="400"/> 
<img src="visualizations\AQI\Distribution_pred_residuals.png" width="400"/> 
<img src="visualizations\AQI\Residuals_over_time.png" width="400"/> 
<img src="visualizations\AQI\Res_vs_actual.png" width="400"/> 
<img src="visualizations\AQI\Top_features.png" width="400"/> 

# Pollen Prediction Models

Predicting pollen is substantially more difficult due to:

- Biological triggers (not all recorded)

- Environmental thresholds (temperature, rain suppression)

- Sudden, sharp multi-day spikes

We implemented three non-linear models, each capturing different aspects.

## LightGBM Regressor (Best Overall Model)
### Performance

| MAE  | 84.3  |
| RMSE | 238.3 |
| R²   | 0.357 |

### Interpretation

- Learns smooth seasonal and short-term pollen movements

- Underpredicts extreme spikes (biological events not captured in weather alone)

- Most stable generalization across 2023–2024 test set

<img src="visualizations\pollen model images\lightgbm_actual_vs_pred.png" width="400"/> 

<img src="visualizations\pollen model images\lightgbm_residuals.png" width="400"/> 

<img src="visualizations\pollen model images\lightgbm_timeseries.png" width="400"/> 

## XGBoost Regressor

### Performance

R² ≈ 0.30  
RMSE ≈ 210  

### Interpretation

- Performs competitively on mid-range counts

- Slightly worse generalization than LightGBM

- Similar difficulties with spike prediction

<img src="visualizations\pollen model images\xgboost_actual_vs_pred.png" width="400"/> 

<img src="visualizations\pollen model images\xgboost_actual_vs_pred_timeseries.png" width="400"/> 

<img src="visualizations\pollen model images\xgboost_feature_importance.png" width="400"/> 

3. Two-Stage Spike Model

(Classifier → Spike Regressor + Non-Spike Regressor)

## Spike Classifier

**AUC: 0.983**

Accurately distinguishes spike vs. non-spike conditions.

### Regression Results

**MAE: ~86  **
**RMSE: ~233  **
**R²: 0.38  **

### Interpretation

- Best theoretical structure

- Limited by small number of spike days

- Useful for threshold-based alerts, even where RMSE is not lowest

<img src="visualizations\pollen model images\spike_model_auc.png" width="400"/> 

<img src="visualizations\pollen model images\spike_model_residuals.png" width="400"/> 

<img src="visualizations\pollen model images\spike_model_timeseries.png" width="400"/> 

# Model Comparison

| **Model**    | **MAE**  | **RMSE**| **R²**   | **Strength **                  |
| ------------ | -------- | ------- | -------- | ------------------------------ |
| **LightGBM** | **84.3** | 238     | **0.36** | Best overall generalization    |
| Spike Model  | 86       | 233     | 0.38     | Best conceptual fit for spikes |
| XGBoost      | 86       | **210** | 0.30     | Strong mid-range performance   |

- **No model perfectly captures extreme spikes**, which align with biological release triggers not accounted for in weather data.

- **LightGBM** is the most robust and interpretable final choice.

- **Spike model** is valuable for distinguishing hazard days.

<img src="visualizations\Saniya\scripts\plots\aqi\full_vs_simple_model.png" width="400"/> 
