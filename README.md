# CS506 Project Proposal

### Members:
* Anna LaPrade - U14515609 - alaprade@bu.edu
* Saniya Sekhon - U68321677 - saneya52@bu.edu 
* Sorathorn Thongpitukthavorn - U01165313 - plum@bu.edu
* Aline Mukadi - U43727980 - alemuk@bu.edu
</br></br>

---

### Description

Air quality plays a critical role in public health, especially for vulnerable groups such as children, the elderly, and people with asthma. The goal of this project is to build a machine learning pipeline that predicts the daily air quality category in Boston (e.g., Good, Moderate, Unhealthy) using weather and allergen (e.g., tree, grass, weed pollen) data. The project will cover the full data science lifecycle, including data collection, cleaning, feature extraction, visualization, and model training.

Our approach has two components:
1. Predicting AQI and allergen levels for a given day of the year to capture broad seasonal patterns.
2. Incorporating weather data from the previous three days to improve short-term forecasts.

Since AQI and pollen are influenced both by cyclical seasonal trends and immediate weather conditions, this dual approach allows us to model long-term patterns while also accounting for short-term variability.
</br></br>

---


### Clear Goals


* Predict the daily pollen count (tree, grass, weed) and AQI with a mean absolute error (MAE) of less than 10% of the observed range, or achieve an R² score of at least 0.75 on the test set.
* Demonstrate a statistically significant relationship (p < 0.05) between specific weather variables and allergen/air quality levels.
* Develop visualizations that highlight trends, seasonal cycles, and allergen intensity for public health stakeholders.
</br></br>

---


### Data Collection


The project will collect a comprehensive dataset by integrating data from several sources.

#### **Data Sources**
* **Weather Data:** Daily weather variables from the **Open-Meteo API**, including temperature (max, min, avg), humidity, wind speed/direction, precipitation, and atmospheric pressure.
* **Allergen and Air Quality Data:** Daily pollen counts and AQI from the **AirNow API**, including pollutants (e.g., PM2.5).
* **Lagged and Time-Based Features:** Data from the **three previous days ($t-1$, $t-2$, $t-3$)** will be included as predictors. Features for the day of the year and month will capture seasonality.

#### **Data Alignment Strategy**
To ensure spatial and temporal consistency across data sources:
- **Primary Location:** All data will be anchored to the **Downtown Boston AirNow monitoring station** (lat: 42.3601°N, lon: -71.0589°W).
- **Weather Data:** Open-Meteo API will be queried using the exact coordinates of the AirNow station. Open-Meteo interpolates data from nearby weather stations, ensuring accurate local conditions.
- **AQI Data:** Collected directly from the Downtown Boston AirNow station via the AirNow API.
- **Pollen Data:** We will use Ambee API, which provides pollen estimates for our exact coordinates, ensuring alignment with the reference point.
- **Acceptable Distance Threshold:** All data sources will be within a **10-mile radius** of the reference point. For pollen, we acknowledge that counts represent regional conditions, which is appropriate given pollen dispersion patterns.
- **Temporal Alignment:** All data will be aggregated to **daily values**, with timestamps aligned to noon Eastern Time for consistency.
</br></br>

---


### Modeling
We will develop two prediction models:

1. **Pollen Prediction Model:**
   - **Task:** Multi-output regression predicting continuous pollen counts for tree, grass, and weed allergens.
   - **Algorithms:** Linear Regression (baseline), Random Forest Regressor, and XGBoost Regressor.
   - **Evaluation Metrics:** RMSE, MAE, and R² on continuous predictions.

2. **AQI Prediction Model:**
   - **Task:** Regression predicting continuous AQI values, followed by categorization into EPA standard categories (Good/Moderate/Unhealthy/etc.).
   - **Algorithms:** 
      - Linear Regression: as a baseline to observe relationships between variables
      - Random Forest Regressor: less prone to over-fitting, less affected by hyperparameterization
      - XGBoost Regressor: Generally considered to have higher performance, although could be too complex
   - **Evaluation Metrics:**
     - Regression: RMSE, MAE, and R² on continuous AQI values.
     - Classification: Accuracy, F1-score, and Confusion Matrix on categorized AQI predictions.

The models will use the following general form:

$Y_t = \beta_0 + \beta_1 X_{1,t} + \beta_2 X_{2,t-1} + \beta_3 X_{3,t-2} + ... + \epsilon$

Where:
* $Y_t$ is the allergen count or AQI on day $t$.
* $X$ are the various weather and air quality features, including lagged data from previous days.
</br></br>

---


### Data Visualization
Visualization will be crucial for exploratory data analysis and communicating findings:
* **Correlation Heatmaps:** To reveal relationships between weather features and target variables.
* **Time Series Line Plots:** To show long-term trends and spikes in AQI and allergen data. Dual-axis plots will compare weather variables to AQI/allergen counts.
* **Scatter Plots with Regression Lines:** To visualize pairwise relationships and assess linearity assumptions.
* **Feature Importance Bar Charts (XGBoost):** To highlight features with the strongest influence on predictions.
</br></br>
---

### Test Plan
The project will use an **out-of-sample forecasting** test plan:
* **Training Data:** All collected data up to the end of 2024.
* **Testing Data:** All data collected from **2025 onward** to evaluate performance on untrained data.
* **Benchmarking:** Models will be compared using **5-fold cross-validation** on the training set and final evaluation on the 2025 test set.
</br></br>

---

### Correlation Analysis - Heatmaps
To observe any possible linear relationships, Pearson correlation matrices were computed and displayed in a heat map for easy viewing. 
