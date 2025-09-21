# CS506 Project Proposal


### Members:
* Anna LaPrade - U14515609 - alaprade@bu.edu
* Saniya Sekhon - U68321677 - saneya52@bu.edu 
* Sorathorn Thongpitukthavorn - U01165313 - plum@bu.edu
* Aline Mukadi - U43727980 - alemuk@bu.edu
</br></br>

---

### Description


Air quality plays a critical role in public health, especially for vulnerable groups such as children, the elderly, and people with asthma. The goal of this project is to build a machine learning pipeline that predicts the daily air quality category in Boston (e.g., Good, Moderate, Unhealthy) using weather and allergen ((e.g. tree, grass, weed pollen) data. The project will cover the full data science lifecycle, including data collection, cleaning, feature extraction, visualization, and model training.

Our approach has two components: (1) predicting AQI and allergen levels for a given day of the year to capture broad seasonal patterns, and (2) incorporating weather data from the previous three days to improve short-term forecasts. Since AQI and pollen are influenced both by cyclical seasonal trends and immediate weather conditions, this dual approach allows us to model long-term patterns while also accounting for short-term variability.
</br></br>

---


### Clear Goals


* Successfully predict the daily pollen count for various types (e.g., tree, grass, weed) and the overall Air Quality Index (AQI).
* Demonstrate a quantifiable relationship between specific weather variables and allergen/air quality levels.
* Create data visualizations that show global trends, seasonal cycles, and the intensity of allergens over time.
</br></br>

---


### Data Collection


The project will collect a comprehensive dataset by integrating data from several sources.


* **Weather Data**: We will gather daily weather variables from an API (Open-Meteo), including temperature (max, min, avg), humidity, wind speed/direction, precipitation, and possibly atmospheric pressure.
* **Allergen and Air Quality Data**: We will use APIs, such as the **AirNow API**, to collect daily pollen counts and the Air Quality Index (AQI), including its pollutants (e.g., PM2.5).
* **Lagged and Time-Based Features**: To capture the time-series nature of the problem, we will also include data from the **three previous days ($t-1$, $t-2$, $t-3$)** as predictors (for example, strong winds or rain could affect pollen for the next couple days). We will also add features for the day of the year and month to capture effects of seasonality.
</br></br>

---


### Modeling


We will begin with **linear regression**, which provides an interpretable baseline model. The linear relationship is justified by the theoretical connection between weather variables and allergen levels (i.e., wind speed linearly increasing pollen dispersal) as well as the seasonal nature of AQI and allergen levels.


The model will take the following general form:


$Y_t = \beta_0 + \beta_1 X_{1,t} + \beta_2 X_{2,t-1} + \beta_3 X_{3,t-2} + ... + \epsilon$


Where:
* $Y_t$ is the allergen count or AQI on day $t$.
* $X$ are the various weather and air quality features, including the lagged data from previous days.


After establishing this baseline, we plan on going more in depth by using XGBoost in order to consider more complex interactions (for example, what occurs when heat is high and wind is low vs low heat and low wind, etc.)
</br></br>

---


### Data Visualization


Visualization will be crucial for both exploratory data analysis and communicating the model's findings.


* **Correlation Heatmaps**: These would be used to reveal relationships between weather features and target variables, guiding how we further develop our models. 
* **Time Series Line Plot**: A primary visualization to show **long-term trends and spikes** in both AQI and allergen data over time. Dual-axis plots will be used to compare weather variables to AQI/allergen counts on the same timeline. 
* **Scatter Plots with Regression Lines**: This would help visualize pairwise relationships between functions and visually assess the linearity assumptions made by our baseline regression model. 
* **Feature Importance Bar Charts (XGBoost)**: This would be used to visualize which features have the strongest influence on the model’s predictions. 
</br></br>

---


### Test Plan


The project will use an **out-of-sample forecasting** test plan, which is appropriate for time-series data.


* **Training Data**: We will train our model on all collected data up to the end of 2024.
* **Testing Data**: We will withhold and test the model on all data collected from **2025 onward**. This is to evaluate the model on untrained data.
</br></br>
