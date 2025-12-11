import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from functools import reduce

class LightGBMPollen:
    def __init__(self):
        self.models = {}
        self.pollen_types = ["Tree", "Grass", "Weed"]
        self.trained = False
        self.test_data = None
        self.metrics = {}
        
        # Exact split years from your script
        self.VAL_YEAR = 2022
        self.TEST_YEAR = 2023

    def short_description(self):
        return "LightGBM Classifier (Exact Script Replication: Lags, Ratios, 2022 Val Split)."

    def _clean_column_name(self, col):
        """Replicates: c.replace(' ', '_').replace('(', '').replace(')', '').replace('°', '')"""
        return col.replace(" ", "_").replace("(", "").replace(")", "").replace("°", "")

    def _merge_data(self, data_dict):
        dfs = []
        if "pollen" in data_dict:
            df_p = data_dict["pollen"].copy()
            # Normalize Date -> date for merge, then we rename back later if needed
            if "Date" in df_p.columns: df_p = df_p.rename(columns={"Date": "date"})
            if "time" in df_p.columns: df_p = df_p.rename(columns={"time": "date"})
            dfs.append(df_p)
            
        if "weather" in data_dict:
            df_w = data_dict["weather"].copy()
            if "time" in df_w.columns: df_w = df_w.rename(columns={"time": "date"})
            dfs.append(df_w)
            
        if "pollutants" in data_dict:
            df_pol = data_dict["pollutants"].copy()
            if "Date" in df_pol.columns: df_pol = df_pol.rename(columns={"Date": "date"})
            if "time" in df_pol.columns: df_pol = df_pol.rename(columns={"time": "date"})
            dfs.append(df_pol)

        if not dfs:
            raise RuntimeError("No data found in data_dict")

        df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
        return df_merged

    def _classify_pollen(self, df):
        """Exact classification thresholds from your script."""
        def classify_tree(c):
            if c < 1: return 0
            elif c <= 14: return 1
            elif c <= 89: return 2
            elif c <= 1499: return 3
            else: return 4
        
        def classify_weed(c):
            if c < 1: return 0
            elif c <= 9: return 1
            elif c <= 49: return 2
            elif c <= 499: return 3
            else: return 4

        def classify_grass(c):
            if c < 1: return 0
            elif c <= 4: return 1
            elif c <= 19: return 2
            elif c <= 199: return 3
            else: return 4

        # Apply classifications and cast to category (int codes)
        if "Tree" in df.columns: df["Tree_Level"] = df["Tree"].apply(classify_tree).astype(int)
        if "Weed" in df.columns: df["Weed_Level"] = df["Weed"].apply(classify_weed).astype(int)
        if "Grass" in df.columns: df["Grass_Level"] = df["Grass"].apply(classify_grass).astype(int)
        return df

    def _engineer_features(self, df):
        # 1. Clean Column Names
        df.columns = [self._clean_column_name(c) for c in df.columns]
        
        # Ensure 'date' is datetime (normalized name might be 'date' or 'Date' depending on clean)
        date_col = 'date' if 'date' in df.columns else 'Date'
        df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
        df["Year"] = df["Date"].dt.year

        # 2. Numeric Conversion & FillNa(0)
        # Replicating logic: select all numeric except Date/metadata, force numeric, fill 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Ensure we don't accidentally fill Date or Year if they got caught (Year is numeric but handled separately)
        cols_to_fill = [c for c in numeric_cols if c not in ['Date', 'Year']]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)

        # 3. Time Features
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # 4. Pollen Features (Exact Lags/Rolls/Ratios)
        lag_days = [1, 2, 3, 4, 7, 14, 21, 30] # Exact list from script
        engineered_cols = []
        
        for pt in self.pollen_types:
            if pt not in df.columns: continue
            
            # Rolling
            df[f"{pt}_roll_2day"] = df[pt].shift(1).rolling(2).mean()
            df[f"{pt}_roll_7day"] = df[pt].shift(1).rolling(7).mean()
            df[f"{pt}_roll_30day"] = df[pt].shift(1).rolling(30).mean()
            engineered_cols.extend([f"{pt}_roll_2day", f"{pt}_roll_7day", f"{pt}_roll_30day"])

            # Lags
            for lag in lag_days:
                df[f"{pt}_lag_{lag}"] = df[pt].shift(lag)
                engineered_cols.append(f"{pt}_lag_{lag}")
            
            # Ratio (Exact formula from script)
            df[f"{pt}_ratio_1d_7d"] = df[f"{pt}_lag_1"] / df[f"{pt}_roll_7day"]
            engineered_cols.append(f"{pt}_ratio_1d_7d")

        # 5. Generate Targets
        df = self._classify_pollen(df)

        # 6. Define Feature Set
        # Remove original pollen counts and metadata
        target_levels = [f"{pt}_Level" for pt in self.pollen_types]
        remove_cols = ["Tree", "Grass", "Weed", "Ragweed", "Total_Pollen", 
                       "date", "Date", "Year", "day_of_year", "month"] + target_levels
        
        # Base numeric features (excluding pollen counts)
        numeric_no_pollen = [c for c in numeric_cols if c not in ["Tree", "Grass", "Weed", "Ragweed", "Total_Pollen"]]
        base_feats = [c for c in numeric_no_pollen if c not in remove_cols]
        
        extra_feats = ["day_sin", "day_cos", "month_sin", "month_cos"] + engineered_cols
        
        features = base_feats + extra_feats
        # Filter to ensure they exist in DF
        features = [f for f in features if f in df.columns]

        # 7. Drop NaNs based on rolling columns
        required_cols = [f"{pt}_roll_30day" for pt in self.pollen_types if f"{pt}_roll_30day" in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols).reset_index(drop=True)

        return df, features

    def predict(self, data_dict):
        # 1. Merge & Prep
        df_merged = self._merge_data(data_dict)
        df_proc, features = self._engineer_features(df_merged)

        # 2. Split Train / Validation / Test (Exact Logic)
        train_idx = df_proc["Year"] < self.VAL_YEAR
        val_idx = df_proc["Year"] == self.VAL_YEAR
        test_idx = df_proc["Year"] >= self.TEST_YEAR
        
        X_train = df_proc.loc[train_idx, features]
        X_val = df_proc.loc[val_idx, features]
        X_test = df_proc.loc[test_idx, features]
        
        test_dates = df_proc.loc[test_idx, "Date"]
        results = pd.DataFrame({"date": test_dates.values})

        # 3. Train Loop
        overall_accuracy = []

        for pt in self.pollen_types:
            target_col = f"{pt}_Level"
            if target_col not in df_proc.columns:
                continue

            y_train = df_proc.loc[train_idx, target_col]
            y_val = df_proc.loc[val_idx, target_col]
            y_test = df_proc.loc[test_idx, target_col]

            # Class Weights Logic
            class_counts = y_train.value_counts().sort_index()
            # Dynamic Max Level based on training data or default to 4
            max_level = max(class_counts.index) if not class_counts.empty else 4
            total_samples = len(y_train)
            num_classes = len(class_counts)
            
            class_weights = {}
            for i in range(max_level + 1):
                count = class_counts.get(i, 1e-6)
                class_weights[i] = total_samples / (num_classes * count)
            
            # Filter weights only for classes present in training
            present_classes = set(y_train.unique())
            clean_weights = {k: v for k, v in class_weights.items() if k in present_classes}

            # Init LGBM
            clf = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=max_level + 1,
                n_estimators=1000,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                class_weight=clean_weights,
                verbose=-1
            )
            
            # Fit with Early Stopping using Validation Set
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0) # Silence logging
                ]
            )
            self.models[pt] = clf

            # Predict
            preds = clf.predict(X_test)
            
            # Store results
            results[f"{pt}_Observed"] = y_test.values
            results[f"{pt}_Forecast"] = preds
            
            acc = accuracy_score(y_test, preds)
            self.metrics[pt] = acc
            overall_accuracy.append(acc)

        self.trained = True
        self.metrics["Average_Accuracy"] = np.mean(overall_accuracy) if overall_accuracy else 0
        self.test_data = results
        
        return results

    def plot_results(self, data=None):
        if not self.trained or self.test_data is None:
            fig = go.Figure()
            fig.add_annotation(text="Model not trained.", x=0.5, y=0.5, showarrow=False)
            return fig

        # Default to Tree, or first available
        viz_type = "Tree"
        if f"{viz_type}_Observed" not in self.test_data.columns:
            possible = [p for p in self.pollen_types if f"{p}_Observed" in self.test_data.columns]
            if possible: viz_type = possible[0]

        y_test = self.test_data[f"{viz_type}_Observed"]
        preds = self.test_data[f"{viz_type}_Forecast"]
        dates = self.test_data['date']
        acc = self.metrics.get(viz_type, 0)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{viz_type} Pollen: Observed vs Forecast (Accuracy={acc:.2f})", f"{viz_type} Pollen Level Over Time"),
            vertical_spacing=0.15
        )

        # 1. Jittered Scatter
        jitter_y = preds + np.random.normal(0, 0.1, size=len(preds))
        jitter_x = y_test + np.random.normal(0, 0.1, size=len(y_test))

        fig.add_trace(
            go.Scatter(x=jitter_x, y=jitter_y, mode='markers', name='Predictions', 
                       marker=dict(color='orange', opacity=0.6, size=6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 4], y=[0, 4], mode='lines', name='Perfect Fit', 
                       line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # 2. Time Series
        fig.add_trace(
            go.Scatter(x=dates, y=y_test, mode='lines+markers', name='Observed Level', line=dict(color='gray', width=1)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=preds, mode='lines+markers', name='Forecast Level', line=dict(color='orange', width=2, dash='dot')),
            row=2, col=1
        )

        # Axis Formatting
        fig.update_yaxes(tickvals=[0,1,2,3,4], ticktext=["None","Low","Mod","High","V.High"], row=1, col=1)
        fig.update_xaxes(tickvals=[0,1,2,3,4], ticktext=["None","Low","Mod","High","V.High"], title="Observed Level", row=1, col=1)
        fig.update_yaxes(tickvals=[0,1,2,3,4], ticktext=["None","Low","Mod","High","V.High"], row=2, col=1)

        fig.update_layout(template="plotly_white", showlegend=True, height=900)
        return fig