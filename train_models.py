import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the cleaned and preprocessed dataset
    csv_path = 'cleaned_fermentation_dataset.csv'
    print(f"Loading cleaned dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['Yield'])
    y = df['Yield'].values
    
    # Features and categorical definitions
    cat_cols = ['Substrate', 'Microbial_Strain', 'Dataset_Source', 'Country']
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # Fit StandardScaler on the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("\nTraining models...")
    
    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Model 3: XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    
    # Evaluation Metrics
    models = {'Linear Regression': (lr, y_pred_lr), 
              'Random Forest': (rf, y_pred_rf), 
              'XGBoost': (xgb, y_pred_xgb)}
              
    print("\n=== MODEL COMPARISON ===")
    best_r2 = -float('inf')
    best_model_name = None
    best_model = None
    metrics_summary = {}
    
    for name, (model, y_pred) in models.items():
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        metrics_summary[name] = {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        print(f"{name}:")
        print(f"  R² Score : {r2:.4f}")
        print(f"  MAE      : {mae:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Save Feature Importance plot of the best model (if Random Forest or XGBoost)
    if best_model_name in ['Random Forest', 'XGBoost']:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
        plt.title(f'Feature Importance ({best_model_name})')
        plt.xlabel('Importance Value')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("Saved feature importance chart to 'feature_importance.png'")
        
    # Categorical mappings for Streamlit UI
    substrate_mapping = {'Control': 0, 'D-Glucose': 1, 'L-Fucose': 2, 'L-Rhamnose': 3}
    strain_mapping = {'Clostridium strain AK1': 0, 'Not Specified': 1}
    source_mapping = {'Bioethanol_Growth_Prediction_Dataset': 0, 'Clostridium_AK1_Lab_Data': 1}
    country_mapping = {'Brazil': 0, 'China': 1, 'Germany': 2, 'India': 3, 'USA': 4}
    
    # Save preprocessing and best model pipeline to a single pickle file
    pipeline = {
        'best_model_name': best_model_name,
        'model': best_model,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(),
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'mappings': {
            'Substrate': substrate_mapping,
            'Microbial_Strain': strain_mapping,
            'Dataset_Source': source_mapping,
            'Country': country_mapping
        },
        'metrics': metrics_summary,
        'medians': X.median().to_dict()  # for default fillings in app
    }
    
    with open('fermentation_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("Exported all pipeline artifacts to 'fermentation_pipeline.pkl'")

if __name__ == '__main__':
    main()
