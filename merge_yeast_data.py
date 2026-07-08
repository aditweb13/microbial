import pandas as pd
import numpy as np

# Mappings from the original train_models.py to decode the dataset
SUBSTRATE_REV = {0: 'Control', 1: 'D-Glucose', 2: 'L-Fucose', 3: 'L-Rhamnose'}
STRAIN_REV = {0: 'Clostridium strain AK1', 1: 'Not Specified'}
SOURCE_REV = {0: 'Bioethanol_Growth_Prediction_Dataset', 1: 'Clostridium_AK1_Lab_Data'}
COUNTRY_REV = {0: 'Brazil', 1: 'China', 2: 'Germany', 3: 'India', 4: 'USA'}

def main():
    print("Loading original dataset...")
    df_main = pd.read_csv('cleaned_fermentation_dataset.csv')
    
    # Inverse transform the categoricals to strings so we can append new strings
    df_main['Substrate'] = df_main['Substrate'].map(SUBSTRATE_REV)
    df_main['Microbial_Strain'] = df_main['Microbial_Strain'].map(STRAIN_REV)
    df_main['Dataset_Source'] = df_main['Dataset_Source'].map(SOURCE_REV)
    df_main['Country'] = df_main['Country'].map(COUNTRY_REV)
    
    # Get medians for imputation
    medians = df_main.select_dtypes(include=[np.number]).median().to_dict()
    
    print("Loading 30C Yeast Dataset...")
    df_30 = pd.read_csv('ethanol_30C_216strains.csv')
    
    print("Loading 40C Yeast Dataset...")
    df_40 = pd.read_csv('ethanol_40C_31strains.csv')
    
    # Process 30C Dataset
    df_30_clean = pd.DataFrame()
    df_30_clean['Microbial_Strain'] = 'Saccharomyces cerevisiae (Strain: ' + df_30['combination'] + ')'
    df_30_clean['Ethanol_mM'] = df_30['Average Ethanol g/L'] * (1000 / 46.07)
    df_30_clean['OD600'] = df_30['Average OD600']
    df_30_clean['Sugar_Consumed_pct'] = df_30['Glucose consumption g/L '] # It's basically out of 100g/L
    df_30_clean['Acetate_mM'] = df_30['Acetate g/L'] * (1000 / 60.05)
    df_30_clean['Temperature_C'] = df_30['temperature_C']
    
    # Process 40C Dataset
    df_40_clean = pd.DataFrame()
    df_40_clean['Microbial_Strain'] = 'Saccharomyces cerevisiae (Strain: ' + df_40['combination'] + ')'
    df_40_clean['Ethanol_mM'] = df_40['EtOH(g/L)'] * (1000 / 46.07)
    df_40_clean['OD600'] = df_40['OD600']
    df_40_clean['Sugar_Consumed_pct'] = df_40['%Glucose consumption']
    df_40_clean['Acetate_mM'] = df_40['Acetate'] * (1000 / 60.05)
    df_40_clean['Temperature_C'] = df_40['temperature_C']
    
    # Combine yeast data
    df_yeast = pd.concat([df_30_clean, df_40_clean], ignore_index=True)
    
    # Fill remaining columns with medians or constants
    df_yeast['Initial_pH'] = 5.0
    df_yeast['Substrate'] = 'D-Glucose'
    df_yeast['Dataset_Source'] = 'IMCTlab_Yeast_Data'
    df_yeast['Country'] = 'USA' # Defaulting to USA for macro variables
    
    for col in df_main.columns:
        if col not in df_yeast.columns and col not in ['Substrate', 'Microbial_Strain', 'Dataset_Source', 'Country']:
            df_yeast[col] = medians.get(col, 0)
            
    # Ensure column order matches
    df_yeast = df_yeast[df_main.columns]
    
    # Concatenate all
    df_final = pd.concat([df_main, df_yeast], ignore_index=True)
    
    # Save the new dataset
    df_final.to_csv('cleaned_fermentation_dataset_v2.csv', index=False)
    print(f"Successfully generated cleaned_fermentation_dataset_v2.csv with {len(df_final)} rows.")
    print(f"Added {len(df_yeast)} new yeast strains!")

if __name__ == '__main__':
    main()
