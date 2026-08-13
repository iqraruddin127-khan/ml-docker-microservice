import pandas as pd
from scipy.stats import ks_2samp

def evaluate_production_drift(baseline_path, production_logs_path):
    print("\n==============================================")
    print("📈 REAL ESTATE DATA DRIFT MONITORING PIPELINE ")
    print("==============================================\n")
    
    # Load historical reference vs operational data arrays
    train_df = pd.read_csv(baseline_path)
    prod_df = pd.read_csv(production_logs_path)
    
    # CLEANING LAYER: Strip away hidden trailing spaces or quotes from column names
    train_df.columns = train_df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    prod_df.columns = prod_df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # Monitor the target price metric
    feature_name = 'predicted_price_pkr'
    
    ref_feature = train_df[feature_name]
    curr_feature = prod_df[feature_name]
    
    # Execute Two-Sample Kolmogorov-Smirnov distribution metric test
    statistic, p_value = ks_2samp(ref_feature, curr_feature)
    
    print(f"📊 Evaluation Metric Target : {feature_name}")
    print(f"🔬 Calculated K-S Statistic : {statistic:.4f}")
    print(f"📉 Calculated P-Value       : {p_value:.4f}")
    print("----------------------------------------------")
    
    # Statistical monitoring assertion barrier (Alpha threshold: 0.05)
    if p_value < 0.05:
        print("🛑 SYSTEM ALARM: Significant Data Drift Detected!")
        print("   Live market property evaluation metrics have shifted drastically")
        print("   away from your original baseline model training rules.")
    else:
        print("✅ SYSTEM NORMAL: No significant data drift identified.")
        print("   Live transaction data profiles match baseline parameters.")
    print("==============================================\n")

if __name__ == "__main__":
    evaluate_production_drift("training_baseline.csv", "production_inference_logs.csv")
