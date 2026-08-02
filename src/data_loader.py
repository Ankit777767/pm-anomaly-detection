#AI4I 2020 Predictive Maintenance dataset
from pathlib import Path
import pandas as pd 

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"

COLUMN_RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

FEATURE_COLUMNS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
#y hai
# Columns that are LABELS -- evaluation only, never fed to a model's fit()
LABEL_COLUMNS = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]

def load_raw(path: Path=RAW_PATH)->pd.DataFrame:
  df = pd.read_csv(path)
  df = df.rename(columns=COLUMN_RENAME)
  df["udi"] = df["udi"].astype(int)
  return df

def split_features_labels(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
  features = df[["udi","product_id"]+FEATURE_COLUMNS].copy()
  labels = df[["udi"]+LABEL_COLUMNS].copy()
  return features,labels

def basic_sanity_checks(df:pd.DataFrame)->dict:
  """Quick structural checks worth printing/logging during EDA."""
  return {
    "n_rows": len(df),
    "n_duplicates": int(df.duplicated(subset=["udi"]).sum()),
    "n_missing_total": int(df.isna().sum().sum()),
    "failure_rate_pct": round(100 * df["machine_failure"].mean(), 2),
    "type_distribution": df["type"].value_counts(normalize=True).round(3).to_dict(),# It will normalize with sum=1 across al samples between valued 0 1o 1.0
  }
  
if __name__ == "__main__":
  data = load_raw()#will get the dataframe
  feats,labels = split_features_labels(data)
  print("Loaded:", data.shape)
  print("Feature columns:", feats.columns.tolist())
  print("Label columns:", labels.columns.tolist())
  print(basic_sanity_checks(data))
