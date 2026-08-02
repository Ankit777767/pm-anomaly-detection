#Physics informed Feature Engineering
"""Each engineered feature is tied to one of the AI4I 2020 failure-mode
definitions (Matzka, 2020), so every column here maps to a concrete
argument:

    power_w          -> Power Failure (PWF) fires when power < 3500W or
                         > 9000W. Power isn't in the raw data -- it has to
                         be derived from torque and rotational speed.
    temp_diff_k       -> Heat Dissipation Failure (HDF) fires when the
                         air/process temperature difference is small AND
                         rotational speed is low (poor heat dissipation).
    torque_wear_nm_min-> Overstrain Failure (OSF) fires when tool_wear *
                         torque crosses a type-dependent threshold.
    rotational_speed_rad_s -> intermediate unit conversion, kept since it's
                         the physically correct unit for power calc.
"""

import numpy as np
import pandas as pd
from data_loader import load_raw,split_features_labels


RPM_TO_RAD_S = 2 * np.pi / 60

def add_feature_engg(features:pd.DataFrame)->pd.DataFrame:
  
  df = features.copy()#deep copy st og never changes
  
  df["rotational_speed_rad_s"] = df["rotational_speed_rpm"] * RPM_TO_RAD_S
  
  # Ties directly to the PWF failure mode (power outside [3500, 9000] W).
  df["power_w"] = df["torque_nm"] * df["rotational_speed_rad_s"]
  
  # Temperature differential -- ties to HDF (heat dissipation failure),
  # which is driven by *how close* process and air temperature are,
  # not their absolute values.
  df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
  
  # Cumulative mechanical stress proxy -- ties to OSF (overstrain failure),
  # which is defined as tool_wear * torque crossing a type-dependent
  # threshold (11,000 / 12,000 / 13,000 min*Nm for L/M/H).
  df["wear_torque_product"] = df["tool_wear_min"] * df["torque_nm"]

  return df

def encode_type(df: pd.DataFrame) -> pd.DataFrame:
  """One-hot encode the categorical product quality variant (L/M/H).
  Kept separate from add_engineered_features so ablation configs can
  toggle it independently (raw vs. engineered vs. +type-encoded).
  """
  return pd.get_dummies(df, columns=["type"], prefix="type", dtype=int)

FEATURE_SETS = {
    # For ablation study Phase 3: compare how much engineered features
    # actually help vs. raw sensor readings alone.
    "raw": ["air_temp_k", "process_temp_k", "rotational_speed_rpm",
            "torque_nm", "tool_wear_min"],
    "engineered": ["air_temp_k", "process_temp_k", "rotational_speed_rpm",
                   "torque_nm", "tool_wear_min", "power_w", "temp_diff_k",
                   "wear_torque_product"],
}


if __name__ == "__main__":
  data=load_raw()
  feats, labels = split_features_labels(data)
  feats_added = add_feature_engg(feats)
  print(feats_added[["power_w", "temp_diff_k", "wear_torque_product"]].describe())


