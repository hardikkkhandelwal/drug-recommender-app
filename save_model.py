import pandas as pd
import ast
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix

# Load
df = pd.read_csv("indian_pharmaceutical_products_cleaned_data.csv", encoding='latin-1')
df["ai_names"] = df["active_ingredients"].apply(lambda x: [i["name"] for i in ast.literal_eval(x)])

# --- THE SHRINK: Only keep drugs that actually have alternatives ---
# We group by ingredients, form, and strength. If a group has only 1 drug, 
# it has no alternatives, so we drop it from the app's database.
df['match_key'] = df.apply(lambda x: f"{sorted(x['ai_names'])}-{x['dosage_form']}-{x['primary_strength']}", axis=1)
counts = df['match_key'].value_counts()
useful_keys = counts[counts > 1].index
df_mini = df[df['match_key'].isin(useful_keys)].copy()

# Drop the helper key and heavy columns
df_mini = df_mini[["brand_name", "manufacturer", "dosage_form", "primary_strength", "price_inr", "ai_names", "primary_ingredient"]]

# --- TRAIN SMALLER MODEL ---
mlb = MultiLabelBinarizer()
X = csr_matrix(mlb.fit_transform(df_mini["ai_names"]))
y = df_mini["primary_ingredient"]

# Use fewer estimators and limit depth to keep the .pkl tiny
model = RandomForestClassifier(n_estimators=30, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save
df_mini.to_parquet("medicine_data.parquet", compression='brotli') # Brotli is stronger than Snappy
joblib.dump({"model": model, "mlb": mlb}, "model_tools.pkl", compress=9) # Max compression

print(f"New Data Size: {len(df_mini)} rows")