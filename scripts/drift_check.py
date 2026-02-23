import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from config import DATABASE_URL, REPORTS_DIR
from utils import load_df_from_postgres
from schema import ALL_FEATURES

os.makedirs(REPORTS_DIR, exist_ok=True)

ref = load_df_from_postgres("reference_data", DATABASE_URL)[ALL_FEATURES]
cur = load_df_from_postgres("new_data", DATABASE_URL)[ALL_FEATURES]

# --- Validation Guards ---
if ref.empty:
    raise ValueError(f"reference_data table is empty. Insert data before running drift check.")

if cur.empty:
    raise ValueError(f"new_data table is empty. Insert data before running drift check.")

if len(ref) < 30:
    print(f"⚠️  Warning: reference_data only has {len(ref)} rows. Drift results may be unreliable. Recommend 50+ rows.")

if len(cur) < 30:
    print(f"⚠️  Warning: new_data only has {len(cur)} rows. Drift results may be unreliable. Recommend 50+ rows.")

for c in ALL_FEATURES:
    if c not in ref.columns:
        raise ValueError(f"Column '{c}' missing from reference_data. Available: {list(ref.columns)}")
    if c not in cur.columns:
        raise ValueError(f"Column '{c}' missing from new_data. Available: {list(cur.columns)}")
    if ref[c].dropna().empty:
        raise ValueError(f"Column '{c}' in reference_data is all NULL/empty.")
    if cur[c].dropna().empty:
        raise ValueError(f"Column '{c}' in new_data is all NULL/empty.")

print(f"✅ Validation passed — ref: {len(ref)} rows, cur: {len(cur)} rows")
# --- End Validation ---

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

out_path = os.path.join(REPORTS_DIR, "drift_report.html")
report.save_html(out_path)

print(f"✅ Drift report saved: {out_path}")