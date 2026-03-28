import pandas as pd
import numpy as np 
import pickle

# ⚠️ Load your dataset (UPDATE PATH if needed)
df = pd.read_csv('laptop_data.csv')

# ---------------- PREPROCESSING ---------------- #

# Example (must match your original training logic)
df['Ram'] = df['Ram'].str.replace('GB','').astype(int)
df['Weight'] = df['Weight'].str.replace('kg','').astype(float)

# Extract features (same as your model training)
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Resolution split
df['X_res'] = df['ScreenResolution'].apply(lambda x: int(x.split()[-1].split('x')[0]))
df['Y_res'] = df['ScreenResolution'].apply(lambda x: int(x.split()[-1].split('x')[1]))

df['ppi'] = ((df['X_res']**2 + df['Y_res']**2)**0.5) / df['Inches']

# CPU brand
df['Cpu brand'] = df['Cpu'].apply(lambda x: "Intel" if "Intel" in x else ("AMD" if "AMD" in x else "Other"))

# GPU brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])

# OS
df['os'] = df['OpSys'].apply(lambda x: "Windows" if "Windows" in x else ("Mac" if "Mac" in x else ("Linux" if "Linux" in x else "Other")))
# ---------------- MEMORY PROCESSING ---------------- #

# Remove spaces
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')

# Split
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = new[0]
df["first"] = df["first"].str.strip()

df["second"] = new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '', regex=True)

df["second"].fillna("0", inplace=True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '', regex=True)

# Convert to int
df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

# Final HDD & SSD
df["HDD"] = (df["first"] * df["Layer1HDD"] +
             df["second"] * df["Layer2HDD"])

df["SSD"] = (df["first"] * df["Layer1SSD"] +
             df["second"] * df["Layer2SSD"])

# ---------------- FEATURES ---------------- #

X = df[['Company','TypeName','Ram','Weight','Touchscreen','Ips','ppi','Cpu brand','HDD','SSD','Gpu brand','os']]
y = np.log(df['Price'])

# ---------------- MODEL ---------------- #

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

step1 = ColumnTransformer([
    ('col_tnf', OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'   # 🔥 ADD THIS
    ), ['Company','TypeName','Cpu brand','Gpu brand','os'])
], remainder='passthrough')

step2 = RandomForestRegressor()

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X, y)

# ---------------- SAVE FILES ---------------- #

pickle.dump(pipe, open('pipe.pkl','wb'))
pickle.dump(X.columns.tolist(), open('columns.pkl','wb'))

# ✅ IMPORTANT: options
options = {
    "Company": sorted(X['Company'].unique()),
    "TypeName": sorted(X['TypeName'].unique()),
    "Cpu brand": sorted(X['Cpu brand'].unique()),
    "Gpu brand": sorted(X['Gpu brand'].unique()),
    "os": sorted(X['os'].unique())
}

pickle.dump(options, open('options.pkl','wb'))

print("✅ All files saved successfully!")