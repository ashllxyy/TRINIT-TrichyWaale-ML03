from os.path import exists
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from catboost.utils import eval_metric
import pickle

df_path = "Crop_recommendation.csv" if exists("Crop_recommendation.csv") else "https://github.com/ashllxyy/TRINIT-TrichyWaale-ML03/raw/main/Crop_recommendation.csv"
df = pd.read_csv(df_path, on_bad_lines='skip')
df = df.sample(n=2_000, random_state=0)

df["N"] = df["N"].astype(str)
df["P"] = df["P"].astype(str)
df["K"] = df["K"].astype(str)
df["temperature"] = df["temperature"].astype(str)
df["humidity"] = df["humidity"].astype(str)
df["ph"] = df["ph"].astype(str)
df["rainfall"] = df["rainfall"].astype(str)

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

train = df[df["temperature"] < "22.45145981"]
test = df[df["temperature"] >= "22.45145981"]

train_features = train.drop(columns=["label"])
train_target = train["label"]
test_features = train.drop(columns=["label"])
test_target = train["label"]

model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0)

model.fit(train_features, train_target)
preds = model.predict(test_features)
eval_metric(test_target.values, preds, "SMAPE")

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
