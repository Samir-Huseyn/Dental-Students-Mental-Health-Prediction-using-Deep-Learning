import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

df = pd.read_csv("Dent.csv")
print(df.info())
print(df.head(3))
df = df.drop(columns = ["id"])

X = df.drop(columns = ["cesd"]).values
y = df["cesd"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
print(X_train.shape)

model = Sequential([
    Dense(units=18, activation="relu", input_shape = (18,)),
    Dense(units=9, activation="relu"),
    Dense(units=3, activation="relu"),
    Dense(units=1)
])

model.compile(optimizer = "adam", loss = "mse", metrics = ["mae"])
model.fit(X_train, y_train, epochs = 30, batch_size = 16, validation_data = (X_test, y_test))
loss, mae = model.evaluate(X_test, y_test)
print(loss)
print(mae)