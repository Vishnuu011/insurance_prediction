import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("/content/insurance.csv")
data.head()

data.tail()

data.info()

data.isnull().sum()

data.describe()

sns.histplot(data.age, kde=True)
plt.title(f"distribution of age column | skewness: {data.age.skew()}")
plt.show()

data.select_dtypes(include=np.number).columns

sns.histplot(data.bmi, kde=True)
plt.title(f"distribution of bmi column | skewness: {data.bmi.skew()}")
plt.show()

sns.histplot(data.children, kde=True)
plt.title(f"distribution of children column | skewness: {data.children.skew()}")
plt.show()

sns.histplot(data.charges, kde=True)
plt.title(f"distribution of charges column | skewness: {data.charges.skew()}")
plt.show()

for i in data.select_dtypes(include=np.number).columns.tolist():
    sns.boxplot(data[i])
    plt.title(f"boxplot of {i} column")
    plt.show()
    print("-"*100)

data.select_dtypes(include=np.number).columns.tolist()

col=['age', 'bmi', 'children']

for i in col:

    sns.regplot(
    x=data[i],
    y="charges",
    data=data,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
    )

    plt.title("Children vs Insurance Charges")
    plt.show()
    print("-"*100)

X = data.drop(columns="charges", axis=1)
y_log = np.log1p(data["charges"])


################################################################################
# Model building, data spliting, data preprocessing, and model compile and fiting 
################################################################################


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from typing import  List, Tuple
import joblib
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class InsuranceRegressionModel:

    def __init__(self, X, y_log):

        self.X = X
        self.y_log = y_log


    def data_split(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y_log,
            test_size=0.2,
            random_state=42
        )

        return X_train, X_test, y_train, y_test


    def preprocessor(self) -> ColumnTransformer:

        num_features = ["age", "bmi", "children"]
        cat_features = ["sex", "smoker", "region"]

        preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ), cat_features)
        ]
        )

        return preprocessor


    def build_model(self, input_dim: int) -> tf.keras.models.Model:

        model = Sequential([
            Dense(
                64,
                activation="relu",
                input_shape=(input_dim,)
            ),
            Dense(
                32,
                activation="relu"
            ),
            Dense(1)
        ])

        return model


    def compaile_and_fit(self, epochs: int) -> tf.keras.models.Model:

        X_train, X_test, y_train, y_test = self.data_split()
        preprocessor = self.preprocessor()


        X_train_processed = preprocessor.fit_transform(X_train)
        input_dim = X_train_processed.shape[1]

        print(f"shape: {input_dim}")


        model = self.build_model(input_dim=input_dim)

        joblib.dump(preprocessor, "preprocessor.pkl")

        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=0.001),
            metrics=["mae", "r2_score"]
        )

        hist = model.fit(
            X_train_processed,
            y_train,
            validation_data=(
                preprocessor.transform(X_test),
                y_test
            ),
            epochs=epochs
        )

        model.save("model.h5")

        return model, hist
    
#################################################################
# Model training
#################################################################


model_obj = InsuranceRegressionModel(X, y_log)

model, hist = model_obj.compaile_and_fit(epochs=70)


#################################################################
# Model r2_score, mae, loss ploting 
#################################################################


fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Loss
axes[0].plot(hist.history["loss"], label="Train")
axes[0].plot(hist.history["val_loss"], label="Val")
axes[0].set_title("Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

# MAE
axes[1].plot(hist.history["mae"], label="Train")
axes[1].plot(hist.history["val_mae"], label="Val")
axes[1].set_title("MAE")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

# R2
axes[2].plot(hist.history["r2_score"], label="Train")
axes[2].plot(hist.history["val_r2_score"], label="Val")
axes[2].set_title("R² Score")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

#################################################################
# Model r2_score, mae, rmsle test
#################################################################

X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_log,
            test_size=0.2,
            random_state=42
        )

preprocessosr = joblib.load("preprocessor.pkl")
X_test_transformed = preprocessosr.transform(X_test)

y_pred_log = model.predict(X_test_transformed).ravel()
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score

r2 = r2_score(y_true, y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

print("MAE:", mae)
print("RMSLE:", rmsle)
print("R² Score:", r2)



#################################################################
# Model prediction 
#################################################################


age = int(input("Enter the age: "))
sex = input("Enter the sex (male/female): ")
bmi = float(input("Enter the BMI: "))
children = int(input("Enter the number of children: "))
smoker = input("Enter the smoking status (yes/no): ")
region = input("Enter the region (northeast/northwest/southeast/southwest): ")

dicts = {
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region]
}

data_new = pd.DataFrame(dicts)
preprocessosr = joblib.load("preprocessor.pkl")
pred_data_transformed = preprocessosr.transform(data_new)
y_pred_log = model.predict(pred_data_transformed).ravel()
y_pred = np.expm1(y_pred_log)

print(f"predicted Charges is : {y_pred}")

