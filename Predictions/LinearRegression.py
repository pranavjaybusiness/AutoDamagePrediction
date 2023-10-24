import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
class AutoInsuranceFraud:
    def __init__(self, data_path, features):
        self.data_path = data_path
        self.features = features
        self.label_encoders = {}
        self.clf = DecisionTreeRegressor(random_state=0)  # Use Regressor

    def load_data(self):
        self.data = pd.read_csv(self.data_path, low_memory = False)
        self.original_target = self.data.columns[26]  # Store original target name
        self.X = self.data[self.features]
        self.y = self.data[self.original_target]  # Use original target name here
        le = LabelEncoder()
        self.y = le.fit_transform(self.y.fillna('Unknown'))  # Handle NaN values
        self.label_encoders[self.original_target] = le

    def encode_features(self):
        for feature in self.features:
            le = LabelEncoder()
            self.X.loc[:, feature] = le.fit_transform(self.X.loc[:, feature])
            self.label_encoders[feature] = le

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )

    def train(self):
        # Combine X_train and y_train to ensure they have the same number of rows.
        training_data = pd.concat([self.X_train, pd.Series(self.y_train, name=self.original_target)], axis=1)

        # Drop rows with NaN values.
        training_data.dropna(inplace=True)

        # Separate X_train and y_train.
        self.X_train = training_data[self.features]
        self.y_train = training_data[self.original_target]  # Use the original target name here
        self.clf.fit(self.X_train, self.y_train)

    def evaluate(self):
        try:
            y_pred = self.clf.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)  # Use mean squared error for evaluation
            var = np.var(y_pred)  # Calculate the variance of the predictions
            print(f"Mean Squared Error: {mse}")
            print(f"Variance of Predictions: {var}")  # Print the variance
        except Exception as e:
            print(f"An error occurred while evaluating the model: {e}")

    def predict(self, new_data):
        try:
            new_data_encoded = [
                self.label_encoders[self.features[i]].transform([new_data[i]])[0]
                for i in range(len(self.features))
            ]
            prediction = self.clf.predict([new_data_encoded])
            print(f"The predicted severity of the accident is: {prediction[0]}")
        except Exception as e:
            print(f"An error occurred: {e}")
            # print("Check if your new data contains only categories seen during training.")

    def check_data_types(self):
        print(self.X.dtypes)

# Usage
if __name__ == "__main__":
    features = ['ACRS Report Type','Collision Type', 'Weather',  'Driver At Fault','Vehicle First Impact Location',
                'Vehicle Second Impact Location','Vehicle Body Type','Vehicle Movement','Speed Limit','Vehicle Model']

    data_path = 'Data/NewYorkData.txt'
    model = AutoInsuranceFraud(data_path, features)

    model.load_data()
    # model.check_data_types()
    model.encode_features()
    model.split_data()
    # model.check_nan_values()
    model.train()
    model.evaluate()

    new_data = ['Property Damage Crash', 'STRAIGHT MOVEMENT ANGLE', 'CLEAR', 'Yes', 'TWELVE OCLOCK', 'TWELVE OCLOCK', 'PICKUP TRUCK', 'MAKING LEFT TURN', '15', 'F150']
    model.predict(new_data)