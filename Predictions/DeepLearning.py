import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)  # Set output to num_classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


class AutoInsuranceDamage:

    def __init__(self, data_path, features):
        self.data_path = data_path
        self.features = features
        self.label_encoders = {}
        self.num_classes = 8  # Set the number of classes here
        self.model = NeuralNetwork(len(features), self.num_classes)
        self.SEVERITY_MAPPING = {
            0: "NO DAMAGE",
            1: "SUPERFICIAL",
            2: "FUNCTIONAL",
            3: "DESTROYED",
            4: "DISABLING"
        }

    def load_data(self):
        # Read the data
        self.data = pd.read_csv(self.data_path, low_memory=False)

        # Filter out rows with 'Vehicle Damage Extent' as 'N/A' or 'UNKNOWN'
        self.data = self.data[self.data['Vehicle Damage Extent'].isin(['N/A', 'UNKNOWN']) == False]

        # Assuming the 26th column is the target variable
        self.original_target = self.data.columns[26]

        # Features and Target
        self.X = self.data[self.features]
        self.target = self.data[self.original_target]

        # Label encode the target variable
        le = LabelEncoder()
        self.y = le.fit_transform(self.target)
        self.label_encoders[self.original_target] = le

        # Dynamically determine the number of unique labels
        self.num_classes = len(np.unique(self.y))

        # Adjust the Neural Network's output layer to match the number of classes
        self.model = NeuralNetwork(len(self.features), self.num_classes)

    def encode_features(self):
        for feature in self.features:
            le = LabelEncoder()
            self.X.loc[:, feature] = le.fit_transform(self.X.loc[:, feature])
            self.label_encoders[feature] = le

        # After encoding, ensure the data is numeric
        self.X = self.X.apply(pd.to_numeric, errors='coerce')

    def split_data(self):
        # Handling NaN values here after encoding and before splitting
        self.X = self.X.fillna(0)  # Filling NaN with 0 (or use other strategies)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )

    def train(self, epochs=10, batch_size=64, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_data = TensorDataset(torch.tensor(self.X_train.values).float(), torch.tensor(self.y_train).long())
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(self.X_test.values).float())
            y_pred_classes = torch.argmax(y_pred, axis=1)
            print(classification_report(self.y_test, y_pred_classes))

    def predict(self, new_data):
        self.model.eval()
        with torch.no_grad():
            new_data_encoded = [
                self.label_encoders[self.features[i]].transform([new_data[i]])[0]
                for i in range(len(self.features))
            ]
            prediction = self.model(torch.tensor(new_data_encoded).float())
            predicted_class = torch.argmax(prediction).item()
            print(f"The predicted severity of the accident is: {self.SEVERITY_MAPPING[predicted_class]}")


# Usage
if __name__ == "__main__":
    features = ['ACRS Report Type','Collision Type','Driver At Fault','Speed Limit','Vehicle Body Type',
                'Vehicle Model','Vehicle Movement',
                'Weather']

    data_path = 'Data/NewYorkData.txt'
    model = AutoInsuranceDamage(data_path, features)

    model.load_data()
    model.encode_features()
    model.split_data()
    model.train(epochs=10, batch_size=64, lr=0.001)
    model.evaluate()

    new_data = ['Property Damage Crash','SAME DIR REAR END','Yes','65','PASSENGER CAR','TK','MOVING CONSTANT SPEED','CLEAR']
    model.predict(new_data)
