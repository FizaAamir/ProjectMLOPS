from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the pre-trained models and label encoder
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier()
rf_classifier = RandomForestClassifier()
label_encoder = LabelEncoder()

# Load the dataset
fruits_df = pd.read_csv("fruits_dataset.csv")

# Map categorical features to numerical values
color_mapping = {'Red': 0, 'Green': 1, 'Yellow': 2, 'Orange': 3, 'Purple': 4}
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
shape_mapping = {'Round': 0, 'Cylindrical': 1, 'Oval': 2}
weight_mapping = {'Light': 0, 'Medium': 1, 'Heavy': 2}
label_mapping = {'Apple': 0, 'Banana': 1, 'Mango': 2, 'Orange': 3, 'Purple': 4}

fruits_df['Color'] = fruits_df['Color'].map(color_mapping)
fruits_df['Size'] = fruits_df['Size'].map(size_mapping)
fruits_df['Shape'] = fruits_df['Shape'].map(shape_mapping)
fruits_df['Weight'] = fruits_df['Weight'].map(weight_mapping)
fruits_df['Label'] = fruits_df['Label'].map(label_mapping)

# Features and target
X = fruits_df[['Color', 'Size', 'Shape', 'Weight']]
y = fruits_df['Label']

# Fit the models
dt_classifier.fit(X, y)
knn_classifier.fit(X, y)
rf_classifier.fit(X, y)

# Fit the label encoder
label_encoder.fit(y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    color = request.form.get('color')
    size = request.form.get('size')
    shape = request.form.get('shape')
    weight = request.form.get('weight')

    # Transform input to numerical values
    color = color_mapping[color]
    size = size_mapping[size]
    shape = shape_mapping[shape]
    weight = weight_mapping[weight]

    # Make a prediction using the models
    dt_prediction = label_encoder.inverse_transform(dt_classifier.predict([[color, size, shape, weight]]))[0]
    knn_prediction = label_encoder.inverse_transform(knn_classifier.predict([[color, size, shape, weight]]))[0]
    rf_prediction = label_encoder.inverse_transform(rf_classifier.predict([[color, size, shape, weight]]))[0]

    return render_template('index.html', dt_prediction=dt_prediction, knn_prediction=knn_prediction, rf_prediction=rf_prediction)

if __name__ == '__main__':
    app.run(debug=True)

