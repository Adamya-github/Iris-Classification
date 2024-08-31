from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Iris Flower Classification</title>
        </head>
        <body>
            <h1>Iris Flower Classification</h1>
            <input type="text" id="sepalLength" placeholder="Sepal Length">
            <input type="text" id="sepalWidth" placeholder="Sepal Width">
            <input type="text" id="petalLength" placeholder="Petal Length">
            <input type="text" id="petalWidth" placeholder="Petal Width">
            <button onclick="classifyFlower()">Classify Flower</button>
            <p id="result"></p>

            <script>
                function classifyFlower() {
                    const sepalLength = document.getElementById('sepalLength').value;
                    const sepalWidth = document.getElementById('sepalWidth').value;
                    const petalLength = document.getElementById('petalLength').value;
                    const petalWidth = document.getElementById('petalWidth').value;

                    fetch('/classify', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            'sepal length (cm)': parseFloat(sepalLength),
                            'sepal width (cm)': parseFloat(sepalWidth),
                            'petal length (cm)': parseFloat(petalLength),
                            'petal width (cm)': parseFloat(petalWidth),
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('result').textContent = `The flower is classified as: ${data.prediction}`;
                    });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    # Predict the class
    prediction = model.predict(df_scaled)
    return jsonify({'prediction': iris.target_names[prediction][0]})

if __name__ == '__main__':
    app.run(debug=True)
