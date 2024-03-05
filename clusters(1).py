import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify,send_file
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import BytesIO
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    try:
        # Read the uploaded CSV file
        file = request.files['file']
        if file:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            print(df.head())
            # Data preprocessing (cleaning and scaling)
            cleaned_data = preprocess_data(df)

            # Perform clustering on selected columns
            columns = request.form['columns'].split(',')
            selected_columns = cleaned_data[columns]
            print(selected_columns)
            x = selected_columns.iloc[:, 0].values
            y = selected_columns.iloc[:, 1].values
            data = list(zip(x, y))

            # Perform K-Means clustering with the user-selected number of clusters
            num_clusters = int(request.form['num_clusters'])
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data)

            # Get cluster labels
            cluster_labels = kmeans.labels_.tolist()

            return jsonify({'cluster_labels': cluster_labels})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot', methods=['POST'])
def plot_clusters():
    try:
        # Read the uploaded CSV file
        file = request.files['file']
        if file:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            columns = request.form['columns'].split(',')
            cleaned_data = preprocess_data(df)
            selected_columns = cleaned_data[columns]
            inertias=[]
            x = selected_columns.iloc[:, 0].values
            y = selected_columns.iloc[:, 1].values
            data = list(zip(x, y))
            for i in range(1,11):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)

            plt.plot(range(1,11), inertias, marker='o')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            img_buffer1 = BytesIO()
            plt.savefig(img_buffer1, format='png')
            img_buffer1.seek(0)            
            num_clusters = int(request.form['num_clusters'])
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(list(zip(x, y)))
            plt.scatter(x, y, c=kmeans.labels_)
            plt.title('K-Means Clustering')
            plt.xlabel('Column 1')
            plt.ylabel('Column 2')
            img_buffer2 = BytesIO()
            plt.savefig(img_buffer2, format='png')
            img_buffer2.seek(0)
            return send_file(img_buffer1, mimetype='image/png')
            return send_file(img_buffer2, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_data(df):
    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Define data transformations for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    # Apply transformations to columns based on data type
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns)
        ], remainder = 'passthrough')

    # Fit and transform the data
    preprocessed_data = preprocessor.fit_transform(df)

    # Convert the preprocessed data back to a DataFrame
    df_preprocessed = pd.DataFrame(preprocessed_data)

    return df_preprocessed

if __name__== '__main__':
    app.run(debug=False)