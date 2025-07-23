from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from django.http import JsonResponse
import os
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder


def home_view(request):
    return render(request, 'home.html')

def process_view(request):
    return render(request, 'process.html')


def train_and_save_model(csv_path, model_path):
    df = pd.read_csv(csv_path)

    # Fill missing values
    df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0], inplace=True)
    df['Parental_Education_Level'].fillna(method='ffill', inplace=True)
    df['Distance_from_Home'].fillna(method='ffill', inplace=True)

    # Columns
    categorical_columns = ['Extracurricular_Activities','Internet_Access','School_Type',
                           'Peer_Influence','Learning_Disabilities','Parental_Education_Level','Gender']
    ordinal_columns = ['Parental_Involvement','Access_to_Resources','Motivation_Level',
                       'Teacher_Quality','Family_Income','Distance_from_Home']
    numeric_columns = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                       'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']

    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]

    # Ordinal encoder
    ordinal_encoder = OrdinalEncoder(categories=[
        ['Low', 'Medium', 'High'],       # Parental_Involvement
        ['Low', 'Medium', 'High'],       # Access_to_Resources
        ['Low', 'Medium', 'High'],       # Motivation_Level
        ['Low', 'Medium', 'High'],       # Teacher_Quality
        ['Low', 'Medium', 'High'],       # Family_Income
        ['Near', 'Moderate', 'Far']      # Distance_from_Home
    ])

    # Full preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numeric_columns)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', GradientBoostingRegressor())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    joblib.dump(model_pipeline, model_path)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#3498db')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Exam Scores')
    plt.grid(True)
    plt.show()

    return {
        "r2_score": r2,
        "mae": mae,
        "rmse": rmse,
        
    }





def train_model_view(request):
    csv_path = os.path.join(settings.MEDIA_ROOT, 'StudentPerformanceFactors.csv')
    model_path = os.path.join('home', 'student_model.pkl')

    metrics = train_and_save_model(csv_path, model_path)

    return render(request, 'training_results.html', {
        'r2_score': f"{metrics['r2_score']:.3f}",
        'mae': f"{metrics['mae']:.2f}",
        'rmse': f"{metrics['rmse']:.2f}",
        
    })

from django.core.files.storage import default_storage


from django.http import FileResponse

def predict_bulk_view(request):
    if request.method == 'POST' and request.FILES.get('data_file'):
        data_file = request.FILES['data_file']
        
        file_path = os.path.join(settings.MEDIA_ROOT, data_file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in data_file.chunks():
                destination.write(chunk)

        try:
            df = pd.read_csv(file_path)
            model_path = os.path.join('home', 'student_model.pkl')
            model = joblib.load(model_path)

            if 'Exam_Score' in df.columns:
                df = df.drop(columns=['Exam_Score'])

            predictions = model.predict(df)
            df['Predicted_Exam_Score'] = predictions

            os.remove(file_path)

            # Save predicted result to CSV for download
            result_csv_name = "prediction_results.csv"
            result_csv_path = os.path.join(settings.MEDIA_ROOT, result_csv_name)
            df.to_csv(result_csv_path, index=False)

            return render(request, 'bulk_predictions.html', {
                'columns': df.columns,
                'rows': df.to_dict(orient='records'),
                'download_url': f'/download-result/{result_csv_name}'
            })

        except Exception as e:
            return render(request, 'bulk_predictions.html', {'error': str(e)})
    else:
        return render(request, 'bulk_predictions.html')
    


from django.http import FileResponse

def download_result_view(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    else:
        return JsonResponse({'error': 'File not found'}, status=404)    