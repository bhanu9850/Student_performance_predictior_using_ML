# Student_performance_predictior_using_ML

A Django-based web application that uses a machine learning pipeline to predict student exam scores based on multiple academic, personal, and socio-economic factors. Users can train the model with historical data and upload CSV files to get bulk predictions.

---

##  Features

- Train a regression model (Gradient Boosting Regressor) on student performance data.
- Bulk prediction via CSV file upload.
- Preprocessing includes:
  - Handling missing values
  - Ordinal & one-hot encoding
  - Feature scaling
- Displays evaluation metrics: R² Score, MAE, RMSE
- Downloadable prediction results in CSV format
- Visual plots for model performance
- Informative UI with a homepage, training page, and prediction workflow

##  Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/bhanu9850/Student_performance_predictior_using_ML.git
   cd eduinsight
2.  **Install dependencies**
pip install -r requirements.txt

4. **Run migrations**
python manage.py migrate

5. **Start the server**
python manage.py runserver

##  How to Use
Go to http://localhost:8000/

Train the model using the sample CSV dataset.

Navigate to Bulk Prediction and upload a properly formatted .csv file.

View predicted scores in the table.

Option to download prediction results.

**Sample Input Format**

The CSV file should have the following columns:
Parental_Involvement,Access_to_Resources,Motivation_Level,Teacher_Quality,
Family_Income,Distance_from_Home,Extracurricular_Activities,Internet_Access,
School_Type,Peer_Influence,Learning_Disabilities,Parental_Education_Level,
Gender,Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,
Tutoring_Sessions,Physical_Activity

**Model Performance**

**After training:**

R² Score: 0.87+

MAE: ~3.5

RMSE: ~4.2
Visuals and evaluation available under the training results.

 **Pages Overview**
 
/ → Project Introduction

/train-model/ → Train the machine learning model

/predict-bulk/ → Upload CSV and get predicted scores

/process/ → See how the ML pipeline is built

Let me know if you want:
- A `requirements.txt`
- A GitHub badge-style README (build status, license, etc.)
- A section on deployment (Heroku, Render, or Docker)  

I'm here to help wrap it up perfectly!
