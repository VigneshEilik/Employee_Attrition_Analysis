# Employee Attrition Analysis & Prediction Dashboard
📌 Project Overview

This project analyzes employee attrition trends and predicts at-risk employees using machine learning. It provides HR teams with data-driven insights to improve retention strategies through an interactive Streamlit dashboard.

# The system integrates:

Exploratory Data Analysis (EDA) of HR datasets

Attrition Prediction Model (Logistic Regression/ML model)

Performance Prediction Model

Streamlit Dashboard for real-time insights & HR reports

# ⚙️ Tech Stack

Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)

Streamlit (Interactive dashboards)

Pickle (Model serialization)

Jupyter Notebook (EDA & preprocessing)

# 📂 Project Structure

Employee_Attrition_Analysis/
|  data/
|   │── attrition_cleaned.csv          # Cleaned dataset
|   │── Employee_attrition_analysis.ipynb # Data cleaning, EDA, modeling
|  model/
|   │── attrition_model.pkl            # Trained attrition prediction model
|   │── performance_model.pkl          # Trained performance prediction model
│── app.py                         # Streamlit dashboard
│── Employee_Attrition_Project_Report.pdf # Project documentation
│── README.md                      # Project description

# 🚀 Features

Attrition Trends: Visualize attrition by department, job role, and demographics

Predict Attrition: Identify employees at risk of leaving

Predict Performance: Forecast employee performance ratings

Business Insights: Attrition rate, estimated cost of attrition, department-level analysis

Downloadable Reports: HR can export reports as CSV

# 📊 Dashboard Preview

Home Page – Overview of attrition and key metrics

Attrition Prediction – Predict which employees are likely to leave

Performance Prediction – Predict employee performance ratings

Business Insights – Attrition cost analysis and HR report download

# 🧠 Model Evaluation

Attrition Model

Algorithm: Logistic Regression / Classification model

Accuracy: ~80%

Key Features: JobRole, Overtime, MonthlyIncome, YearsAtCompany, Age

Performance Model

Algorithm: RandomForestClassifier

Accuracy: ~75%

Predicts: Employee performance rating

# ▶️ How to Run the Dashboard

Clone this repo:

 
* git clone https://github.com/yourusername/Employee_Attrition_Analysis.git
* cd Employee_Attrition_Analysis


Create a virtual environment and install dependencies:

* pip install -r requirements.txt

* Run Streamlit app:

* streamlit run app.py


Open the link in your browser (default: http://localhost:8501)

# 📥 Download HR Reports

The dashboard includes an option to download CSV reports for HR review.

# 📌 Business Value

✅ Identify employees most likely to leave
✅ Reduce turnover costs with proactive HR actions
✅ Optimize workforce planning
✅ Provide data-driven retention strategies

# 📜 License

This project is licensed under the MIT License – feel free to use and modify with attribution.

🔥 With this dashboard, HR teams can transform raw data into actionable insights and make smarter retention decisions.