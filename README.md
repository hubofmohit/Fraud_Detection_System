# Fraud Detection System

This machine learning system detects fraudulent credit card transactions. It uses anomaly detection with Isolation Forest and supervised classification with Logistic Regression, enhanced by SMOTE. The system is built with Python and real-world financial data from Kaggle.

## Project Structure

```
fraud_detection_project/
├── fraud_detection.py         # Main script for model training and evaluation
├── logistic_model.pkl         # Saved Logistic Regression model
├── scaler.pkl                 # StandardScaler for amount normalization
├── creditcard.csv             # Dataset from Kaggle (not included in repo)
├── README.md                  # This documentation
```

## Requirements

Install the necessary Python packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:** Contains 284,807 transactions, with only 492 frauds (about 0.17%)
- **Note:** Download and place `creditcard.csv` in your project directory.

## How to Run

1. **Download the dataset** from Kaggle and place it in the project folder.
2. **Run the Python script:**

```bash
python fraud_detection.py
```

3. **Output will include:**
   - F1 Score and AUC-ROC for Isolation Forest
   - Logistic Regression performance with SMOTE
   - Classification report
   - ROC Curve plot

## Techniques Used

| Technique            | Description |
|----------------------|-------------|
| Isolation Forest     | Unsupervised anomaly detection |
| SMOTE                | Balances classes by oversampling the minority (fraud) class |
| Logistic Regression  | Supervised binary classification |
| StandardScaler       | Normalizes the `Amount` feature |
| ROC Curve            | Visual performance evaluation |

## Evaluation Metrics

- **F1 Score**
- **AUC-ROC**
- **Confusion Matrix**
- **ROC Curve**

## Model Saving

The script saves the trained model and scaler:

- `logistic_model.pkl` – Trained classifier
- `scaler.pkl` – Scaler for future data preprocessing

You can load these later to make predictions in real-time or serve them through an API.

## Future Enhancements

- Real-time prediction API using FastAPI
- Deep learning with Autoencoders for better anomaly detection
- Streamlit dashboard to visualize fraud trends
- Dockerize the pipeline for deployment

## Author

**Mohit Garg**  
📧 [Email - gargmohit0104@gmail.com]  
🔗 [GitHub Profile](https://github.com/hubofmohit)

## License

This project is open-source and free to use for educational and research purposes.

## CSV File
https://drive.google.com/file/d/1nntEYz4ccY12QGJvwnbLErxge6TIJDq_/view?usp=drive_link
