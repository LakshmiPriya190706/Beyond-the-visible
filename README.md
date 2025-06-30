#  Vestibular Schwannoma Confidence Scoring Using Deep Learning and Questionnaire

This project implements a hybrid system for estimating the **confidence score** of **Vestibular Schwannoma (VS)** tumors using a **Convolutional Neural Network (CNN)** on MRI scans, combined with a lightweight **clinical symptom questionnaire**.

---

## Project Highlights

-  **Confidence Score** prediction (0–100) for tumor severity
-  **MRI Analysis** via trained CNN model
-  **Questionnaire Integration** for symptom-based inputs
-  Hybrid scoring: `90% CNN + 10% Symptoms`
-  Evaluation metrics: MAE, MSE, R², and Visualization

---

Symptom Questionnaire Integration
Hearing loss, Dizziness, Tinnitus, Facial weakness

Implemented as a form (Flask optional)

Scored and normalized for model integration

Confidence Score Calculation
python
Copy
Edit
Final Score = (0.9 × CNN_Predicted_Score) + (0.1 × Questionnaire_Score)

