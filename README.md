# Bitcoin Flash Crash Prediction 🚨📉

This project uses deep learning and sentiment analysis to predict flash crashes in Bitcoin prices using Reddit discussions and BTC OHLCV data.

---

## 🧠 Objective

To detect rapid price drops (flash crashes) by combining:
- Reddit sentiment scores (VADER + FinBERT)
- Bitcoin price/volume data
- Deep learning models: LSTM, BiLSTM, and TCN

---

## 🧪 Approach

- Custom label creation for hard and soft flash crashes
- Aggregated Reddit sentiment at minute-level
- Created sequence windows ending in crash/no-crash
- Balanced class ratio (1 crash : 2 non-crash)
- Evaluated using precision, recall, F1, and ROC-AUC

---

## 🔧 Tech Stack

`Python` • `TensorFlow` • `Pandas` • `VADER` • `FinBERT` • `scikit-learn` • `Matplotlib`

---

## 📁 Files in This Repo

Balancing_Data_Prep.ipynb – Your data preprocessing script
BTC_Finbert_Model.ipynb – FinBERT-based model notebook
BTC_Vader_model.ipynb – VADER-based model notebook
final_report.md- Final report and analysis 


---

## 📝 Note

Due to size, data files and model weights are excluded. Contact for reproducible setup.
