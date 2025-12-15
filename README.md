# FinancialNewsSentimentAnalysis
Streamlit application using Hugging Face deep learning models to analyze financial news. The app performs sentiment analysis with a fine-tuned transformer and generates concise summaries to support investment decision-making and efficient processing of unstructured financial text.

# Financial News Sentiment & Summarization App

## 📌 Project Overview

This repository contains a **Streamlit-based business application** that leverages **deep learning models from Hugging Face** to analyze financial news articles. The application performs **sentiment analysis** using a fine-tuned transformer model and generates concise **text summaries** to support faster and more informed financial decision-making.

The project is developed in compliance with the **ISOM5240 Group Project requirements**, including the use of multiple Hugging Face pipelines, model fine-tuning, and public deployment.

---

## 🧠 Application Functionality

The application integrates **two Hugging Face pipelines**, satisfying the requirement for a multi-pipeline deep learning system:

### 1. Financial Sentiment Analysis (Fine-Tuned Model)

* **Task:** Text Classification
* **Model:** Fine-tuned DistilBERT model
* **Dataset:** Financial sentiment dataset from Hugging Face
* **Output:**

  * Sentiment label (*positive, negative, or neutral*)
  * Confidence score for each sentiment class

This component enables users to quickly assess market sentiment embedded in financial news articles.

---

### 2. News Summarization (Pre-trained Model)

* **Task:** Text Summarization
* **Model:** `facebook/bart-large-cnn` (pre-trained)
* **Output:**

  * A concise summary highlighting the key points of the input financial news text

This feature reduces the cognitive load associated with reading long financial articles and improves information consumption efficiency.

---

## ⚙️ Code Workflow

1. **User Input**
   The user pastes a financial news article into the Streamlit interface.

2. **Model Loading**
   Hugging Face pipelines are loaded once using Streamlit caching to optimize runtime performance.

3. **Sentiment Inference**
   The fine-tuned sentiment model evaluates the input text and returns sentiment probabilities.

4. **Summary Generation**
   A pre-trained summarization model generates a short, readable summary of the news content.

5. **Results Display**
   The predicted sentiment, confidence score, runtime, and generated summary are displayed in the web application.

---

## 🌐 Deployment

* The application is deployed on **Streamlit Cloud** and is publicly accessible.
* All models used in the application are sourced exclusively from **Hugging Face**, in compliance with course requirements.
* Environment variables are supported for secure loading of private fine-tuned models.

---

## 📊 Business Value

This application demonstrates how deep learning can be applied to **financial text analytics** by:

* Identifying market sentiment from news articles in near real time
* Supporting investment and risk-related decision-making
* Automating the processing of unstructured financial information

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Hugging Face Transformers**
* **PyTorch**
* **DistilBERT (fine-tuned)**
* **BART (pre-trained)**

---

## 📁 Repository Structure

```
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔐 Notes

* If the fine-tuned model is private, a Hugging Face access token must be provided via Streamlit Cloud secrets.
* The application is optimized for CPU-based deployment environments.

---

## 📌 Course Context

This project was developed as part of the **ISOM5240 Group Project**, focusing on the practical application of deep learning techniques for real-world business problems using Python and Hugging Face.
