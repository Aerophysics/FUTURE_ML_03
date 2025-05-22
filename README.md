# FUTURE_ML_03

# 🤖 Project 3: Customer Support Chatbot

This project implements a customer support chatbot using both Naive Bayes and LSTM models for ticket classification. The chatbot is accessible through a Streamlit web interface.

### Features

- Dual model approach (Naive Bayes and LSTM) for ticket classification
- Real-time chat interface using Streamlit
- Support for various types of customer inquiries
- Beautiful and responsive UI
- Chat history tracking

### Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model training script first:
```bash
python chatbot.py
```

3. Launch the Streamlit app:
```bash
streamlit run app.py
```

### Usage

1. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
2. Type your customer support query in the text input field
3. The chatbot will process your query and provide a response based on the classification

### Project Structure

- `chatbot.py`: Script for training and saving the machine learning models
- `app.py`: Streamlit web interface for the chatbot
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

### Model Information

The chatbot uses two models for classification:
1. Naive Bayes: Fast and efficient for text classification
2. LSTM: Deep learning model for better understanding of context and sequence

The models are trained on a customer support ticket dataset and can classify queries into different categories such as:
- Technical Support
- Billing Issues
- Product Information
- Account Management
- General Inquiries
