# This is our main app file that creates a customer support chatbot
# We'll use Streamlit to create a nice web interface, and Gemini AI to make smart responses

import streamlit as st  # Streamlit helps us create web apps easily

# Set up how our web page looks (this must be the first Streamlit command)
st.set_page_config(
    page_title="Customer Support Chatbot",  # The title that shows in your browser tab
    page_icon="🤖",  # The emoji that shows in your browser tab
    layout="wide"  # Makes our app use the full width of the screen
)

# Import all the tools we need
import pandas as pd  # For working with data tables
import numpy as np  # For doing math with numbers
from sklearn.feature_extraction.text import TfidfVectorizer  # Helps us understand text
import tensorflow as tf  # For our AI model
from tensorflow.keras.preprocessing.text import Tokenizer  # Helps prepare text for our AI
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Makes all text the same length
import pickle  # For saving and loading our AI models
import os  # For working with files and folders
import kagglehub  # For downloading our dataset
from sklearn.metrics.pairwise import cosine_similarity  # Helps find similar text
from collections import deque  # A special list that helps us remember recent things
import re  # For finding patterns in text
from datetime import datetime  # For working with dates and times
import google.generativeai as genai  # Google's AI that we'll use for responses
from typing import List, Dict  # Helps us write better code
from dotenv import load_dotenv  # For keeping our secret API key safe
import requests  # For downloading things from the internet
from requests.exceptions import ConnectTimeout, RequestException  # For handling internet problems
import time  # For adding delays when needed

# Load our secret API key from a hidden file
load_dotenv()

# Set up our connection to Google's AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Start up our AI model (we're using the free version)
try:
    model = genai.GenerativeModel('gemini-1.5-flash-8b')  # This is a smaller, faster version of the AI
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# Make our app look nice with custom styling
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;  /* Make the text input bigger */
    }
    .chat-message {
        padding: 1.5rem;  /* Add space around messages */
        border-radius: 0.5rem;  /* Make corners rounded */
        margin-bottom: 1rem;  /* Space between messages */
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        border: 1px solid #2b313e;  /* Style for user messages */
    }
    .chat-message.bot {
        border: 1px solid #475063;  /* Style for bot messages */
    }
    .context-info {
        font-size: 0.8em;  /* Make context info smaller */
        color: #888;  /* Make it gray */
        margin-top: 0.5rem;  /* Add space above it */
    }
    </style>
""", unsafe_allow_html=True)

# Set up our memory (this helps the chatbot remember the conversation)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # List to remember all messages
if 'dataset' not in st.session_state:
    st.session_state.dataset = None  # Will store our customer support data
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {
        'current_topic': None,  # What we're talking about
        'user_intent': None,  # What the user wants
        'follow_up_questions': deque(maxlen=3),  # Recent questions
        'last_response': None,  # When we last replied
        'start_time': datetime.now(),  # When the chat started
        'current_subject': None,  # What product we're discussing
        'subject_details': {},  # Details about the product
        'current_issue': None,  # What problem we're solving
        'issue_details': {},  # Details about the problem
        'last_entities': []  # Things we've talked about
    }
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'previous_queries': [],  # What the user has asked before
        'common_topics': set(),  # Topics they talk about most
        'query_count': 0  # How many questions they've asked
    }

# Function to get our customer support data
@st.cache_resource  # This makes it faster by remembering the data
def load_dataset():
    max_retries = 3  # Try 3 times if it fails
    retry_delay = 5  # Wait 5 seconds between tries
    
    for attempt in range(max_retries):
        try:
            # Try to download the data from Kaggle
            path = kagglehub.dataset_download("waseemalastal/customer-support-ticket-dataset")
            df = pd.read_csv(f"{path}/customer_support_tickets.csv", encoding='latin1')
            return df
        except (ConnectTimeout, RequestException) as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                st.error("Failed to download dataset from Kaggle. Using local fallback data.")
                # Try to use a local copy if we have one
                try:
                    df = pd.read_csv("customer_support_tickets.csv", encoding='latin1')
                    return df
                except FileNotFoundError:
                    st.error("""
                    Could not load the dataset. Please ensure you have:
                    1. A stable internet connection
                    2. The dataset file 'customer_support_tickets.csv' in your project directory
                    
                    You can download the dataset manually from:
                    https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset
                    """)
                    st.stop()

# Function to load our AI models
@st.cache_resource  # This makes it faster by remembering the models
def load_models():
    # Load all the AI models we need
    with open('nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)  # This model helps understand what type of question it is
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)  # This helps us find similar questions
    lstm_model = tf.keras.models.load_model('lstm_model.keras')  # This is our main AI model
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)  # This helps prepare text for our AI
    return nb_model, vectorizer, lstm_model, tokenizer

# Function to clean up text before giving it to our AI
def preprocess_text(text):
    if isinstance(text, str):
        # Make everything lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = ''.join([c for c in text if c.isalpha() or c.isspace()])
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove common words that don't help us understand
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        return text
    return ""

# Function to understand what the user wants
def recognize_intent(text, df):
    # Common things users might want
    intent_patterns = {
        'greeting': r'hi|hello|hey|greetings',  # When they say hello
        'farewell': r'bye|goodbye|see you|thanks',  # When they say goodbye
        'clarification': r'what do you mean|can you explain|i don\'t understand',  # When they need more info
        'urgency': r'urgent|asap|immediately|right now',  # When they need help fast
        'confirmation': r'yes|no|correct|right|wrong'  # When they're answering questions
    }
    
    # Check which pattern matches
    for intent, pattern in intent_patterns.items():
        if re.search(pattern, text.lower()):
            return intent
    
    return 'query'  # If no pattern matches, it's a regular question

# Function to keep track of what we're talking about
def update_conversation_context(text, prediction, similar_tickets):
    context = st.session_state.conversation_context
    
    # Remember what we're talking about
    context['current_topic'] = prediction
    
    # Remember what the user wants
    context['user_intent'] = recognize_intent(text, st.session_state.dataset)
    
    # Try to figure out what product they're talking about
    if context['current_subject'] is None:
        # Look for patterns like "my gopro" or "the gopro"
        product_patterns = [
            r'my\s+(\w+(?:\s+\w+)*)',  # "my gopro"
            r'the\s+(\w+(?:\s+\w+)*)',  # "the gopro"
            r'this\s+(\w+(?:\s+\w+)*)', # "this gopro"
            r'that\s+(\w+(?:\s+\w+)*)', # "that gopro"
            r'it\s+is\s+(\w+(?:\s+\w+)*)' # "it is gopro hero"
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, text.lower())
            if match:
                context['current_subject'] = match.group(1)
                context['subject_details'] = {
                    'type': 'product',
                    'first_mentioned': text,
                    'last_mentioned': text
                }
                break
    else:
        # Update when they last mentioned the product
        context['subject_details']['last_mentioned'] = text
    
    # Try to figure out what problem they're having
    issue_patterns = [
        r'not\s+connecting\s+to\s+(\w+)',  # "not connecting to wifi"
        r'issue\s+with\s+(\w+)',           # "issue with wifi"
        r'problem\s+with\s+(\w+)',         # "problem with wifi"
        r'(\w+)\s+not\s+working',          # "wifi not working"
        r'(\w+)\s+issue',                  # "wifi issue"
        r'(\w+)\s+problem'                 # "wifi problem"
    ]
    
    for pattern in issue_patterns:
        match = re.search(pattern, text.lower())
        if match:
            context['current_issue'] = match.group(1)
            context['issue_details'] = {
                'type': 'technical_issue',
                'first_mentioned': text,
                'last_mentioned': text
            }
            break
    
    # Remember when we last replied
    context['last_response'] = datetime.now()
    
    # Keep track of what they've asked before
    st.session_state.user_profile['previous_queries'].append(text)
    st.session_state.user_profile['common_topics'].add(prediction)
    st.session_state.user_profile['query_count'] += 1

# Function to find similar questions
def find_similar_tickets(query, df, vectorizer, top_k=3):
    # Turn the question into numbers our AI can understand
    query_vec = vectorizer.transform([preprocess_text(query)])
    
    # Turn all our known questions into numbers
    ticket_vecs = vectorizer.transform(df['processed_text'])
    
    # Find which questions are most similar
    similarities = cosine_similarity(query_vec, ticket_vecs).flatten()
    
    # Get the top 3 most similar questions
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return df.iloc[top_indices]

# Function to handle when the AI is too busy
def generate_with_retry(prompt, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                st.warning(f"Rate limit reached. Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Wait longer each time
            else:
                raise e

# Function to create a response using our AI
def generate_gemini_response(query: str, similar_tickets: pd.DataFrame, context: Dict) -> str:
    # Remove personal information from our data
    sensitive_columns = ['Customer Name', 'Customer Email', 'Customer Age', 'Customer Gender']
    clean_dataset = st.session_state.dataset.drop(columns=sensitive_columns, errors='ignore')
    clean_similar_tickets = similar_tickets.drop(columns=sensitive_columns, errors='ignore')
    
    # Prepare information about our data
    dataset_context = "Full Dataset Summary:\n"
    dataset_context += f"Total Records: {len(clean_dataset)}\n"
    
    # List all the types of information we have
    dataset_context += "\nAvailable Data Fields:\n"
    for column in clean_dataset.columns:
        dataset_context += f"- {column}\n"
    
    # Show some examples from each type of information
    dataset_context += "\nSample Data from Each Field:\n"
    sample_data = clean_dataset.sample(min(5, len(clean_dataset)))
    for column in clean_dataset.columns:
        unique_values = clean_dataset[column].dropna().unique()
        if len(unique_values) > 0:
            dataset_context += f"\n{column} Examples:\n"
            for value in unique_values[:5]:  # Show up to 5 examples
                dataset_context += f"- {value}\n"
    
    # Show how many of each type of ticket we have
    dataset_context += "\nData Distribution:\n"
    for column in clean_dataset.columns:
        if clean_dataset[column].dtype == 'object':  # For text columns
            value_counts = clean_dataset[column].value_counts()
            if len(value_counts) <= 10:  # Only show if not too many types
                dataset_context += f"\n{column} Distribution:\n"
                for value, count in value_counts.items():
                    dataset_context += f"- {value}: {count} records\n"
    
    # Show similar cases we found
    similar_records_context = "\nMost Relevant Records to Your Query:\n"
    for _, record in clean_similar_tickets.iterrows():
        similar_records_context += "\nRecord Details:\n"
        for column in clean_dataset.columns:
            if pd.notna(record[column]):
                similar_records_context += f"{column}: {record[column]}\n"
    
    # Create instructions for our AI
    prompt = f"""You are a customer support chatbot with access to our complete database. Use this information to provide accurate and helpful responses.

    Dataset Overview:
    {dataset_context}

    Most Relevant Records:
    {similar_records_context}

    User Question: {query}

    Current Context:
    - Topic: {context['current_topic']}
    - Intent: {context['user_intent']}
    - Current Subject: {context['current_subject'] if context['current_subject'] else 'None'}
    - Subject Details: {context['subject_details'] if context['subject_details'] else 'None'}
    - Current Issue: {context['current_issue'] if context['current_issue'] else 'None'}
    - Issue Details: {context['issue_details'] if context['issue_details'] else 'None'}
    - Previous Queries: {', '.join(context.get('previous_queries', [])[-3:])}

    Please provide a helpful response based on the data above. Consider:
    1. All available data fields and their values
    2. The specific examples most relevant to the user's query
    3. Common patterns and resolutions for similar issues
    4. If the data doesn't contain relevant information, acknowledge this and ask for more details.
    5. Maintain context about both the current subject (e.g., GoPro Hero) and the current issue (e.g., WiFi connection) being discussed.
    6. When the user provides new information, integrate it with the existing context rather than starting over.
    7. If the user mentions a specific issue, maintain that context throughout the conversation."""

    try:
        # Get a response from our AI
        response = generate_with_retry(prompt)
        
        # Get the text from the response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Add special formatting for urgent or unclear questions
        if context['user_intent'] == 'urgency':
            response_text = "🔴 " + response_text + "\n\nI understand this is urgent. I'll prioritize your request."
        elif context['user_intent'] == 'clarification':
            response_text = "Let me clarify: " + response_text
        
        return response_text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error. Please try asking your question again."

# The main part of our app
def main():
    st.title("🤖 Customer Support Chatbot")
    
    # Load our data if we haven't already
    if st.session_state.dataset is None:
        st.session_state.dataset = load_dataset()
        st.session_state.dataset['processed_text'] = st.session_state.dataset['Ticket Description'].apply(preprocess_text)
    
    # Create a sidebar with helpful information
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI-powered chatbot uses Gemini and advanced natural language processing to understand and respond to your queries.
        It learns from past interactions and adapts to your needs while staying grounded in our customer support data.
        """)
        
        st.header("Current Context")
        st.write(f"Topic: {st.session_state.conversation_context['current_topic'] or 'Not set'}")
        st.write(f"Intent: {st.session_state.conversation_context['user_intent'] or 'Not set'}")
        st.write(f"Session duration: {(datetime.now() - st.session_state.conversation_context['start_time']).seconds // 60} minutes")
        
        st.header("Common Topics")
        ticket_types = st.session_state.dataset['Ticket Type'].unique()
        for ticket_type in ticket_types:
            st.write(f"- {ticket_type}")
        
        st.header("Example Questions")
        example_tickets = st.session_state.dataset.sample(3)
        for _, ticket in example_tickets.iterrows():
            st.write(f"- {ticket['Ticket Description'][:100]}...")
    
    # Show the main chat interface
    st.write("How can I help you today?")
    
    # Get the user's question
    user_input = st.text_input("Type your message here...", key="user_input")
    
    if user_input:
        # Remember what the user said
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        try:
            # Load our AI models
            nb_model, vectorizer, lstm_model, tokenizer = load_models()
            
            # Figure out what type of question it is
            processed_text = preprocess_text(user_input)
            nb_features = vectorizer.transform([processed_text])
            nb_pred = nb_model.predict(nb_features)[0]
            
            # Use our main AI model to understand the question
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(sequence, maxlen=200)
            lstm_pred = np.argmax(lstm_model.predict(padded, verbose=0), axis=1)[0]
            
            # Choose the best prediction
            prediction = nb_pred if nb_pred == lstm_pred else lstm_pred
            similar_tickets = find_similar_tickets(user_input, st.session_state.dataset, vectorizer)
            
            # Update what we know about the conversation
            update_conversation_context(user_input, prediction, similar_tickets)
            
            # Get a response from our AI
            response = generate_gemini_response(user_input, similar_tickets, st.session_state.conversation_context)
            
            # Remember what we said
            st.session_state.chat_history.append({"role": "bot", "content": response})
            
            # Show similar cases that might help
            if not similar_tickets.empty:
                st.write("Similar cases that might help:")
                for _, ticket in similar_tickets.iterrows():
                    if pd.notna(ticket['Resolution']):
                        st.write(f"- {ticket['Resolution'][:200]}...")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append({
                "role": "bot",
                "content": "I apologize, but I encountered an error. Please try asking your question again."
            })
    
    # Show the chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user">
                    <div>👤 You:</div>
                    <div>{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot">
                    <div>🤖 Bot:</div>
                    <div>{message['content']}</div>
                    <div class="context-info">
                        Topic: {st.session_state.conversation_context['current_topic'] or 'Not set'} | 
                        Intent: {st.session_state.conversation_context['user_intent'] or 'Not set'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Start the app when this file is run
if __name__ == "__main__":
    main() 