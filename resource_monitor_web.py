# Standard Library Imports
import os
import time
import random
import csv
from datetime import datetime
import traceback

# Third-Party Library Imports
import psutil
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is properly set
if not api_key:
    raise ValueError("API key not found. Please ensure OPENAI_API_KEY is set in your .env file.")

# Set the API key for OpenAI
openai.api_key = api_key

# Function to train a simple TensorFlow model
def tensorflow_task(task_id, epochs=5):
    st.write(f"Starting TensorFlow Task {task_id}...")
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(2, size=(1000, 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    st.success(f"Task {task_id} completed!")

# Function to check system resources
def check_resources():
    return {
        "CPU Usage (%)": psutil.cpu_percent(interval=0),
        "Memory Usage (%)": psutil.virtual_memory().percent,
        "GPU Utilization (%)": random.randint(0, 70),  # Mocked GPU metric
        "GPU Memory Usage (%)": random.randint(0, 80)  # Mocked GPU memory
    }

# Function to predict future resource usage
def predict_resources(data, steps=5):
    if len(data) < steps:
        return np.mean(data)  # If not enough data, use current mean
    return np.mean(data[-steps:])  # Predict using the last `steps` values

# Function to track OpenAI API usage
def track_openai_usage(prompt, model="gpt-3.5-turbo"):
    try:
        # OpenAI ChatCompletion request
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
        )

        # Extract response text and usage details
        response_text = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']
        cost = calculate_cost(total_tokens, model)

        return {
            "response_text": response_text.strip(),
            "total_tokens": total_tokens,
            "cost": cost,
        }
    except Exception as e:
        st.error(f"An error occurred while calling the OpenAI API: {e}")
        st.text(traceback.format_exc())
        return None

# Function to calculate cost based on token usage
def calculate_cost(total_tokens, model):
    if model == "gpt-3.5-turbo":
        cost_per_1k_tokens = 0.002  # $0.002 per 1,000 tokens
    elif model == "gpt-4":
        cost_per_1k_tokens = 0.03  # $0.03 per 1,000 tokens
    else:
        cost_per_1k_tokens = 0.001  # Default pricing for other models
    return (total_tokens / 1000) * cost_per_1k_tokens

# Function to log API usage
log_file_llm = "llm_usage_logs.csv"
def log_llm_usage(prompt, tokens, cost):
    with open(log_file_llm, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), prompt, tokens, cost])

# Initialize Streamlit app
st.title("Resource Allocator with Predictive Scaling")
st.write("This app monitors system resources, visualizes them, logs data, and runs TensorFlow tasks dynamically with predictive scaling.")

# Sidebar Configuration
st.sidebar.header("Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 2)
task_count = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=20, value=5)

# Sidebar Input for LLM Prompts
st.sidebar.subheader("LLM Prompt")
prompt = st.sidebar.text_area("Enter your LLM prompt:")
if st.sidebar.button("Send Prompt"):
    if prompt:
        result = track_openai_usage(prompt)
        if result:
            st.subheader("OpenAI API Response")
            st.write(result["response_text"])

            st.subheader("Usage Details")
            st.write(f"Total Tokens Used: {result['total_tokens']}")
            st.write(f"Cost of Request: ${result['cost']:.4f}")

            log_llm_usage(prompt, result["total_tokens"], result["cost"])
        else:
            st.error("Failed to retrieve response from OpenAI API.")
    else:
        st.warning("Please enter a prompt.")

# Historical Logs Section for LLM API Usage
if st.sidebar.button("View API Usage Logs"):
    if os.path.exists(log_file_llm):
        df_llm = pd.read_csv(log_file_llm, names=["Timestamp", "Prompt", "Tokens", "Cost"])
        st.subheader("Logged LLM API Usage")
        st.dataframe(df_llm)

        st.subheader("LLM Usage Trends")
        fig, ax = plt.subplots()
        ax.plot(df_llm["Tokens"], label="Tokens Used")
        ax.legend()
        ax.set_xlabel("Request Index")
        ax.set_ylabel("Tokens")
        ax.set_title("Tokens Used Per Request")
        st.pyplot(fig)
    else:
        st.warning("No LLM API usage logs found.")
