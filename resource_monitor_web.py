# Standard Library Imports
import psutil
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import random
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in your .env file.")
openai.api_key = openai_api_key

# Function to simulate a TensorFlow-like task
def simulate_task(task_id, duration=5):
    with st.spinner(f"Running Task {task_id}..."):
        time.sleep(duration)  # Simulate task running time
    st.success(f"Task {task_id} completed!")

# Function to check system resources
def check_resources():
    return {
        "CPU Usage (%)": psutil.cpu_percent(interval=0),
        "Memory Usage (%)": psutil.virtual_memory().percent,
        "GPU Utilization (%)": random.randint(0, 70),  # Mocked GPU metric
        "GPU Memory Usage (%)": random.randint(0, 80)  # Mocked GPU memory
    }

# Function to track OpenAI API usage
def track_openai_usage(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response["choices"][0]["message"]["content"]
        usage = response["usage"]
        total_tokens = usage.get("total_tokens", 0)
        cost = calculate_cost(total_tokens, model)
        return {
            "response_text": response_text,
            "total_tokens": total_tokens,
            "cost": cost
        }
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

# Function to calculate API cost
def calculate_cost(total_tokens, model):
    if model == "gpt-3.5-turbo":
        cost_per_1k_tokens = 0.002
    elif model == "gpt-4":
        cost_per_1k_tokens = 0.03
    else:
        cost_per_1k_tokens = 0.001
    return (total_tokens / 1000) * cost_per_1k_tokens

# Initialize Streamlit app
st.title("Resource Monitor with OpenAI Integration")
st.write("This app monitors system resources, logs data, and integrates with OpenAI API.")

# Sidebar Configuration
st.sidebar.header("Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 2)
task_count = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=20, value=5)
log_file = "resource_logs.csv"
log_file_llm = "llm_usage_logs.csv"

# Initialize CSV file for resource logging
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "CPU Usage (%)", "Memory Usage (%)", "GPU Utilization (%)", "GPU Memory Usage (%)"])

# Session State Initialization
if "task_queue" not in st.session_state:
    st.session_state.task_queue = [(f"Task-{i+1}", 5) for i in range(task_count)]

if "cpu_data" not in st.session_state:
    st.session_state.cpu_data = []
    st.session_state.memory_data = []
    st.session_state.gpu_data = []
    st.session_state.gpu_memory_data = []

# Real-Time Resource Graphs
st.subheader("Live Resource Usage")
resource_placeholder = st.empty()
chart_placeholder = st.empty()

# Task Execution Button
if st.button("Start Task Execution"):
    for task_id, duration in st.session_state.task_queue:
        # Gather Resource Data
        resources = check_resources()
        st.session_state.cpu_data.append(resources["CPU Usage (%)"])
        st.session_state.memory_data.append(resources["Memory Usage (%)"])
        st.session_state.gpu_data.append(resources["GPU Utilization (%)"])
        st.session_state.gpu_memory_data.append(resources["GPU Memory Usage (%)"])

        # Write data to CSV for historical logging
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                resources["CPU Usage (%)"],
                resources["Memory Usage (%)"],
                resources["GPU Utilization (%)"],
                resources["GPU Memory Usage (%)"]
            ])

        # Limit data points to the last 50 for clean graphs
        st.session_state.cpu_data = st.session_state.cpu_data[-50:]
        st.session_state.memory_data = st.session_state.memory_data[-50:]
        st.session_state.gpu_data = st.session_state.gpu_data[-50:]
        st.session_state.gpu_memory_data = st.session_state.gpu_memory_data[-50:]

        # Display Resources
        resource_placeholder.write(resources)

        # Generate Live Graph
        fig, ax = plt.subplots()
        ax.plot(st.session_state.cpu_data, label="CPU Usage (%)")
        ax.plot(st.session_state.memory_data, label="Memory Usage (%)")
        ax.plot(st.session_state.gpu_data, label="GPU Utilization (%)")
        ax.plot(st.session_state.gpu_memory_data, label="GPU Memory Usage (%)")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time")
        ax.set_ylabel("Usage (%)")
        ax.set_title("System Resource Usage (Real-Time)")

        # Update the graph
        chart_placeholder.pyplot(fig)

        # Simulate task execution
        simulate_task(task_id, duration)

        # Add delay based on update interval
        time.sleep(update_interval)

    st.success("All tasks completed!")
    st.info(f"Resource usage data has been saved to `{log_file}`.")

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
            with open(log_file_llm, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now(), prompt, result["total_tokens"], result["cost"]])
        else:
            st.error("Failed to retrieve response from OpenAI API.")
    else:
        st.warning("Please enter a prompt.")

# Historical Logs Section
if st.sidebar.button("View Historical Logs"):
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.subheader("Resource Usage Logs")
        st.dataframe(df)

    if os.path.exists(log_file_llm):
        df_llm = pd.read_csv(log_file_llm, names=["Timestamp", "Prompt", "Tokens", "Cost"])
        st.subheader("OpenAI API Usage Logs")
        st.dataframe(df_llm)
    else:
        st.warning("No logs found.")
