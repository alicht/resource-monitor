import psutil
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import csv
import os
import random
import tensorflow as tf


# Function to train a simple TensorFlow model
def tensorflow_task(task_id, epochs=5):
    st.write(f"Starting TensorFlow Task {task_id}...")
    # Generate dummy training data
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(2, size=(1000, 1))

    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
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

# Initialize Streamlit app
st.title("Resource Allocator with Predictive Scaling")
st.write("This app monitors system resources, visualizes them, logs data, and runs TensorFlow tasks dynamically with predictive scaling.")

# Sidebar Configuration
st.sidebar.header("Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 2)
task_count = st.sidebar.number_input("Number of Tasks", min_value=1, max_value=20, value=5)

# Adjustable Thresholds
st.sidebar.subheader("Resource Thresholds")
cpu_threshold = st.sidebar.slider("CPU Threshold (%)", 0, 100, 70)
memory_threshold = st.sidebar.slider("Memory Threshold (%)", 0, 100, 80)
gpu_threshold = st.sidebar.slider("GPU Utilization Threshold (%)", 0, 100, 70)
gpu_memory_threshold = st.sidebar.slider("GPU Memory Threshold (%)", 0, 100, 80)

# Initialize CSV file for logging
log_file = "resource_logs.csv"
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

        # Predict future resource usage
        predicted_cpu = predict_resources(st.session_state.cpu_data)
        predicted_memory = predict_resources(st.session_state.memory_data)
        predicted_gpu = predict_resources(st.session_state.gpu_data)
        predicted_gpu_memory = predict_resources(st.session_state.gpu_memory_data)

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

        # Display Resources and Predictions
        resource_placeholder.write({
            "Current Resources": resources,
            "Predicted Resources": {
                "CPU": predicted_cpu,
                "Memory": predicted_memory,
                "GPU": predicted_gpu,
                "GPU Memory": predicted_gpu_memory,
            }
        })

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

        # Task Execution Check Against Predicted Thresholds
        if (
            predicted_cpu < cpu_threshold and
            predicted_memory < memory_threshold and
            predicted_gpu < gpu_threshold and
            predicted_gpu_memory < gpu_memory_threshold
        ):
            tensorflow_task(task_id, duration)
        else:
            st.warning(f"Task {task_id} delayed: Predicted resources exceed thresholds.")
        
        # Add delay based on update interval
        time.sleep(update_interval)

    st.success("All tasks completed!")
    st.info(f"Resource usage data has been saved to `{log_file}`.")

# Historical Logs Section
st.subheader("Historical Logs")
if st.button("View Historical Logs"):
    if os.path.exists(log_file):
        # Load the CSV file
        df = pd.read_csv(log_file)

        # Display Data Table
        st.write("**Logged Resource Data:**")
        st.dataframe(df)

        # Display Trend Graphs
        st.write("**Historical Trend Graphs:**")
        fig, ax = plt.subplots()
        ax.plot(df["Time"], df["CPU Usage (%)"], label="CPU Usage (%)")
        ax.plot(df["Time"], df["Memory Usage (%)"], label="Memory Usage (%)")
        ax.plot(df["Time"], df["GPU Utilization (%)"], label="GPU Utilization (%)")
        ax.plot(df["Time"], df["GPU Memory Usage (%)"], label="GPU Memory Usage (%)")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Usage (%)")
        ax.set_title("Resource Usage Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No historical logs found.")
