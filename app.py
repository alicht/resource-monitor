import streamlit as st
import psutil
import pynvml
import time

# Initialize NVIDIA NVML for GPU monitoring
try:
    pynvml.nvmlInit()
    gpu_available = True
except:
    gpu_available = False
    st.warning("No NVIDIA GPU detected. GPU stats will not be displayed.")

# Function to get GPU stats
def get_gpu_stats():
    if gpu_available:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return gpu_utilization, memory_info.used / memory_info.total * 100
    return None, None

# Streamlit UI
st.title("Real-Time Resource Monitoring")

# Metrics display
cpu_usage_placeholder = st.empty()
ram_usage_placeholder = st.empty()
gpu_util_placeholder = st.empty()
gpu_mem_placeholder = st.empty()

st.write("Monitoring system resources in real-time. Refreshes every second.")

while True:
    # Get CPU and RAM usage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    ram_usage = memory.percent

    # Get GPU stats
    gpu_util, gpu_mem = get_gpu_stats()

    # Update metrics
    cpu_usage_placeholder.metric("CPU Usage", f"{cpu_usage}%")
    ram_usage_placeholder.metric("RAM Usage", f"{ram_usage}%")
    if gpu_available:
        gpu_util_placeholder.metric("GPU Utilization", f"{gpu_util}%")
        gpu_mem_placeholder.metric("GPU Memory Usage", f"{gpu_mem:.2f}%")
    else:
        st.info("GPU stats are not available on this system.")

    # Wait for a short interval to refresh
    time.sleep(1)
