import psutil
from queue import Queue
import threading
import time
import tensorflow as tf
import numpy as np
import random  # For mocking GPU metrics

# Simulated TensorFlow workload
def tensorflow_task(task_id, duration=30):
    print(f"Starting TensorFlow Task {task_id}...")

    # Dummy TensorFlow model training
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(2, size=(1000, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Simulate training for a given duration (number of epochs)
    model.fit(x_train, y_train, epochs=duration, batch_size=32, verbose=0)

    print(f"Task {task_id} completed.")

# Function to check system CPU, memory, and MOCKED GPU resources
def check_resources():
    return {
        "cpu": psutil.cpu_percent(interval=0),
        "memory": psutil.virtual_memory().percent,
        "gpu": random.randint(0, 70),  # Mocked GPU utilization (0% - 70%)
        "gpu_memory": random.randint(0, 80)  # Mocked GPU memory usage (0% - 80%)
    }

# Resource allocator function
def resource_allocator(task_queue, cpu_threshold=70, memory_threshold=80, gpu_threshold=70, gpu_memory_threshold=80):
    while not task_queue.empty():
        resources = check_resources()
        print(f"Resources: CPU {resources['cpu']}%, Memory {resources['memory']}%, GPU {resources['gpu']}%, GPU Memory {resources['gpu_memory']}%")

        # Check if system and mocked GPU resources are available
        if (resources["cpu"] < cpu_threshold and
            resources["memory"] < memory_threshold and
            resources["gpu"] < gpu_threshold and
            resources["gpu_memory"] < gpu_memory_threshold):

            # Start TensorFlow task
            task_id, duration = task_queue.get()
            threading.Thread(target=tensorflow_task, args=(task_id, duration)).start()
        else:
            print("Resources too high, waiting...")
        time.sleep(2)

# Create a queue of TensorFlow tasks
task_queue = Queue()
for i in range(5):  # 5 tasks with a duration of 5 epochs each
    task_queue.put((f"Task-{i+1}", 5))

# Start the resource allocator
if __name__ == "__main__":
    try:
        print("Starting Resource Allocator with TensorFlow Workloads and Mocked GPU Metrics...")
        resource_allocator(task_queue)
    except KeyboardInterrupt:
        print("\nResource Allocator stopped by user.")
