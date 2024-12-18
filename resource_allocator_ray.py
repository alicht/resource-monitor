import psutil
import ray
import time
import numpy as np
import tensorflow as tf

# Initialize Ray
ray.init()

# Define TensorFlow task as a remote function
@ray.remote
def tensorflow_task(task_id, duration=30):
    print(f"Starting TensorFlow Task {task_id}...")
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(2, size=(1000, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=duration, batch_size=32, verbose=0)
    print(f"Task {task_id} completed.")
    return f"Task {task_id} completed."

# Resource monitor
def check_resources():
    return {"cpu": psutil.cpu_percent(interval=0), "memory": psutil.virtual_memory().percent}

# Allocate tasks with Ray
def resource_allocator_ray(num_tasks=5, cpu_threshold=70, memory_threshold=80):
    futures = []
    for i in range(num_tasks):
        resources = check_resources()
        print(f"Resources: CPU {resources['cpu']}%, Memory {resources['memory']}%")

        # Allocate tasks only when resources are available
        if resources["cpu"] < cpu_threshold and resources["memory"] < memory_threshold:
            task_id = f"Task-{i+1}"
            futures.append(tensorflow_task.remote(task_id, duration=5))
            print(f"Submitted {task_id}")
        else:
            print("Resources too high, waiting...")
            time.sleep(2)

    # Wait for all tasks to complete
    ray.get(futures)

if __name__ == "__main__":
    print("Starting Resource Allocator with Ray...")
    resource_allocator_ray(num_tasks=5)
