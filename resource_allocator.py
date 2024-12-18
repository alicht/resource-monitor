import psutil
from queue import Queue
import threading
import time
from simulate_tensorflow_workload import tensorflow_training_task  # Import the TensorFlow task

# Check available resources
def check_resources():
    return {
        "cpu": psutil.cpu_percent(interval=0),
        "memory": psutil.virtual_memory().percent
    }

# Allocate tasks based on resource thresholds
def resource_allocator(task_queue, cpu_threshold=70, memory_threshold=80):
    while not task_queue.empty():
        resources = check_resources()
        print(f"Resources: CPU {resources['cpu']}%, Memory {resources['memory']}%")

        # If resources are below thresholds, start a task
        if resources["cpu"] < cpu_threshold and resources["memory"] < memory_threshold:
            task_id, task_params = task_queue.get()
            threading.Thread(target=tensorflow_training_task, args=(task_id, *task_params)).start()
        else:
            print("Resources too high, waiting...")
        time.sleep(2)

# Create a queue of TensorFlow tasks
task_queue = Queue()
for i in range(3):  # 3 TensorFlow tasks
    # Each task has (task_id, (epochs, batch_size)) as parameters
    task_queue.put((f"Task-{i+1}", (10, 32)))

# Start the resource allocator
if __name__ == "__main__":
    resource_allocator(task_queue)
