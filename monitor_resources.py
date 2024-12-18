# import psutil
# import time

# def monitor_resources(interval=1):
#     while True:
#         # CPU usage
#         cpu_usage = psutil.cpu_percent(interval=0)
#         print(f"CPU Usage: {cpu_usage}%")

#         # Memory usage
#         memory = psutil.virtual_memory()
#         print(f"Memory Usage: {memory.percent}%")

#         # Disk usage
#         disk = psutil.disk_usage('/')
#         print(f"Disk Usage: {disk.percent}%")

#         # Add a delay for the next reading
#         time.sleep(interval)
#         print("\n--- Resource Stats Updated ---\n")

# # Run the monitoring function
# if __name__ == "__main__":
#     try:
#         monitor_resources(interval=2)  # Update every 2 seconds
#     except KeyboardInterrupt:
#         print("\nMonitoring stopped by user.")

# import psutil
# import time

# def monitor_resources(interval=1, output_file="resource_log.txt"):
#     with open(output_file, "w") as f:
#         while True:
#             # Collect metrics
#             cpu_usage = psutil.cpu_percent(interval=0)
#             memory = psutil.virtual_memory()
#             disk = psutil.disk_usage('/')

#             # Format the output
#             log_line = (
#                 f"CPU Usage: {cpu_usage}%\n"
#                 f"Memory Usage: {memory.percent}%\n"
#                 f"Disk Usage: {disk.percent}%\n"
#                 f"---\n"
#             )

#             # Print and save to file
#             print(log_line)
#             f.write(log_line)
#             time.sleep(interval)

# if __name__ == "__main__":
#     try:
#         monitor_resources(interval=2)
#     except KeyboardInterrupt:
#         print("\nMonitoring stopped by user.")


import psutil
import time

def monitor_resources(interval=1):
    while True:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0)
        print(f"CPU Usage: {cpu_usage}%")

        # Memory usage
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}%")

        # Disk usage
        disk = psutil.disk_usage('/')
        print(f"Disk Usage: {disk.percent}%")

        # Add a delay for the next reading
        time.sleep(interval)
        print("\n--- Resource Stats Updated ---\n")

# Run the monitoring function
if __name__ == "__main__":
    try:
        monitor_resources(interval=2)  # Update every 2 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
