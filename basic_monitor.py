# import psutil

# # Monitor CPU Usage
# cpu_usage = psutil.cpu_percent(interval=1)
# print(f"CPU Usage: {cpu_usage}%")

# # Monitor Memory Usage
# memory = psutil.virtual_memory()
# print(f"Memory Usage: {memory.percent}%")

# # Monitor Disk Usage
# disk = psutil.disk_usage('/')
# print(f"Disk Usage: {disk.percent}%")


# import psutil
# import time

# while True:
#     # Monitor CPU Usage
#     cpu_usage = psutil.cpu_percent(interval=1)
#     print(f"CPU Usage: {cpu_usage}%")

#     # Monitor Memory Usage
#     memory = psutil.virtual_memory()
#     print(f"Memory Usage: {memory.percent}%")

#     # Monitor Disk Usage
#     disk = psutil.disk_usage('/')
#     print(f"Disk Usage: {disk.percent}%")

#     # Wait for 1 second before the next update
#     time.sleep(1)

import psutil
import time

# Open a file to log the data
with open("resource_log.txt", "a") as log_file:
    while True:
        # Gather stats
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Format the data
        log_entry = (
            f"CPU: {cpu_usage}%, "
            f"Memory: {memory.percent}%, "
            f"Disk: {disk.percent}%\n"
        )
        print(log_entry.strip())  # Print to console
        log_file.write(log_entry)  # Write to file

        # Wait for 1 second before the next update
        time.sleep(1)
