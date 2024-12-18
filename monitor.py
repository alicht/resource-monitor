import psutil

# Monitor CPU usage
print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

# Monitor memory usage
memory = psutil.virtual_memory()
print(f"Memory Usage: {memory.percent}%")

# Monitor disk usage
disk = psutil.disk_usage('/')
print(f"Disk Usage: {disk.percent}%")
