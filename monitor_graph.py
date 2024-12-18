# import psutil
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Lists to store data
# cpu_data = []
# memory_data = []

# # Update function for the graph
# def update(frame):
#     cpu_data.append(psutil.cpu_percent(interval=0))
#     memory_data.append(psutil.virtual_memory().percent)

#     # Keep only the last 50 data points
#     cpu_data[:] = cpu_data[-50:]
#     memory_data[:] = memory_data[-50:]

#     plt.cla()
#     plt.plot(cpu_data, label="CPU Usage (%)")
#     plt.plot(memory_data, label="Memory Usage (%)")
#     plt.ylim(0, 100)
#     plt.legend(loc="upper right")
#     plt.tight_layout()

# # Set up the real-time graph
# ani = FuncAnimation(plt.gcf(), update, interval=1000)

# plt.show()


import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Lists to store data
cpu_data = []
memory_data = []
disk_data = []

# Update function for the graph
def update(frame):
    # Append data to the lists
    cpu_data.append(psutil.cpu_percent(interval=0))
    memory_data.append(psutil.virtual_memory().percent)
    disk_data.append(psutil.disk_usage('/').percent)

    # Keep only the last 50 data points for smooth real-time updates
    cpu_data[:] = cpu_data[-50:]
    memory_data[:] = memory_data[-50:]
    disk_data[:] = disk_data[-50:]

    # Clear the current axes and plot new data
    plt.cla()
    plt.plot(cpu_data, label="CPU Usage (%)", linestyle="--")
    plt.plot(memory_data, label="Memory Usage (%)", linestyle="-")
    plt.plot(disk_data, label="Disk Usage (%)", linestyle=":")
    plt.ylim(0, 100)  # All metrics are in percentages
    plt.legend(loc="upper right")
    plt.title("Real-Time Resource Usage Monitoring")
    plt.xlabel("Time (most recent)")
    plt.ylabel("Usage (%)")
    plt.grid(True)
    plt.tight_layout()

    # Save the current frame to a file (optional)
    plt.savefig("monitor_graph.png")

# Set up the real-time graph
def run_monitor(interval=1000):
    # Create the animation
    ani = FuncAnimation(plt.gcf(), update, interval=interval)

    # Display the graph
    plt.show()

# Run the monitoring tool
if __name__ == "__main__":
    try:
        # Set the update interval in milliseconds (default is 1000 ms = 1 second)
        run_monitor(interval=1000)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
