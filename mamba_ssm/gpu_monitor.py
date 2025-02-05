import os
import time
from datetime import datetime

# Define log file name
LOG_FILE = "nvidia_smi_log.txt"

# Function to fetch and log nvidia-smi output
def log_nvidia_smi():
    try:
        # Get nvidia-smi output
        output = os.popen("nvidia-smi").read()
        last_lines = "\n".join(output.splitlines()[-8:])
        
        # Timestamp for the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append output to the log file
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"\n[{timestamp}]\n")
            log_file.write(last_lines)
            log_file.write("\n" + "-"*80 + "\n")
    except Exception as e:
        print(f"Error logging nvidia-smi: {e}")

# Main loop to log every 10 minutes
if __name__ == "__main__":
    while True:
        log_nvidia_smi()
        time.sleep(600)
