import os
import datetime

# Setup logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "judge_mathvista/logs")
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(
    logs_dir, f"judge_mathvista_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

def log_info(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - INFO - {message}"

    # Print to console
    print(log_message)

    # Write to file with immediate flush
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")
        f.flush()

# Initialize logging
# log_info("Logging initialized successfully") 