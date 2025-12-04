import os
import datetime

# Setup logging
# Check if log file path is already set (e.g., from PBS script)
log_file_path = os.environ.get("EVAL_RUN_LOG_FILE")

if not log_file_path:
    # Create new log file if not already specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(
        logs_dir, f"bon_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    # Export the log file path as environment variable for telegram notifications
    os.environ["EVAL_RUN_LOG_FILE"] = log_file_path
else:
    # Ensure the directory exists for the provided log file path
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

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