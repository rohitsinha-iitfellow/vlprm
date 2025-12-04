#!/usr/bin/env python3
"""
Test script for Telegram notifications.

Usage:
    # Set environment variables first:
    export TELEGRAM_BOT_TOKEN="your_bot_token_here"
    export TELEGRAM_CHAT_ID="@your_channel_or_chat_id"
    
    # Run test:
    python test_telegram_notification.py
    
    # Or run with custom values:
    python test_telegram_notification.py --bot-token "YOUR_TOKEN" --chat-id "@channel"
"""

import os
import sys
import argparse
import datetime
import json
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from send_telegram_notifications_helper import send_telegram_job_summary


def create_dummy_json_file():
    """Create a dummy JSON results file for testing."""
    dummy_data = {
        "test_results": {
            "accuracy": 0.85,
            "total_samples": 100,
            "correct": 85,
            "incorrect": 15
        },
        "metadata": {
            "dataset": "test_dataset",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    
    dummy_file = Path("/tmp/test_evaluation_results.json")
    with open(dummy_file, "w") as f:
        json.dump(dummy_data, f, indent=2)
    
    return str(dummy_file)


def create_dummy_log_file():
    """Create a dummy log file for testing."""
    log_content = f"""
[{datetime.datetime.now()}] Test log file
[INFO] Starting evaluation...
[INFO] Processing sample 1/100
[INFO] Processing sample 50/100
[INFO] Processing sample 100/100
[INFO] Evaluation complete!
[INFO] Accuracy: 85%
"""
    
    dummy_log = Path("/tmp/test_evaluation.log")
    with open(dummy_log, "w") as f:
        f.write(log_content)
    
    return str(dummy_log)


def test_basic_message(bot_token=None, chat_id=None):
    """Test sending a basic message without files."""
    print("\n=== Test 1: Basic Message (no files) ===")
    
    try:
        send_telegram_job_summary(
            model_path_name="TestModel/v1.0",
            evaluation_results_json_file=None,
            evaluation_run_logs_file=None,
            bot_token=bot_token,
            chat_id=chat_id,
            separator="\t",
            include_header=True,
            send_files=False,
            message_prefix="🧪 TEST MODE - Basic Message",
            extra_fields={
                "test_type": "basic",
                "status": "testing"
            }
        )
        print("✅ Basic message sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send basic message: {e}")
        return False


def test_message_with_files(bot_token=None, chat_id=None):
    """Test sending message with file attachments."""
    print("\n=== Test 2: Message with Files ===")
    
    # Create dummy files
    json_file = create_dummy_json_file()
    log_file = create_dummy_log_file()
    
    print(f"Created dummy JSON: {json_file}")
    print(f"Created dummy log: {log_file}")
    
    try:
        send_telegram_job_summary(
            model_path_name="TestModel/v1.0-with-files",
            evaluation_results_json_file=json_file,
            evaluation_run_logs_file=log_file,
            bot_token=bot_token,
            chat_id=chat_id,
            separator="\t",
            include_header=True,
            send_files=True,
            message_prefix="🧪 TEST MODE - Message with Files",
            extra_fields={
                "test_type": "with_files",
                "accuracy": "85%",
                "dataset": "test_dataset"
            }
        )
        print("✅ Message with files sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send message with files: {e}")
        return False


def test_spreadsheet_format(bot_token=None, chat_id=None):
    """Test spreadsheet-friendly format with different separators."""
    print("\n=== Test 3: Spreadsheet Format ===")
    
    # Test with tab separator
    print("Testing TAB separator...")
    try:
        send_telegram_job_summary(
            model_path_name="Qwen2.5-VL-7B",
            evaluation_results_json_file="/path/to/results.json",
            evaluation_run_logs_file="/path/to/logs.log",
            bot_token=bot_token,
            chat_id=chat_id,
            separator="\t",
            include_header=False,  # No header for cleaner spreadsheet paste
            send_files=False,
            extra_fields={
                "accuracy": "92.5",
                "dataset": "mmmu_dev",
                "samples": "150"
            }
        )
        print("✅ TAB-separated format sent!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test with comma separator
    print("Testing COMMA separator...")
    try:
        send_telegram_job_summary(
            model_path_name="Qwen2.5-VL-7B",
            evaluation_results_json_file="/path/to/results.json",
            evaluation_run_logs_file="/path/to/logs.log",
            bot_token=bot_token,
            chat_id=chat_id,
            separator=",",
            include_header=True,
            send_files=False,
            message_prefix="CSV Format:",
            extra_fields={
                "accuracy": "92.5",
                "dataset": "mathvista",
                "samples": "100"
            }
        )
        print("✅ Comma-separated format sent!")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_full_pipeline_simulation(bot_token=None, chat_id=None):
    """Simulate a full evaluation pipeline notification."""
    print("\n=== Test 4: Full Pipeline Simulation ===")
    
    # Create realistic dummy files
    json_file = create_dummy_json_file()
    log_file = create_dummy_log_file()
    
    try:
        send_telegram_job_summary(
            model_path_name="/scratch_aisg/models/Qwen2.5-VL-7B-Instruct",
            evaluation_results_json_file=json_file,
            evaluation_run_logs_file=log_file,
            bot_token=bot_token,
            chat_id=chat_id,
            separator="\t",
            include_header=True,
            send_files=True,
            message_prefix="✅ Evaluation Complete",
            message_suffix="Ready for analysis",
            extra_fields={
                "reward_model": "VisualPRM-7B",
                "dataset": "mmmu_dev",
                "data_range": "0-150",
                "accuracy": "85.0%",
                "runtime": "2h 35m",
                "gpu": "A100-80GB"
            }
        )
        print("✅ Full pipeline simulation sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send full simulation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Telegram notification system")
    parser.add_argument("--bot-token", help="Telegram bot token (overrides env var)")
    parser.add_argument("--chat-id", help="Telegram chat ID (overrides env var)")
    parser.add_argument("--test", choices=["basic", "files", "spreadsheet", "full", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    # Check for credentials
    bot_token = args.bot_token
    chat_id = args.chat_id
    
    if not bot_token and not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("❌ Error: TELEGRAM_BOT_TOKEN not set!")
        print("Set it via environment variable or --bot-token argument")
        sys.exit(1)
    
    if not chat_id and not os.getenv("TELEGRAM_CHAT_ID"):
        print("❌ Error: TELEGRAM_CHAT_ID not set!")
        print("Set it via environment variable or --chat-id argument")
        sys.exit(1)
    
    print("=" * 50)
    print("TELEGRAM NOTIFICATION TEST SUITE")
    print("=" * 50)
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Bot Token: {'***' + (bot_token or os.getenv('TELEGRAM_BOT_TOKEN'))[-6:] if (bot_token or os.getenv('TELEGRAM_BOT_TOKEN')) else 'NOT SET'}")
    print(f"  Chat ID: {chat_id or os.getenv('TELEGRAM_CHAT_ID') or 'NOT SET'}")
    
    # Run tests
    results = []
    
    # if args.test in ["basic", "all"]:
    #     results.append(("Basic Message", test_basic_message(bot_token, chat_id)))
    
    # if args.test in ["files", "all"]:
    #     results.append(("Message with Files", test_message_with_files(bot_token, chat_id)))
    
    # if args.test in ["spreadsheet", "all"]:
    #     results.append(("Spreadsheet Format", test_spreadsheet_format(bot_token, chat_id)))
    
    if args.test in ["full", "all"]:
        results.append(("Full Pipeline", test_full_pipeline_simulation(bot_token, chat_id)))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The notification system is working correctly.")
        print("\nNext steps:")
        print("1. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment")
        print("2. Run your evaluation pipeline - notifications will be sent automatically")
    else:
        print("\n⚠️ Some tests failed. Please check your configuration.")
        print("\nTroubleshooting:")
        print("1. Verify your bot token is correct")
        print("2. Ensure the bot has been added to your channel/group")
        print("3. Check that the chat ID is correct (use @username for public channels)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())