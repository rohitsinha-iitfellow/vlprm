# currently not functional because of network firewall issues

"""
Generic Telegram notification helper for evaluation runs.

Requires python-telegram-bot (v22+). Install with uv:
    uv add python-telegram-bot==22.3

Docs: https://docs.python-telegram-bot.org/en/stable/

Usage (synchronous):
    from evaluation.common.send_telegram_notifications_helper import send_telegram_job_summary

    send_telegram_job_summary(
        model_path_name="Qwen/Qwen2.5-VL-7B-Instruct",
        evaluation_results_json_file="/abs/path/to/result.json",
        evaluation_run_logs_file="/abs/path/to/run.log",
        # Optional overrides (or use env vars below):
        # bot_token="123456:ABC...",
        # chat_id="@your_channel_username" or chat_id=123456789,
        separator="\t",            # [Deprecated] no longer affects output format
        include_header=True,       # [Deprecated] no longer affects output format
        send_files=True,
        message_prefix="Eval job completed",
    )

Environment variables (used if parameters not provided):
    TELEGRAM_BOT_TOKEN   -> bot token string
    TELEGRAM_CHAT_ID     -> chat id (e.g. "@channelname" or numeric id)
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from telegram import Bot
import dotenv

__all__ = ["send_telegram_job_summary", "send_telegram_error_notification"]

dotenv.load_dotenv()


# Removed old table formatting functions - now using human-readable format


def _build_readable_summary(header_to_value: Sequence[Tuple[str, Any]]) -> str:
    """Format fields in a clean, semantically grouped format with bold titles."""
    lines: List[str] = []
    
    # Convert to dict for easier lookup
    fields = {name: value for name, value in header_to_value if value is not None and str(value).strip() != ""}
    
    # Core evaluation info
    if "evaluation_score" in fields:
        lines.append(f"**Score**: {fields['evaluation_score']}")
    
    if "evaluation_summary" in fields:
        lines.append(f"**Summary**: {fields['evaluation_summary']}")
    
    lines.append("")  # Empty line for separation
    
    # Dataset info  
    if "data" in fields:
        lines.append(f"**Dataset**: {fields['data']}")
    
    if "data_begin" in fields and "data_end" in fields:
        lines.append(f"**Data range**: {fields['data_begin']} - {fields['data_end']}")
    
    lines.append("")  # Empty line for separation
    
    # Model paths (make reward_model_path more readable)
    if "model_path_name" in fields:
        lines.append(f"**Policy Model**: {fields['model_path_name']}")
    
    if "reward_model_path" in fields:
        reward_path = str(fields["reward_model_path"])
        # Extract just the final directory name for readability
        import os
        reward_model_name = os.path.basename(reward_path)
        lines.append(f"**Reward Model**: {reward_model_name}")
        # lines.append(f"**Reward Model Path**: {reward_path}")
    
    lines.append("")  # Empty line for separation
    
    # File paths
    if "evaluation_results_json_file" in fields:
        lines.append(f"**Results JSON**:\n{fields['evaluation_results_json_file']}")
    
    if "evaluation_run_logs" in fields:
        lines.append(f"**Log File**:\n{fields['evaluation_run_logs']}")
    
    # Timestamp and other info
    if "timestamp" in fields:
        lines.append(f"**Timestamp**: {fields['timestamp']}")
    
    if "development_mode" in fields:
        lines.append(f"**Development Mode**: {fields['development_mode']}")
    
    # Any remaining fields that weren't handled above
    handled_fields = {
        "evaluation_score", "evaluation_summary", "data", "data_begin", "data_end",
        "model_path_name", "reward_model_path", "evaluation_results_json_file", 
        "evaluation_run_logs", "timestamp", "development_mode"
    }
    
    remaining_fields = [(name, value) for name, value in header_to_value 
                       if name not in handled_fields and value is not None and str(value).strip() != ""]
    
    if remaining_fields:
        lines.append("")  # Empty line for separation
        lines.append("**Other Info**:")
        for name, value in remaining_fields:
            clean_value = str(value).strip()
            lines.append(f"  {name}: {clean_value}")
    
    return "\n".join(lines)


async def _async_send(
    *,
    bot_token: str,
    chat_id: str | int,
    message_text: str,
    result_path: Optional[str],
    log_path: Optional[str],
    send_files: bool,
) -> None:
    bot = Bot(token=bot_token)
    # Send the summary text first
    await bot.send_message(chat_id=chat_id, text=message_text, disable_web_page_preview=True)

    if not send_files:
        return

    # Send result JSON if available
    if result_path and os.path.isfile(result_path):
        try:
            # Open file and send with proper filename
            with open(result_path, 'rb') as f:
                await bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=os.path.basename(result_path),
                    caption=f"📊 Evaluation Results (JSON): {result_path}",
                )
        except Exception:
            # Swallow and continue, we don't want the whole notification to fail
            pass

    # Send log file if available
    if log_path and os.path.isfile(log_path):
        try:
            # Open file and send with proper filename
            with open(log_path, 'rb') as f:
                await bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=os.path.basename(log_path),
                    caption=f"📝 Evaluation Run Logs: {log_path}",
                )
        except Exception:
            pass


def send_telegram_job_summary(
    *,
    model_path_name: str,
    evaluation_results_json_file: Optional[str] = None,
    evaluation_run_logs_file: Optional[str] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
    bot_token: Optional[str] = None,
    chat_id: Optional[str | int] = None,
    separator: Optional[str] = "\t",
    include_header: bool = True,
    send_files: bool = True,
    message_prefix: Optional[str] = None,
    message_suffix: Optional[str] = None,
) -> None:
    """
    Send a single Telegram message summarizing an evaluation run, optionally attaching files.

    Parameters
    - model_path_name: Model identifier or path
    - evaluation_results_json_file: Absolute path to JSON results (optional)
    - evaluation_run_logs_file: Absolute path to run log (optional)
    - extra_fields: Additional columns (mapping name -> value). Keys are sorted for stable order.
    - bot_token, chat_id: Override env vars TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
    - separator: [Deprecated - no longer used as output is human-readable format]
    - include_header: [Deprecated - no longer used as output is human-readable format]
    - send_files: Also upload the JSON and log files (when paths exist)
    - message_prefix/suffix: Optional free-text lines before/after the table

    Notes
    - Uses python-telegram-bot v22+ asynchronous API, executed via asyncio.run.
    - Docs: https://docs.python-telegram-bot.org/en/stable/
    """
    resolved_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not resolved_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not provided and not set in environment")

    resolved_chat_id: str | int | None = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if resolved_chat_id is None:
        raise RuntimeError("TELEGRAM_CHAT_ID not provided and not set in environment")

    # Separator and include_header no longer used - keeping parameters for backward compatibility
    _ = separator, include_header  # Suppress unused parameter warnings

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Core columns first, then any extras (sorted for determinism)
    header_to_value: List[Tuple[str, Any]] = [
        ("timestamp", timestamp),
        ("model_path_name", model_path_name),
        ("evaluation_results_json_file", evaluation_results_json_file or ""),
        ("evaluation_run_logs", evaluation_run_logs_file or ""),
    ]

    if extra_fields:
        for key in sorted(extra_fields.keys()):
            header_to_value.append((key, extra_fields[key]))

    summary_text = _build_readable_summary(header_to_value)

    lines: List[str] = []
    if message_prefix:
        lines.append(str(message_prefix).strip())
    lines.append(summary_text)
    if message_suffix:
        lines.append(str(message_suffix).strip())
    message_text = "\n".join(lines)

    try:
        asyncio.run(
            _async_send(
                bot_token=resolved_token,
                chat_id=resolved_chat_id,
                message_text=message_text,
                result_path=evaluation_results_json_file,
                log_path=evaluation_run_logs_file,
                send_files=send_files,
            )
        )
    except RuntimeError as exc:  # In case an event loop is already running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Fire-and-forget best-effort when inside an active loop
                loop.create_task(
                    _async_send(
                        bot_token=resolved_token,
                        chat_id=resolved_chat_id,
                        message_text=message_text,
                        result_path=evaluation_results_json_file,
                        log_path=evaluation_run_logs_file,
                        send_files=send_files,
                    )
                )
            else:
                loop.run_until_complete(
                    _async_send(
                        bot_token=resolved_token,
                        chat_id=resolved_chat_id,
                        message_text=message_text,
                        result_path=evaluation_results_json_file,
                        log_path=evaluation_run_logs_file,
                        send_files=send_files,
                    )
                )
        except Exception as inner_exc:  # Final fallback: don't crash the job
            # Printing rather than raising to avoid masking the run exit status
            print(f"[telegram notifier] Failed to send notification: {inner_exc or exc}")


def send_telegram_error_notification(
    *,
    model_path_name: str,
    error_message: str,
    error_traceback: Optional[str] = None,
    evaluation_run_logs_file: Optional[str] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
    bot_token: Optional[str] = None,
    chat_id: Optional[str | int] = None,
    send_files: bool = True,
) -> None:
    """
    Send a Telegram notification about evaluation run errors/failures.

    Parameters:
    - model_path_name: Model identifier or path
    - error_message: The main error message to report
    - error_traceback: Full traceback string (optional)
    - evaluation_run_logs_file: Path to log file (optional)
    - extra_fields: Additional context (mapping name -> value)
    - bot_token, chat_id: Override env vars TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
    - send_files: Also upload the log file (when path exists)
    """
    resolved_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not resolved_token:
        print(
            "[telegram notifier] TELEGRAM_BOT_TOKEN not available for error notification"
        )
        return

    resolved_chat_id: str | int | None = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if resolved_chat_id is None:
        print(
            "[telegram notifier] TELEGRAM_CHAT_ID not available for error notification"
        )
        return

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build error summary
    header_to_value: List[Tuple[str, Any]] = [
        ("timestamp", timestamp),
        ("status", "❌ FAILED"),
        ("model_path_name", model_path_name),
        ("error_message", error_message),
        ("evaluation_run_logs", evaluation_run_logs_file or ""),
    ]

    if extra_fields:
        for key in sorted(extra_fields.keys()):
            header_to_value.append((key, extra_fields[key]))

    summary_text = _build_readable_summary(header_to_value)

    # Build full message
    lines: List[str] = [
        "🚨 Evaluation Run Failed",
        "",
        summary_text,
    ]

    # Add traceback if available (but truncate if too long for Telegram)
    if error_traceback:
        lines.extend(
            [
                "",
                "📋 Full Traceback:",
                "```",
                error_traceback[:3000]
                + (
                    "..." if len(error_traceback) > 3000 else ""
                ),  # Telegram message limit
                "```",
            ]
        )

    message_text = "\n".join(lines)

    try:
        asyncio.run(
            _async_send(
                bot_token=resolved_token,
                chat_id=resolved_chat_id,
                message_text=message_text,
                result_path=None,  # No results file for errors
                log_path=evaluation_run_logs_file,
                send_files=send_files,
            )
        )
    except RuntimeError as exc:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(
                    _async_send(
                        bot_token=resolved_token,
                        chat_id=resolved_chat_id,
                        message_text=message_text,
                        result_path=None,
                        log_path=evaluation_run_logs_file,
                        send_files=send_files,
                    )
                )
            else:
                loop.run_until_complete(
                    _async_send(
                        bot_token=resolved_token,
                        chat_id=resolved_chat_id,
                        message_text=message_text,
                        result_path=None,
                        log_path=evaluation_run_logs_file,
                        send_files=send_files,
                    )
                )
        except Exception as inner_exc:
            print(
                f"[telegram notifier] Failed to send error notification: {inner_exc or exc}"
            )
