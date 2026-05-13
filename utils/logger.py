"""
logger.py
---------
Structured logging setup for the man-down detection pipeline.

Features
--------
- Colour-coded console output (INFO=green, WARNING=yellow, ERROR=red, DEBUG=grey)
- Rotating file handler — keeps last N log files, avoids filling Jetson eMMC
- JSON-structured log option for ingestion into a SIEM/monitoring system
  (useful for oil & gas HSE audit requirements)
- Per-module logger factory (get_logger) so each module has its own name
- Alert-specific logger that writes to a separate alerts.log file —
  makes incident investigation trivial

Usage
-----
    from utils.logger import get_logger, setup_logging

    setup_logging(log_dir="logs", level="INFO", enable_json=False)

    logger = get_logger(__name__)
    logger.info("Worker started")
    logger.warning("Track %d: falling suspected", track_id)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ── ANSI colour codes (console only) ──────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BRIGHT_RED = "\033[91m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _GREEN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BRIGHT_RED,
}


# ── Formatters ────────────────────────────────────────────────────────

class ColourConsoleFormatter(logging.Formatter):
    """
    Human-readable coloured console formatter.
    Format: HH:MM:SS  LEVEL  module_name  message
    """
    FMT = "{time}  {level:<8}  {name:<22}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        time_str = time.strftime("%H:%M:%S", time.localtime(record.created))
        level_str = colour + _BOLD + record.levelname + _RESET
        name_str  = _GREY + record.name + _RESET
        msg_str   = colour + record.getMessage() + _RESET

        if record.exc_info:
            msg_str += "\n" + self.formatException(record.exc_info)

        return self.FMT.format(
            time=time_str,
            level=level_str,
            name=name_str,
            msg=msg_str,
        )


class PlainFileFormatter(logging.Formatter):
    """Plain text formatter for rotating log files."""
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for SIEM / monitoring ingestion.
    Each log line is one valid JSON object.

    Fields: timestamp, level, logger, message, [exc_info]
    Alert events additionally include: track_id, alert_state, score
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Extra fields injected by alert logger
        for field in ("track_id", "alert_state", "score", "camera_id"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Setup ─────────────────────────────────────────────────────────────

_logging_configured = False


def setup_logging(
    log_dir:     str  = "logs",
    level:       str  = "INFO",
    enable_json: bool = False,
    max_bytes:   int  = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,                   # keep 5 rotated files
) -> None:
    """
    Configure the root logger. Call once at application startup (in main.py).

    Parameters
    ----------
    log_dir : str
        Directory for log files. Created if it does not exist.
    level : str
        Logging level: DEBUG | INFO | WARNING | ERROR
    enable_json : bool
        If True, write a parallel JSON log file (alerts.json).
    max_bytes : int
        Max size of each rotating log file in bytes.
    backup_count : int
        Number of old log files to retain before deletion.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # ── Console handler (colour) ───────────────────────────────────
    if sys.stdout.isatty() or os.environ.get("FORCE_COLOR"):
        console_fmt = ColourConsoleFormatter()
    else:
        console_fmt = PlainFileFormatter()   # no colour in redirected stdout

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    # ── Rotating plain-text log file ───────────────────────────────
    log_path = Path(log_dir) / "mandown.log"
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(PlainFileFormatter())
    file_handler.setLevel(numeric_level)
    root.addHandler(file_handler)

    # ── Separate alerts log (WARNING and above) ────────────────────
    alert_log_path = Path(log_dir) / "alerts.log"
    alert_handler  = logging.handlers.RotatingFileHandler(
        str(alert_log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    alert_handler.setFormatter(PlainFileFormatter())
    alert_handler.setLevel(logging.WARNING)
    root.addHandler(alert_handler)

    # ── Optional JSON log (for SIEM / oil & gas audit) ─────────────
    if enable_json:
        json_path = Path(log_dir) / "alerts.json"
        json_handler = logging.handlers.RotatingFileHandler(
            str(json_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.WARNING)   # only alerts in JSON
        root.addHandler(json_handler)

    # Suppress noisy third-party loggers
    for noisy in ("ultralytics", "urllib3", "PIL", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(
        "Logging initialised — level=%s, log_dir=%s, json=%s",
        level, log_dir, enable_json,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Use __name__ in each module:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


# ── Alert event logger helper ─────────────────────────────────────────

class AlertLogger:
    """
    Convenience wrapper that emits structured WARNING log entries
    for alert events, including track metadata as extra fields.

    These entries are captured by alerts.log and alerts.json.

    Usage
    -----
        alert_log = AlertLogger(camera_id="CAM_01")
        alert_log.fire(event)
        alert_log.resolve(track_id=5, duration_s=12.4)
    """

    def __init__(self, camera_id: str = "unknown") -> None:
        self._log = get_logger("alert")
        self._camera_id = camera_id

    def fire(self, event) -> None:
        """Log a new alert event."""
        self._log.warning(
            "MAN DOWN ALERT — camera=%s track=%d score=%.2f fallen_frames=%d",
            self._camera_id,
            event.track_id,
            event.smoothed_score,
            event.fallen_counter,
            extra={
                "track_id":    event.track_id,
                "alert_state": event.alert_state.name,
                "score":       event.smoothed_score,
                "camera_id":   self._camera_id,
            },
        )

    def resolve(self, track_id: int, duration_s: float) -> None:
        """Log when an alert is resolved (person recovered)."""
        self._log.warning(
            "ALERT RESOLVED — camera=%s track=%d duration=%.1fs",
            self._camera_id, track_id, duration_s,
            extra={
                "track_id":    track_id,
                "alert_state": "RESOLVED",
                "camera_id":   self._camera_id,
            },
        )