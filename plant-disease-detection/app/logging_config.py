import logging
import sys
from pythonjsonlogger import jsonlogger
import time


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            log_record["timestamp"] = time.time()
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname


def setup_logging():
    """Setup structured JSON logging"""
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Reduce logging noise from dependencies
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)

    return logging.getLogger(__name__)
