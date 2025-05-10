# agents/tools/datetime_tool.py
from .base_tool import Tool, ToolException
from datetime import datetime
import pytz 
import os 

class DateTimeTool(Tool):
    name = "get_current_datetime"
    description = ("Returns the current date and time, optionally in a specified timezone and format. "
                   "Useful for logging, timestamping, or providing time-sensitive information.")
    argument_schema = {
        "timezone": "str (optional): A standard IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'UTC'). Defaults to UTC if not specified.",
        "format": "str (optional): A Python strftime format string (e.g., '%Y-%m-%d %H:%M:%S %Z%z'). If not provided, a default ISO-like format is used."
    }

    def execute(self, **kwargs) -> str:
        tz_name = kwargs.get("timezone", "UTC") 
        time_format = kwargs.get("format")

        try:
            target_tz = pytz.timezone(tz_name)
        except pytz.exceptions.UnknownTimeZoneError:
            print(f"[DateTimeTool] Unknown timezone '{tz_name}', defaulting to UTC.")
            target_tz = pytz.utc
            tz_name = "UTC (fallback due to invalid input)"

        now_utc = datetime.now(pytz.utc) 
        now_target_tz = now_utc.astimezone(target_tz)

        if time_format:
            try:
                return f"Current date and time ({tz_name}): {now_target_tz.strftime(time_format)}"
            except Exception as e:
                print(f"[DateTimeTool] Error formatting date with format '{time_format}' for timezone '{tz_name}': {e}")
                return (f"Error with format '{time_format}'. Current date and time ({tz_name}): "
                        f"{now_target_tz.isoformat()}")
        else:
            return f"Current date and time ({tz_name}): {now_target_tz.isoformat()}"