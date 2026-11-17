from ..global_import import *
RESET = "\033[0m"# Reset Color
COLOR_CODES = {
    'DEBUG': "\033[36m",  # Cyan
    'INFO': "\033[32m",   # Green
    'WARNING': "\033[33m",  # Yellow
    'ERROR': "\033[31m",  # Red
    'CRITICAL': "\033[1;31m",  # Bold Red
}
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLOR_CODES.get(record.levelname, RESET)
        log_message = super().format(record)
        return f"{log_color}{log_message}{RESET}"
    
class HTMLFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'orange',
        'ERROR': 'red',
        'CRITICAL': 'darkred',
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, 'black')
        message = super().format(record)
        return f'<p style="color:{color}">{message}</p>'
class TXTFormatter(logging.Formatter):
    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        levelname = f"{record.levelname:<8}"
        message = super().format(record)        
        formatted_message = f"{asctime} | {levelname} | {message}"
        return formatted_message
