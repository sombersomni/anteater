import logging
import os
from datetime import datetime

def setup_logger(name, log_file="app.log"):
    """
    Sets up a logger that writes to both a file and the console
    
    Args:
        name (str): Name of the logger
        log_file (str): Name of the log file (default: "app.log")
    
    Returns:
        logging.Logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the logging level
    # Create formatters
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_dir = os.path.dirname('./logs')
    os.makedirs(os.path.dirname('./logs'), exist_ok=True)
    log_file = os.path.join(root_dir, log_file)
    # Create file handler
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(log_format)
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    # Clear any existing handlers (prevents duplicate logging)
    logger.handlers.clear()
    # Add handlers to logger
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/app_{timestamp}.log"
    
    # Set up the logger
    logger = setup_logger("MyApp", log_filename)
    
    # Test the logger
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")