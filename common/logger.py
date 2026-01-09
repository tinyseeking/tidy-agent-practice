import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

try:
    # 用于打印日志时，根据不同日志级别按不同颜色显示, 若使用,需要安装：pip install colorlog
    import colorlog

    COLOR_LOG_AVAILABLE = True
except ImportError:
    COLOR_LOG_AVAILABLE = False

# 配置参数化
LOG_NAME = "app"
LOG_FILENAME = f"{LOG_NAME}.log"
LOG_DIR = "./logs"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
FILE_ENCODING = "utf-8"  # 明确设置文件编码
DEFAULT_LOG_LEVEL = logging.DEBUG  #日志打印级别


def configure_logging(
        logger_name: str,
        log_level: int = DEFAULT_LOG_LEVEL,
        log_dir: str = LOG_DIR,
        log_filename: str = LOG_FILENAME,
        max_log_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT,
        log_format: Optional[str] = None,
        file_encoding: str = FILE_ENCODING
) -> logging.Logger:
    """配置并返回一个带文件和控制台输出的日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    if not log_format:
        log_format = (
            "%(asctime)s - %(process)d - %(threadName)s - "
            "%(name)s - %(levelname)s - %(message)s"
        )

    _logger = logging.getLogger(logger_name)

    # 防止日志重复输出
    if not _logger.handlers:
        _logger.setLevel(log_level)

        # 配置文件处理器
        file_formatter = logging.Formatter(log_format)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding=file_encoding  # 设置文件编码
        )
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)

        # 配置控制台处理器
        console_handler = logging.StreamHandler()

        if COLOR_LOG_AVAILABLE:
            # 使用colorlog为不同级别设置颜色
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s" + log_format,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            # 回退到普通格式
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)

        _logger.addHandler(console_handler)

    return _logger


# 使用示例
if __name__ == "__main__":
    # 配置并获取日志记录器
    _logger = configure_logging(__name__)

    _logger.debug("这是一条调试信息")
    _logger.info("这是一条普通信息")
    _logger.warning("这是一条警告信息")
    _logger.error("这是一条错误信息")
    _logger.critical("这是一条严重错误信息")
