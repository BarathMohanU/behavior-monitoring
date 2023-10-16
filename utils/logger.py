import logging
import pytz
from datetime import datetime

class LoggerManager:
    class ISTFormatter(logging.Formatter):
        """
        Logging Formatter to use Indian Standard Time (IST).
        
        Attributes
        ----------
        converter : timezone
            The time zone converter, set to Asia/Kolkata to adapt logs to IST.
        """
        
        converter = pytz.timezone('Asia/Kolkata')
        
        def formatTime(self, record, datefmt=None):
            """
            Override formatTime to use IST in logs.

            Parameters
            ----------
            record : LogRecord
                The record which is being logged.
            datefmt : str, optional
                The date format to use. Defaults to None, and if None, uses
                default_time_format and default_msec_format for formatting.
            
            Returns
            -------
            s : str
                The formatted time string in IST.
            """
            dt = datetime.fromtimestamp(record.created, self.converter)
            if datefmt:
                s = dt.strftime(datefmt)
            else:
                t = dt.strftime(self.default_time_format)
                s = self.default_msec_format % (t, record.msecs)
            return s

    @staticmethod
    def get_logger(name=__name__, log_file='log_files/logs.log'):
        """
        Get a logger instance that logs in Indian Standard Time (IST).
        
        Configures and returns a logger which writes log entries to a specified file, 
        using a specified name, and formatting log timestamps to IST.

        Parameters
        ----------
        name : str, optional
            The name to give to the logger instance. Defaults to the name of the module 
            from which the method is called.
        log_file : str, optional
            The file to which the logger should write log entries. Defaults to 'bot_restart.log'.

        Returns
        -------
        logger : Logger
            The configured logger instance.
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Create file handler and set level to debug
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter with time and message format, and set the formatter for the handler
        formatter = LoggerManager.ISTFormatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(fh)
        
        return logger