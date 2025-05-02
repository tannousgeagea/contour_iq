import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class KeepTrackOfTime:
    def __init__(self):
        self.start_time = {}
        self.end_time = {}
        self.what_is_the_time = time.time()

    def check_if_time_less_than_diff(self, start, end, diff=1):
        return (end - start) < diff
    
    def update_time(self, new):
        self.what_is_the_time = new

    def start(self, task:str):
        self.start_time[task] = time.time()
        self.end_time[task] = None
    
    def end(self, task:str):
        if not self.start_time.get(task):
            logging.warning("Start time must be initialized first !")
            return
        
        self.end_time[task] = time.time()

    def log(self, task:str, prefix='',):
        if not self.start_time.get(task):
            logging.warning("Start time must be initialized first !")
            return

        if not self.end_time.get(task):
            logging.warning("End time must be initialized first !")
            return
        
        logging.info(f"{prefix}: {round((self.end_time[task] - self.start_time[task]) * 1000)} milliseconds")
