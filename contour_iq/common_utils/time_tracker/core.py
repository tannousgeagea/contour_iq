import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class KeepTrackOfTime:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.what_is_the_time = time.time()

    def check_if_time_less_than_diff(self, start, end, diff=1):
        return (end - start) < diff
    
    def update_time(self, new):
        self.what_is_the_time = new

    def start(self):
        self.start_time = time.time()
        self.end_time = None
    
    def end(self):
        if not self.start_time:
            print(self.start_time)
            logging.warning("Start time must be initialized first !")
            return
        
        self.end_time = time.time()

    def log(self, prefix=''):
        if not self.start_time:
            logging.warning("Start time must be initialized first !")
            return

        if not self.end_time:
            logging.warning("End time must be initialized first !")
            return
        
        logging.info(f"{prefix}: {round((self.end_time - self.start_time) * 1000)} milliseconds")
