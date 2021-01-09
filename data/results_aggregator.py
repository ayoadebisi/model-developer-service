import asyncio
import datetime
import threading

from constants import TIMEOUT
from helper import obtain_training_data

loop = asyncio.get_event_loop()


def get():
    print('Retrieving training data from processor service', datetime.datetime.now())
    loop.run_until_complete(obtain_training_data())
    print('Retrieved training data from processor service', datetime.datetime.now())
    threading.Timer(TIMEOUT, get).start()
