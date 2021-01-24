import asyncio
import datetime
import threading

from constants import TIMEOUT
from helper import obtain_betting_data

loop = asyncio.get_event_loop()


def get():
    print('Retrieving betting data from betting SOR', datetime.datetime.now())
    loop.run_until_complete(obtain_betting_data())
    print('Retrieved betting data from betting SOR', datetime.datetime.now())
    threading.Timer(TIMEOUT, get).start()
