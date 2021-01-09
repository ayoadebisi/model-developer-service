import asyncio
import datetime
import threading

from constants import TIMEOUT
from helper import obtain_elo_data

loop = asyncio.get_event_loop()


def get():
    print('Retrieving elo data from processor service', datetime.datetime.now())
    loop.run_until_complete(obtain_elo_data())
    print('Retrieved elo data from processor service', datetime.datetime.now())
    threading.Timer(TIMEOUT, get).start()
