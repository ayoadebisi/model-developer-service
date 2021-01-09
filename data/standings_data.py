import asyncio
import datetime
import threading

from constants import TIMEOUT
from helper import obtain_standings_data

loop = asyncio.get_event_loop()


def get():
    print('Retrieving standings data from processor service', datetime.datetime.now())
    loop.run_until_complete(obtain_standings_data())
    print('Retrieved standings data from processor service', datetime.datetime.now())
    threading.Timer(TIMEOUT, get).start()
