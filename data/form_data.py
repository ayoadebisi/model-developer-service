import asyncio
import datetime
import threading

from constants import TIMEOUT
from helper import obtain_form_data

loop = asyncio.get_event_loop()


def get():
    print('Retrieving form data from processor service', datetime.datetime.now())
    loop.run_until_complete(obtain_form_data())
    print('Retrieved form data from processor service', datetime.datetime.now())
    threading.Timer(TIMEOUT, get).start()
