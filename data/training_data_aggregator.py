import datetime
import threading

from helper import obtain_training_data


def get():
    print('Retrieving training data from processor service', datetime.datetime.now())
    obtain_training_data()
    print('Retrieved training data from processor service', datetime.datetime.now())
    threading.Timer(604800, get).start()
