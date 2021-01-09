import aiohttp
import asyncio
import json

from pandas import DataFrame

from constants import DATA_PROVIDER_URL, TRAINING_DATA_ENDPOINT, COUNTRIES, LEAGUE_MAP, RATING_TYPES, \
    DATA_PROVIDER_ENDPOINT, SEASON, FORM_DATA_ENDPOINT, LEAGUE_STANDINGS, FORM_DATA, ELO_DATA
from training.classification import train_league_classification
from training.regression import train_league_regression

loop = asyncio.get_event_loop()


async def obtain_training_data():
    league_data = []

    for country in COUNTRIES:
        # url = DATA_PROVIDER_URL + TRAINING_DATA_ENDPOINT + LEAGUE_MAP[country.lower()]
        # league_data.append(await send_request(url))
        filename = 'constants/' + country + '_training_data.json'
        with open(filename, 'r') as j:
            league_data.append(json.loads(j.read()))

    for i in range(len(COUNTRIES)):
        await train_league_classification(DataFrame(league_data[i]), COUNTRIES[i].lower())
        await train_league_regression(DataFrame(league_data[i]), COUNTRIES[i].lower())


async def obtain_elo_data():
    for country in COUNTRIES:
        ratings = {}
        for rating in RATING_TYPES:
            # url = DATA_PROVIDER_URL + DATA_PROVIDER_ENDPOINT + 'rating?country=' + country.lower() +
            # '&rating=' + rating
            # elo_rating =  await send_request(url)
            filename = 'constants/' + rating + '_elo.json'
            with open(filename, 'r') as j:
                elo_rating = json.loads(j.read())
                normalize_elo(elo_rating)
                ratings[rating] = elo_rating
        ELO_DATA[country] = ratings


async def obtain_form_data():
    for country in COUNTRIES:
        # url = DATA_PROVIDER_URL + FORM_DATA_ENDPOINT + LEAGUE_MAP[country.lower()]
        # form = await send_request(url)
        filename = 'constants/' + country + '_form.json'
        with open(filename, 'r') as j:
            FORM_DATA[country] = json.loads(j.read())


async def obtain_standings_data():
    for country in COUNTRIES:
        # url = DATA_PROVIDER_URL + DATA_PROVIDER_ENDPOINT + 'standings?league=' + LEAGUE_MAP[country.lower()] \
        #      + '&season=' + str(SEASON)
        # standings = await send_request(url)
        filename = 'constants/' + country + '_league.json'
        with open(filename, 'r') as j:
            standings = json.loads(j.read())
            LEAGUE_STANDINGS[country] = {'data': standings}


async def send_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()


def normalize_elo(elo):
    min_elo = min_value(elo)
    max_elo = max_value(elo)
    delta = max_elo - min_elo

    for key in elo:
        elo[key]['elo'] = (elo[key]['elo'] - min_elo) / delta


def min_value(elo):
    min_list = []
    for key in elo:
        min_list.append(elo[key]['elo'])
    return min(min_list)


def max_value(elo):
    max_list = []
    for key in elo:
        max_list.append(elo[key]['elo'])
    return max(max_list)
