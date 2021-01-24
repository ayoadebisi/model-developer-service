from data import results_aggregator, elo_data, form_data, standings_data, beting_data


def start_jobs():
    results_aggregator.get()
    elo_data.get()
    form_data.get()
    standings_data.get()
    beting_data.get()
