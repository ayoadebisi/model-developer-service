from constants import ELO_DATA, LEAGUE_STANDINGS, FORM_DATA


def get_elo_feature(league_info, elo_option):
    home_elo = find_elo_rating(league_info['country'], league_info['home_team'], elo_option)
    away_elo = find_elo_rating(league_info['country'], league_info['away_team'], elo_option)
    return home_elo - away_elo


def get_standings_feature(league_info, league_option):
    home_pos = find_standings_data(league_info['country'], league_info['home_team'], league_option)
    away_pos = find_standings_data(league_info['country'], league_info['away_team'], league_option)
    return home_pos - away_pos


def get_form_feature(league_info, form_option):
    home_form = find_form_data(league_info['country'], league_info['home_team'], form_option)
    away_form = find_form_data(league_info['country'], league_info['away_team'], form_option)
    return home_form - away_form


def find_elo_rating(country, team, elo):
    for key in ELO_DATA[country][elo]:
        if ELO_DATA[country][elo][key]['team'].lower() == team.lower():
            return ELO_DATA[country][elo][key]['elo']


def find_standings_data(country, team, metric):
    for key in LEAGUE_STANDINGS[country]['data']:
        if LEAGUE_STANDINGS[country]['data'][key]['team'].lower() == team.lower():
            return LEAGUE_STANDINGS[country]['data'][key][metric]


def find_form_data(country, team, streak):
    for key in FORM_DATA[country]:
        if FORM_DATA[country][key]['team'].lower() == team.lower():
            return FORM_DATA[country][key][streak]
