import datetime
import json

import webhoseio

from Keywords import keywords
from webhose_configs import WEBHOSE_KEY, SCRAPE_TIME_DELTA, DATA_PATH


def query_builder(category, start_time_str):
    formatted_kwds = ['\"' + kwd + '\"' if ' ' in kwd else kwd for kwd in keywords.get(category)]
    q = ' OR '.join(['title:{0}'.format(kwd) for kwd in formatted_kwds])
    q = '(' + q + ')'
    suffix = 'language:english site_type:news (location:"united states" OR location:"united kingdom" OR location:"australia") crawled:>' + start_time_str
    q = q + suffix
    return q

def scrape(query, category, start_time_str, time_diff):
    print('Start scraping data from ' + start_time_str)

    query_params = {"q": query, "sort": "crawled"}

    news_list = []

    while True:
        output = webhoseio.query("filterWebContent", query_params)
        news_list = news_list + output['posts']
        output = webhoseio.get_next()

        if len(news_list) > output['totalResults'] or len(news_list) == 0:
            break

    filename = (
        DATA_PATH +
        'News_{0}_'.format(category) +
        str(
            datetime.datetime.utcnow() + time_diff
        ).replace(' ', '_').replace(':', '_') +
        '.json'
    )

    with open(filename, 'w') as outfile:
        json.dump(news_list, outfile)

    print('Persisted News Article at the following location: ' + filename)
    print('{0} news articles were collected.'.format(len(news_list)))

def scraper():
    current_time = datetime.datetime.utcnow()
    time_diff = datetime.timedelta(hours=-1*SCRAPE_TIME_DELTA)
    start_time = current_time + time_diff
    start_time = start_time.timestamp()
    start_time_str = str(round(start_time))
    start_time_str = start_time_str.ljust(13, '0')

    webhoseio.config(token=WEBHOSE_KEY)

    for category in keywords.keys():
        query = query_builder(category, start_time_str)
        scrape(query, category, start_time_str, time_diff)
