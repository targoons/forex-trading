#from pytrends.request import TrendReq
import pandas as pd
import Keywords
import getopt
import sys


opts, args = getopt.getopt(sys.argv[1:], 's:e:g:k')

DATA_ROOT = "/project/msca/projects/ForexPrediction/data/google_trends/"

print('starting')

for opt, arg in opts:
    print(opt)
    print(arg)
    if opt == '-s':
        start = str(arg)
        start_year = int(start.split("-")[0])
        start_month = int(start.split("-")[1])
        start_day = int(start.split("-")[2])
        start_time = int(start.split("-")[3])
    if opt == '-e':
        end = str(arg)
        end_year = int(end.split("-")[0])
        end_month = int(end.split("-")[1])
        end_day = int(end.split("-")[2])
        end_time = int(end.split("-")[3])
    if opt == '-g':
        geo = str(arg)
    if opt == '-k':
        category = str(arg)
    #if opt == '-t':
     #   sleep_time  = int(float(arg))
#category = "Natural_Disaster"
sleep_time = 60

print("Scraping Google Trends with following arguments \n {0} {1} {2} {3} {4} {5} {6}".format(start, end, start_time, end_time, geo, sleep_time, category))
print("foo")

def query_arg(keyword_tuple):
    temp_list = []
    for keyword in keyword_tuple:
        temp_list.append(keyword)
        if len(temp_list) == 1:
            yield temp_list
            temp_list = []
    yield temp_list

kw = Keywords.keywords

pytrends = TrendReq(hl='en-US', tz=0)

df_list = []

for keyword in kw.get(category):
    print("Scraping Google Trends for {0}".format(keyword))
    if keyword:
        result = pytrends.get_historical_interest(
                [keyword], 
                year_start=start_year, 
                month_start=start_month, 
                day_start=start_day, 
                hour_start=start_time, 
                year_end=end_year, 
                month_end=end_month, 
                day_end=end_day, 
                hour_end=end_time, 
                cat=0, 
                geo=geo, 
                gprop='news', 
                sleep=sleep_time
        )
        df_list.append(result)
        print("Finished scraping Google Trends for {0}".format(keyword))
        result.to_csv('{0}{1}.csv'.format(DATA_ROOT, keyword))

results = pd.concat(df_list, axis=1)
results.drop(list(filter(lambda x: "isPartial" in x, list(df.columns))), axis=1, inplace=True)
file_name = DATA_ROOT + 'google_trends_{0}_{1}-{2}-{3}-{4}_{5}.csv'.format(category, start, start_time, end, end_time, geo)
results.to_csv(file_name)
print("Finished scraping and persisted file:{0}".format(file_name))
