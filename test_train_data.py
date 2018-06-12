import sqlite3
import pandas as pd
from datetime import datetime

timeframes = ['2015-05']

# build the connection and then read sql from pandas
for timeframe in timeframes:
    connection = sqlite3.connect('../data/{}.db'.format(timeframe))
    c = connection.cursor()
    limit = 5000 # how much we will pull at each time to show in panda dataframe
    last_unix = 0  # will help us buffer through the database
    cur_length = limit
    counter = 0
    test_done = False
    # keep making pulls until reach the limit, then we will put the data into the dataframe
    while cur_length == limit:
        # df = dataframe, * = all
        print('Before, Time = {}'.format(str(datetime.now())))
        df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format
        (last_unix, limit), connection)
        print('After, Time = {}'.format(str(datetime.now())))
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            # testing data
            # "from" data is the comment and "to" data is the reply
            with open("../data/test.from", 'a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open("../data/test.to", 'a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(content+'\n')

            test_done = True
        else:
            # training data
            # "from" data is the comment and "to" data is the reply
            with open("../data/train.from", 'a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open("../data/train.to", 'a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(content+'\n')
        counter +=1
        if counter % 20 == 0:
            # every 20 * limit (20*5000=100,000) rows we will see the information printed above
            print(counter*limit, 'rows completed so far')
