import sqlite3
import json # used to load the lines from data
from datetime import datetime # used to log

timeframe = '2018-03'
# to build a big transaction to commit all rows at once instead of one at a time
sql_transaction = []

# if the database doesn't exist, sqlite3 will create the database
connection = sqlite3.connect('/Volumes/Seagate Expansion Drive/RC_2018-03.db'.format(timeframe))
c = connection.cursor()

def create_table():
    # this is the query that stores these values
    c.execute("""CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY,
    comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT,
    unix INT, score INT)""")

# makes sure table is always created
if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0 #counts number of parent-and-child pairs (comments with replies)

    with open("/Volumes/Seagate Expansion Drive/RC_2018-03".format(timeframe.split('-')[0], timeframe), buffer=1000) as f:
        # start iterating through f
        for row in f:
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            # using a helper function called format_data to clean up the data
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
