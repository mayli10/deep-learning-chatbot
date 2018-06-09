import sqlite3
import json # used to load the lines from data
from datetime import datetime # used to log

timeframe = '2018-03'
# to build a big transaction to commit all rows at once instead of one at a time
sql_transaction = []

# if the database doesn't exist, sqlite3 will create the database
connection = sqlite3.connect('/Volumes/Seagate Expansion Drive/RC_{}.db'.format(timeframe))
c = connection.cursor()

def create_table():
    # this is the query that stores these values
    c.execute("""CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY,
    comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT,
    unix INT, score INT)""")


def format_data(data):
    # replace new lines so that the new line character doesn't get tokenized along with the word.
    # create a fake word called redditnewlinechar to replace all new line characters
    # replace all double quotes with single quotes to not confuse our model into thinking
    # there is difference between double and single quotes
    data = data.replace("\n", " redditnewlinechar ").replace('"', "'")

# NOTEe TO SELF: COMBINE THESE INTO ONE FUNCTION!

def find_existing_score(pid):
    try:
        # looks for anywhere where the comment_id is the parent
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        # execute and return results
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    # catches any exceptions
    except Exception as e:
        print("find_existing_score", e)
        return False;

def find_parent(pid):
    try:
        # looks for anywhere where the comment_id is the parent
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        # execute and return results
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    # catches any exceptions
    except Exception as e:
        print("find_parent", e)
        return False;

def acceptable(data):
    # since we'll be using multiple models, we need to keep the data at 50 words
    # we need to make sure that the data has at least 1 word and isn't an empty comment
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    # we don't want to use data with more than 1,000 characters
    elif len(data) > 1000:
        return False
    # we don't want to use comments that are just [deleted] or [removed]
    elif data = '[deleted]' or data = '[removed]':
        return False
    else:
        return True

# makes sure table is always created
if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0 #counts number of parent-and-child pairs (comments with replies)

    with open("/Volumes/Seagate Expansion Drive/RC_{}".format(timeframe), buffering=1000) as f:
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
            parent_data = find_parent(parent_id)

            # ensures that at least 2 people saw the comment (the score represents the upvote count)
            if score >= 2:
                # if a reply already exists for that comment, look at the score of the comment.
                # If the comment has a better score, then update the row
                existing_comment_score = find_existing_score(parent_id)
                if existing_comment_score:
                    if score > existing_comment_score:
