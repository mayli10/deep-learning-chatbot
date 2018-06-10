import sqlite3
import json # used to load the lines from data
from datetime import datetime # used to log

# to build a big transaction to commit all rows at once instead of one at a time
timeframe = '2015-05'
sql_transaction = []

# if the database doesn't exist, sqlite3 will create the database
connection = sqlite3.connect('../data/{}.db'.format(timeframe))
c = connection.cursor()

# this is the query that stores these values
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

# replace new lines so that the new line character doesn't get tokenized along with the word.
# create a fake word called newlinechar to replace all new line characters
# replace all double quotes with single quotes to not confuse our model into thinking
# there is difference between double and single quotes
def format_data(data):
    data = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return data

# We want to global the sql_transaction variable so that we can eventually clear out that variable
def transaction_bldr(sql):
    global sql_transaction
    # keep appending the sql statements to the transaction until it's a certain size
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        # for each sql statement we will try to execute it, otherwise we will just
        # accept the statement
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        # after we execute all the statements, we will just commit
        connection.commit()
        # and then empty out the transaction
        sql_transaction = []

# SQL Queries
def sql_insert_replace_comment(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        # overwrite the information with a new comment with better score
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        # inserting a new row with parent id and parent body
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        # insert information in case the comment is a parent for another comment
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def acceptable(data):
    # since we'll be using multiple models, we need to keep the data at 50 words
    # we need to make sure that the data has at least 1 word and isn't an empty comment
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    # we don't want to use data with more than 1,000 characters
    elif len(data) > 1000:
        return False
    # we don't want to use comments that are just [deleted] or [removed]
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True

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
    except Exception as e:
        #print(str(e))
        return False

def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        #print(str(e))
        return False

# makes sure table is always created and counts number of parent-and-child pairs (comments with replies)
if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0

    with open("../data/RC_{}".format(timeframe), buffering=1000) as f:
        for row in f:
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            # using a helper function called format_data to clean up the data
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            comment_id = row['name']
            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)
            # ensures that at least 2 people saw the comment (the score represents the upvote count)
            if score >= 2:
                # if a reply already exists for that comment, look at the score of the comment.
                existing_comment_score = find_existing_score(parent_id)
                # case 1: If the comment has a better score, then update the row
                if existing_comment_score:
                    if score > existing_comment_score:
                        if acceptable(body):
                            sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                # case 2: if there isn't an existing comment score but there is a parent, insert with the parent_data
                else:
                    if acceptable(body):
                        if parent_data:
                            sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                            paired_rows += 1
                        # case 3: if there is no parent, then the comment itself is also the parent. You still want to sql_insert_no_parent
                        # the data because this comment might still be someone else's parent, so you want to store its information
                        else:
                            sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)
            # test the data to see the rows
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))
