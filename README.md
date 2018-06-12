A Beginner's Guide to Building a Deep Learning Chatbot
===================

Table of Contents
-------------
1. [Foreword](#foreword)
2. [Introduction](#introduction)
3. [Prerequisites: Methods and Tools](#prerequisites-methods-and-tools)
4. [Setup](#setup)
5. [Datasets](#datasets)
6. [Deep Learning vs. Machine Learning](#deep-learning-vs-machine-learning)
7. [Chatbot With Neural Machine Translation (NMT)](#chatbot-with-neural-machine-translation-nmt)
8. [Step 1: Structure and Clean the Data](#step-1-structure-and-clean-the-data)
9. [Step 2: Write SQL Insertions](#step-2-write-sql-insertions)
10. [Step 3: Build Paired Rows](#step-3-build-paired-rows)
11. [Step 4: Partition the Data](#step-4-partition-the-data)
12. [Step 5: Train with nmt-chatbot](#step-5-train-with-nmt-chatbot)
13. [Step 6: Interact and Test](#step-6-interact-and-test)
14. [Results and Reflection](#results-and-reflection)
15. [Acknowledgements](#acknowledgements)

Foreword
---------
*This is a very beginner-oriented tutorial with a deep-dive into every basic detail.* I will be assuming you have no background in machine learning whatsoever, so I will be leaving out the advanced alternatives from my tutorial.
For more advanced options and a less rigorous tutorial such as building the chatbot with the entire Reddit dataset of comments, visit sentdex's [video](https://www.youtube.com/watch?v=dvOnYLDg8_Y&t=140s) or [text](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/) tutorials.

Introduction
-------------
When I hear the buzzwords *neural network* or *deep learning*, my first thought is *intimidated*. 
Even with a background in Computer Science and Math, self-teaching machine learning is challenging. The modern world of artificial intelligence is exhilarating and rapidly-advancing, but the barrier to entry for learning how to build your own machine learning models is still dizzyingly high. 

I began my deep learning journey with a grand idea - I wanted to build a chatbot with functions that I hoped could improve mental healthcare. Because my current college, Vassar College, doesn't offer any machine learning courses, I found my way into a Deep Learning independent study with Professor Josh deLeeuw and began to self-teach with Dr. Andrew Ng's [deeplearning.ai](https://deeplearning.ai) online class. 
But, after continuously finding myself lost in the dense mathematical jargon and beginner-unfriendly tutorials, I realized that I needed to find an alternative. Thus, I stumbled upon sentdex's [tutorials](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/), and found the extensive explanations to be a wonderful relief. 

However, I realized that there is still a signficant learning curve involved for those, like me, who have limited experience with machine learning or Python. While the tutorials are clear to understand, there are multiple bugs, software incompatibilities, and hidden or unexpected technical difficulties that arose when I completed this tutorial. There were many challenges that were near-impossible to solve without consulting external sources of knowledge or extensive research, and many hidden prerequisites that almost forced me to quit my journey through the tutorial as many other people have done. 

My primary goal in building this chatbot is to first understand the foundations for building a deep learning chatbot, and then curating my chatbot to address a specific need in the mental health care industry. My secondary goal is to provide the essentials tips and bug fixes that have not been properly documented in the original tutorial and that I have learned through my own experience. I realized that without this supplemental information, I would not have been able to complete the tutorial by my own. 

**Thus, I decided to document my experience and create this deep-dive beginner-oriented tutorial which will help ease the bugs that arise. All material has been learned and adapted from [sentdex's tutorials](https://www.youtube.com/watch?v=dvOnYLDg8_Y&t=140s). This tutorial serves as both an everything-you-need-to-know walk-through for those who are just beginning to build deep learning models as well as a documentation of my own journey of building this bot!**

Prerequisites: Methods and Tools
---------------------------------
It's essential that you have these prerequisites to even be able to proceed with this tutorial. Otherwise, you can always find a solution, but this tutorial will mostly like not have the answers to those issues.
You must have:

1. **Latest versions of Python3.6+ (the programming language we are using) and Pip3 (package management system for Python3) installed** 
2. **At least 50 GB of unused storage.** The dataset we'll be using is 33.5 GB, but you'll need even more (~8 GB) later on. Free up storage with your pc's Storage Management System or use an external hard drive. 
3. **A text editor such as Atom or Sublime.** Any text editor will work, but it is recommended that you have access to a debugger.
4. **Enough time.** There are 3 timesucks to this project: Step 4 (~6 hours), Step 5 (~2.5 hours), and Step 6 (~40 hours). *Note: I used an external hard drive, so the speed and time it has taken me to run my code is likely to be slower than average. However, every computer is different based on numerous factors such as memory and even internet speed, so always plan to budget more time than expected.* If you want to cut down on time, prepare to trade with money. As I will mention in Step 4, 5, and 6, there are alternatives to explore.

Setup
------
**NOTE: Because my model is not done training, do not execute these steps (yet) since it will not work (yet!!).**
If you want to check out the chatbot that I have built, follow these steps. Note: to run this, you must still have all the prerequisites mentioned above! Otherwise, continue with the tutorial to build your own! 

Open up your terminal, and type: 
1. ```$ git clone --recursive https://github.com/mayli10/deep-learning-chatbot```
2. ```$ cd deep-learning-chatbot``` 
3. ```$ python3 hello_chatbot.py``` This will show questions/comments and its corresponding replies.

If you would like to talk to the chatbot live, then navigate out of the deep-learning-chatbot folder, and clone sentdex's helper utilities repository in a new folder.

4. ```$ git clone --branch v0.1 --recursive https://github.com/daniel-kukiela/nmt-chatbot.git```
5. We will now use the inference utility. In your terminal, type: ```$ python3 inference.py < hello_chatbot.py```
6. If you are having trouble, use sentdex's demo [here](https://github.com/daniel-kukiela/nmt-chatbot/blob/master/README.md#demo-chatbot)

Datasets
--------- 
If you are new to machine learning, a good tip to remember is that the most important and difficult aspect of machine learning is finding enough of the correct training data to train the model on. Training the model could be expensive and time-consuming, and we also need to find the specific type of data to train with. Some good dataset sources for future projects can be found at [r/datasets](https://www.reddit.com/r/datasets/), [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), or [Kaggle](https://www.kaggle.com/). The larger the dataset, the more information the model will have to learn from, and (usually) the better your model will have learned. But, since we are constrained by the memory of our computers or the monetary cost of external storage, let’s build our chatbot with the minimal amount of data needed to train a decent model. 

Even this amount of data is not tiny. We will be using all the Reddit comments from May 2015 
(labeled RC_2015-05.bz2), representing just 54 million rows of a dataset containing 1.7 billion Reddit comments. This month of data will be more than enough to train a decent model, but if you want to take it one step further beyond the basics, read more at sentdex’s tutorial here. Download the May 2015 data [here](http://files.pushshift.io/reddit/comments/), and if you want to view the full dataset, you can find it [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/?st=j9udbxta&sh=69e4fee7). 

If you are unfamiliar with Reddit, the comments are structured in a non-linear tree structure. A Redditor makes a post, and other Redditors comment on the post. There can be:
1.	Comment with no replies
2.	Comment with a reply
3.	Comment with a reply that has a reply

Because we just need a comment (input) and reply (output) pair, we will be addressing how to filter out the data so that we pick comment-reply pairs. Furthermore, if there are multiple replies to the comment, we will pick the top-voted reply. We will address this issue at Step 5.

**How to Download:**
Make sure that you have at least 50 GB of free space on your computer. Use software such as [App Cleaner](https://freemacsoft.net/appcleaner/), [CleanMyMac](https://cleanmymac.com/), or the Storage Management Tool on your computer to do this, but if you still don’t have enough, a good way to address this issue would be to buy an external hard drive. The one I am using is [Seagate](https://www.amazon.com/Seagate-Portable-External-Photography-STDR2000100/dp/B00FRHTSK4), and it contains 1 TB of space for a relatively cheap price of $53. I began with using software to make space for the data, but after multiple efforts and many hours of whittling down my Applications folder, it made sense to just use an external hard drive. It will definitely be slower to use the hard drive, but if it’s the last option for you, then it’s still a viable option.

If you’re using an external storage drive, plug in your drive and make sure to download your file directly into the drive.
If neither of these options work, another option is to use [Amazon Web Services (AWS)](https://aws.amazon.com/) or [Paperspace](https://www.paperspace.com/&R=IJHK5NF). Skip down to step 5 to learn more about Paperspace if you choose this option.

Deep Learning vs. Machine Learning
--------------
Deep learning is a type of machine learning that uses feature learning to continuously and automatically analyze data to detect features or classify data. Essentially, deep learning uses a larger amount of layers of algorithms in models such as a Recurrent Neural Network or Deep Neural Network to take machine learning a step further. 

While machine learning learns using algorithms and makes informed decisions, deep learning is a type of machine learning that furthermore uses these algorithms with more networks of layers to make intelligent inferred decisions. While with machine learning, the programmer needs to provide the features that the model needs for classification, deep learning automatically discovers these features itself. Although deep learning generally needs much more data to train than machine learning, the results are often much more advanced than that of machine learning.

Chatbot with Neural Machine Translation (NMT)
--------------
*Note: the following section provides somewhat dense technical information to assist in your understanding towards the model that is used for the chatbot. However, if this is too difficult to follow, come back to this section later when you are about to train and use your model with [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot)*

Building a chatbot with deep learning is an exciting approach that is radically different than building a chatbot with machine learning. **We want to build a chatbot that can make its own inferences and detect features to use that we don’t explicitly define for them.** With a machine learning chatbot, we would give the bot a set of intents, which are the intentions of the user’s utterance to the bot, and entities, such as the descriptors the user utters. For example, a user could say to the bot, “Tell me your name,” and the engineer would have specified that “tell” is an intent and “name” is an entity.

However, in deep learning, the process is much different. We will not be specifying features to the bot, but will instead expect the bot to detect these features itself and respond appropriately. Particularly, we will be using **Neural Machine Translation (NMT),** which is a vast artificial neural network that uses deep learning and feature learning to model full sentences with machine translation. This approach specializes in producing continuous sequences of words better than the traditional approach of using a recurrent neural network (RNN) because it mimics how humans translate sentences. Essentially, when we translate, we read through the entire sentence, understand what the sentence means, and then we translate the sentence. NMT performs this process with an encoder-decoder architecture; the encoder transforms the sentence into a vector of numbers that represent the meaning of the sentence and then the decoder takes this meaning vector and constructs a translated sentence. Thus, at its core, an NMT model is a deep multi-layer with 2 Bi-Directional Recurrent Neural Networks: the encoder BRNN and the decoder BRNN. Here is an example from sentdex’s tutorial that shows this architecture:

![alt text](/encdec.jpg?raw=true "Encoder-Decoder Architecture from the tensorflow/nmt GitHub")

This **sequence-to-sequence** model (colloquially referred to in the ML community as seq2seq) is often used for machine translation, text summarization, and speech recognition, and TensorFlow provides a tutorial on building your own NMT model [here](https://github.com/tensorflow/nmt). As a beginner, I found that this tutorial was a little too dense to understand, so I recommend using sentdex’s NMT [model](https://github.com/daniel-kukiela/nmt-chatbot) built specifically for this tutorial that includes additional utilities along with a pre-trained NMT model. (Aside: I intend to build my own seq2seq model after further self-learning, as my current attempts at building this model have been insufficient for use in building this deep learning chatbot).

It is essential that we use Bi-Directional Recurrent Neural Networks because with organic human language, there is value in understanding the context of the words or sentences in relation to other words and sentences. Because both the past, present, and future data in the sentence is important to remember and know to understand the sentence as a whole, we need a neural network that has an input sequence that can go both ways (forward and reverse) to understand a sentence. What differentiates the BRNN from a simple RNN is this ability, which is due to a hidden layer between the input and output layers of the network that can pass data both forwards and backwards, essentially giving the network the ability to understand previous, present, and incoming input as a cohesive whole.

Step 1: Structure and Clean the Data
--------------------------------------
Starting at these steps, please view and follow along with my chatbot_database.py file (included below). I provide commentary (indicated by the #) to almost every block of code to explain what is happening at each line. **Note: If you're also following along in the video and text tutorials, sentdex talks about buffering through the data if you're working with multiple months of data. This is an advanced option that I will not be explaining in detail because we will only be working with 1 month, but we will still write the code that sets up the data buffering.**

Let's first store the data into an SQLite database, so we will need to import SQLite3 so we can insert the data into the database with SQLite queries.

We will also need to import JSON to load the lines of data and to import datetime to log and keep track of how long it takes to process the data. **Seeing the date of time of when each set of data finished processing is extremely helpful for determining how long it will take to finish!** 

```
import sqlite3
import json # used to load the lines from data
from datetime import datetime # used to log
```

We will then create some variables, and also structure the code so that we are able to
create one SQL interaction that executes all the code at once instead of one at a time.

```
# to build a big transaction to commit all rows at once instead of one at a time
timeframe = '2015-05'
sql_transaction = []
```

Now comes making the connection to the data. **This is a part that caused me a bit of trouble and was not made clear on the tutorial.**
It is important to note that {} in Python acts as a placeholder for another value that comes d
directly afterwards, often as a parameter inside .format(). You must include /{}.db after
including the path name of your file. To see your path name, you can often just open a terminal and drag and drop your file into the terminal to see the path.

Essentially, this is how you write your connection script with PATH_NAME_OF_DATA replaced with the path name of your data:
```connection = sqlite3.connect('PATH_NAME_OF_DATA/{}.db'.format(timeframe))```

```
# if the database doesn't exist, sqlite3 will create the database
connection = sqlite3.connect('/Volumes/Seagate Expansion Drive/{}.db'.format(timeframe))
c = connection.cursor()
```

Now that you have your data, let’s look at one row of JSON data:
```{"author":"Arve","link_id":"t3_5yba3","score":0,"body":"Can we please deprecate the word \"Ajax\" now? \r\n\r\n(But yeah, this _is_ much nicer)","score_hidden":false,"author_flair_text":null,"gilded":0,"subreddit":"reddit.com","edited":false,"author_flair_css_class":null,"retrieved_on":1427426409,"name":"t1_c0299ap","created_utc":"1192450643","parent_id":"t1_c02999p","controversiality":0,"ups":0,"distinguished":null,"id":"c0299ap","subreddit_id":"t5_6","downs":0,"archived":true}```

The most important fields that we will factor in are parent_id, comment_id, body, name, and score. Let's store all the values into a table, but let's focus on those aforementioned fields when we write our functions to further clean our data. 

```
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")
```
Here, you want to replace new lines so that the new line character doesn't get tokenized along with the word. To do this, we will create a fake word called 'newlinechar' to replace all new line characters. This is the same with quotes, so replace all double quotes with single quotes so to not confuse our model into thinking there is difference between double and single quotes.

```
def format_data(data):
    data = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return data
```

Next, we want to make the sql_transaction variable a global variable so that we can eventually clear out that variable after we execute all the statements and commit them. 

```
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
```

Now, let's make sure that our data is acceptable to use. If the data is an empty comment, removed or deleted (Reddit displays
removed or deleted comments with brackets), or too long of a comment, then we don't want to use that data.

```
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
```
Next, let’s talk about the paired comment-replies in more detail. 
Because we need an input and an output, we need to pick comments that have at least 1 reply as the input, and the most upvoted reply (or only reply) for the output. 

Since we will insert every comment into the database chronologically, every comment will initially be considered a parent. We will write functions to differentiate the replies and organize the rows into comment-reply paired rows. Then, if we find a reply to a parent that has a higher-voted score than the previous reply, we will replace that original reply with the new and better reply.

The find_parent function will take in a parent_id (named in the parameter field as 'pid') and find the parents, which are found when the comment_id also the parent_id. We want to find the parents to create the parent-reply paired rows, as this will serve as our input (parent) and our output that the chatbot will infer its reply from (reply).

```
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
```
Let's also write a function that will find the existing score of the comment using the parent_id. This will help us select the best reply to pair with the parent in the next section.
```
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
```

Step 2: Write SQL Insertions
----------------------------------
Here, we will define SQLite insertions that will essentially add or change information in the database we are building. 
We will define a function called sql_insert_replace_comment that will take in the main fields of a comment, and replace the comment if the comment has a better score than the previous comment.
**NOTE: If your compiler has a difficult time recognizing the ? as characters that will be replaced in a similar way that {} works, then use {} instead. 

```
def sql_insert_replace_comment(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        # overwrite the information with a new comment with better score
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))
```
Next, we will write an insertion query that inserts a new row with the parent_id and parent body if the comment has a parent. This will provide the pair that we will need to train the chatbot.

```
def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        # inserting a new row with parent id and parent body
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))
```
Finally, we will write an insertion query that inserts information that will be used in the case that the comment is no parent. We want to insert this information anyways in case the comment is a parent for another comment.

```
def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        # insert information in case the comment is a parent for another comment
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))
```

Step 3: Build Paired Rows
--------------------------

Now, we will sort out our paired rows using the insertion queries and data-cleaning functions we wrote above. To begin, we will start with a check that makes sure a table is always created regardless of whether or not there is data (but there should be data!). We will also create the variables that count the row we are currently at and the number of paired rows, which are parent-and-child pairs (comments with replies).
```
if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
```
Using our data, let's create our data table by including our features. Assign names for them, such as ```parent_id = 
row['parent_id']``` Notice that each time we finish a row, we will increment the row counter.We are also using the format_data function we created in Step 1.
Don't forget that you need to include your file path name again when you are using the open() function as you will be accessing you data files. Follow the format mentioned in Step 1, but this time, you will not be including '.db'.

**NOTE: While you should not have this issue if you are using the correct dataset (May 2015), many other people had the issue of not being able to find the 'name' field. In later months, the name field is replaced by the field 'id_link', so if you do choose to use later datasets, go ahead and make this change.

```
    with open("/Volumes/Seagate Expansion Drive/RC_{}".format(timeframe), buffering=1000) as f:
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
```
Next, let's look at the score. We want to ensure that at least 1 person saw the comment (the score represents the upvote count), so to be extra safe, let's only proceed with designating this row as a reply of a paired row if the score is greater than or equal to 2 upvotes. If a reply already exists for that comment, look at the score of the comment. If the comment has a better score, then check that the data is acceptable, then update the row. However, if there isn't an existing comment score but there is a parent, insert with the parent's data instead.

```
            if score >= 2:
                existing_comment_score = find_existing_score(parent_id)
                if existing_comment_score:
                    if score > existing_comment_score:
                        if acceptable(body):
                            sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
```
If the score is not greater than or equal to 2 and the data is acceptable, we will check if the data is parent_data. If so, we will count this as the parent of a paired row. If there is no parent, then the comment itself is still considered the parent, so you want to use the insertion query sql_insert_no_parent. We want to do this because the comment still might be someone else's parent!
```
                else:
                    if acceptable(body):
                        if parent_data:
                            sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                            paired_rows += 1
                        else:
                            sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)
```
We will include a print statement that will help track how your data is processing. 
When you run your code, it will output a print statement when the program finishes looking through 100,000 rows. 
```
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))
```
Finally, let's run this code to create the database of paired rows. **This part is crucial and not made very clear in the tutorial.** 

Your code will take 5-10 hours to run. Mine took 6 hours, and another 3 hours to get this part right. If you are running into issues, check:
* Your data's path name that you included in open() which provides the connection to your data
* If you need to change ? to {} for your SQLite insertions
* If you need to change the 'name' field to 'id_link'

To run your code:
1. Open your terminal and navigate to the folder containing this Python script (mine is named chatbot_database.py)
2. In your terminal, type: ```$ python3 chatbot_database.py```
3. Check to see that your code matches or is very similar to the output below, and let it keep running. The amount of paired rows should increase ~4,000 to ~5,000 each time.

```
Total Rows Read: 100000, Paired Rows: 3221, Time: 2017-11-14 15:14:33.748595
Total Rows Read: 200000, Paired Rows: 8071, Time: 2017-11-14 15:14:55.342929
Total Rows Read: 300000, Paired Rows: 13697, Time: 2017-11-14 15:15:18.035447
Total Rows Read: 400000, Paired Rows: 19723, Time: 2017-11-14 15:15:40.311376
Total Rows Read: 500000, Paired Rows: 25643, Time: 2017-11-14 15:16:02.045075
```

Step 4: Partition the Data
--------------------------------
After you have finished pairing get ready for another timesuck. It will take 2 hours for your code to run this next, so make sure to set apart time to do so!

We will be using the data analysis pandas to help us create a data frame to visualize our data. Since the pandas library isn't included in Python3, import this package by navigating in your terminal to this folder, and typing in your terminal: ```pip3 install pandas```

```
import sqlite3
import pandas as pd

timeframes = ['2015-05']
```
Now, build the connection (remember how to do it?) and then create the labels. The label limit will represent how many rows we will pull at each time to show in the pandas dataframe, and last_unix will help us buffer through the database.

```
for timeframe in timeframes:
    connection = sqlite3.connect('/Volumes/Seagate Expansion Drive/{}.db'.format(timeframe))
    c = connection.cursor()
    limit = 5000 
    last_unix = 0 
    cur_length = limit
    counter = 0
    test_done = False
```
Now, we will write a while loop to keep making pulls to the dataframe until we reach the limit to show in the dataframe. Then once we reach the limit, we will put the data into the dataframe.

**NOTE: Since my computer was very slow at creating this dataframe, it would often stall so long that I had thought it stopped running. This was not made clear in sentdex's tutorial, but an easy fix is to just add your own print statement to confirm that your code is running properly. I included the ```print('Before, Time: {}'.format(str(datetime.now())))``` and ```        print('After, Time: {}'.format(str(datetime.now())))``` to ensure that you can see how long it takes in between each pandas pull and log the time to see how much time is left for your code to run.

```
    while cur_length == limit:
        # df = dataframe, * = all
        print('Before, Time: {}'.format(str(datetime.now())))
        df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)
        print('After, Time: {}'.format(str(datetime.now())))
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
```
Let's partition the testing data, and separate the parent ("from") and its corresponding reply ("to"). Let's do the same for the training data. Later, when you look at your data, you'll be able to see that they are paired matches, with a comment and a reply that makes sense!

*Aside: Training data and testing data are separated so that we can train the model to learn, and then we will still have enough data left to test the model to see if it learned what we wanted it to learn. The training dataset will always be significantly larger than the testing data, because the more data that the model is trained on, the more it will most likely learn.*

```
         if not test_done:
            with open('/Volumes/Seagate Expansion Drive/test.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')

            with open('/Volumes/Seagate Expansion Drive/test.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(str(content)+'\n')

            test_done = True
            
        else:
            # training data
            # "from" data is the comment and "to" data is the reply
            with open('/Volumes/Seagate Expansion Drive/train.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')

            with open('/Volumes/Seagate Expansion Drive/train.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(str(content)+'\n')
```
Now, let's increment the counter and keep the loop going. Every 20 x limit (since our limit is 5,000 then 20 x 5,000 = 100,000) rows we will see this information printed. 
```
        counter += 1
        if counter % 20 == 0:
            # 
            print(counter*limit,'rows completed so far')
```

Step 5: Train with [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot) 
------------------------------------------------------------------------------------
**Get ready for the motherlode of timesucks** - *training your model*.

This is where the biggest bugs and obstacles will arise. If you can't train your model, then all this hard work is for nothing, so you and I both will keep finding a way to make it work until it does. Before showing you how to run your model, let me first tell you the story of how I am still fighting this battle right now so you don't make the same mistakes as I had.

I originally naively began attemping to train my bot with my Macbook Pro, a pretty shiny thing will just 15 out of 120 GB available and obviously no graphics cards (GPUs) installed. With my database stored on an external hard drive, this was already posing an issue since the bottleneck of having the data stored outside of the CPU was already going to mean the model would take a very very very long time to train.

But, I didn't even get that far. I realized immediately that I was unable to install tensorflow-gpu, which is essential to training the model, on Macs because [it is no longer supported on macOS systems](https://www.tensorflow.org/install/install_mac).

So, I decided to try and train my model without tensorflow on a Mac with more storage. My boyfriend George Witteman graciously loaned me his own 512 GB Macbook Pro, and I trained a sample set of data on his computer around 50 hours ago. 
It's still running. At 100% CPU load. 

Unsure if this would work properly, I decided it would be worth it to pay money for a virtual environment that has GPU cards installed for faster training. Sentdex mentioned [Paperspace](https://www.paperspace.com/&R=IJHK5NF) so I decided to try it. *PS. this referral link gives you $5 in free credit if you want to use a virtual environment too.*
However, here's a warning: when you first sign up for Paperspace, you are not allowed to order a machine until you submit a written request that needs to be verified and approved by a Paperspace team member. So, another dead end.

Finally, as a last ditch effort, George dug up his old desktop PC that runs on Linux and has 1 TB of storage. I was not able to run tensorflow-gpu on this Linux system and with no GPU cards, the training still remains frustratingly slow. It has been 55 hours and it's still running at 100% CPU load. 

When Paperspace finally granted me the ability to order a virtual environment, it was 12 hours later. I went ahead anyways, but alas, I ran into problems with the Ubuntu operating system in the virtual environment. You cannot install tensorflow-gpu without installing multiple other pieces of software, which requires a much more time-intensive learning curve. I am now pursuing [this option,](https://www.tensorflow.org/install/install_linux) but it is costing me more hours to learn and download (with money too! costs $0.40 an hour and $6 a month on Paperspace). 

So, this is my current state: waiting for the data to finish training on two computers and learning how to train the dataset on a third server. Moral of the story? *Be willing to spend a long, long time, or a lot, lot of money.* 

**Now, if you have decided you are wholly prepared to train your model, let's begin**
As mentioned before, we will be using a set of utilities that uses Tensor Flow's [nmt model](https://github.com/tensorflow/nmt) called [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot) made by sentdex and his friend Daniel Kukiela. nmt-chatbot provides the toolset to train our chatbot, but it will require the following to train:
* tensorflow-gpu 1.4.0 (Use tensorflow if you don't have GPU support)
* Python 3.6+
* CUDA Toolkit 8.0 (Do not use if you don't have GPU support)
* cuDNN 6.1 - to install, see the [Windows](https://www.youtube.com/watch?v=r7-WPbx8VuY) tutorial or [Linux](https://pythonprogramming.net/how-to-cuda-gpu-tensorflow-deep-learning-tutorial) tutorial (Do not use if you don't have GPU support)

Follow these steps on your terminal (as adapted from [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot):
1. ```$ git clone --branch v0.1 --recursive https://github.com/daniel-kukiela/nmt-chatbot.git```
2. ```$ cd nmt-chatbot```
3. ```$ pip3 install -r requirements.txt```
4. ```$ cd setup```

If you've made it this far, open up the "new_data" folder and replace the sample training data with your own (which should already have been labeled train.to and train.from). **Now, here's a tricky part: regardless of whether or not you're using testing data from 2013 and 2012 or not (we aren't...) you want to make a copy of your test.from data and name the first copy tst2013.from and then name the second copy tst2012.from. Now, make a copy of your test.to data and name the first copy tst2013.to and then name the second copy tst2012.to. nmt-chatbot uses these exact filenames, so it is best to stick to their naming conventions. When I tried to modify the code for this, I was berated with errors, so this is the safest route.**

5. ```$ python prepare_data.py``` 
After running this line, a new folder called "data" will be created with prepared training data. 
6. ```$ cd ../```
7. ```$ python train.py```

Now, you have done all you can do to train your model and your last task is simply to wait. This could take hours, days, or even weeks. Sit back, disable your automatic sleep function on your computer, plug in your computer charger, and maybe invest in a fan to put underneath your laptop. Keep an eye on your terminal in case any errors pop up!

Step 6: Interact and Test
------------------------------
**Work in Progress! Data is still training. (It has been 50 hours at this point. I'm also using 2 separate servers...see below!)**
![alt text](/mac.jpeg?raw=true "George's Mac: still running TensorFlow on sample-sized data. (Not Shown) CPU is running at 100%")

Results and Reflection
-----------------------
**Work in Progress! Data is still training. (It has been 55 hours at this point. I'm also using 2 separate servers...see below!)**
![alt text](/pc.jpeg?raw=true "George's PC: still running TensorFlow-GPU on the May 2015 data. Notice that CPU is running at 100%")

Acknowledgements
-----------------
Thank you to sentdex and pythonprogramming.net for the amazing lessons, George Witteman for sacrificing both his computers for an infinite number of training hours, Tensor Flow's NMT model and sentdex & Daniel Kukiela's nmt-chatbot utility for making my learning experience significantly less painful, and Professor Josh deLeeuw for your patience and support!
