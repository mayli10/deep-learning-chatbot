A Beginner's Guide to Building Sentdex's Deep Learning Chatbot
===================

Table of Contents
-------------
1. [Foreword](#foreword)
2. [Prerequisites: Methods and Tools](#prerequisites-methods-and-tools)
3. [Introduction](#introduction)
4. [What is Deep Learning and NMT?](#what-is-deep-learning-and-nmt)
5. [Datasets](#datasets)
6. [Setup](#setup)
7. [Step 1: Structure the Database](#step-1-structure-the-database)
8. [Step 2: Clean the Data](#step-2-clean-the-data)
9. [Step 3: Write SQL Insertions](#step-3-write-sql-insertions)
10. [Step 4: Build Paired Rows](#step-4-build-paired-rows)
11. [Step 5: Partition the Data](#step-5-partition-the-data)
12. [Step 6: Train with nmt-chatbot](#step-6-train-with-nmt-chatbot)
13. [Step 7: Interact and Test](#step-7-interact-and-test)
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

I began my deep learning journey with a grand idea - I wanted to build a chatbot with functions that I hoped could improve mental healthcare. Because my current college, Vassar College, doesn't offer any machine learning courses, I found my way into a Deep Learning independent study with Professor Josh deLeeuw and began to self-teach with Dr. Andrew Ng's deeplearning.ai online class. 
But, after continuously finding myself lost in the dense mathematical jargon and beginner-unfriendly tutorials, I realized that I needed to find an alternative. Thus, I stumbled upon sentdex's [tutorials](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/), and found the extensive explanations to be a wonderful relief. 

However, I realized that there is still a signficant learning curve involved for those, like me, who have limited experience with machine learning or Python. While the tutorials are clear to understand, there are multiple bugs, software incompatibilities, and hidden or unexpected technical difficulties that arose when I completed this tutorial. There were many challenges that were near-impossible to solve without consulting external sources of knowledge or extensive research, and many hidden prerequisites that almost forced me to quit my journey through the tutorial as many other people have done. 

My primary goal in building this chatbot is to first understand the foundations for building a deep learning chatbot, and then curating my chatbot to address a specific need in the mental health care industry. My secondary goal is to provide the essentials tips and bug fixes that have not been properly documented in the original tutorial and that I have learned through my own experience. I realized that without this supplemental information, I would not have been able to complete the tutorial by my own. 
Thus, this tutorial serves as both an everything-you-need-to-know walk-through for those who are just beginning to build deep learning models as well as a documentation of my own journey of building this bot.

**Thus, I decided to document my experience and create this deep-dive beginner-oriented tutorial which will help ease the bugs that arise. All material has been adapted from [sentdex](https://www.youtube.com/watch?v=dvOnYLDg8_Y&t=140s). This tutorial serves as both an everything-you-need-to-know walk-through for those who are just beginning to build deep learning models as well as a documentation of my own journey of building this bot!**



Prerequisites: Methods and Tools
---------------------------------
It's essential that you have these prerequisites to even be able to proceed with this tutorial. Otherwise, you can always find a solution, but this tutorial will mostly like not have the answers to those issues.
You must have:

1. **Latest versions of Python3.6+ (the programming language we are using) and Pip3 (package management system for Python3) installed** 
2. **At least 50 GB of unused storage.** The dataset we'll be using is 33.5 GB, but you'll need even more (~8 GB) later on. Free up storage with your pc's Storage Management System or use an external hard drive. 
3. **An IDE, such as Atom or Sublime.** Any text editor will work, but it is recommended that you have access to a debugger.
4. **Enough time.** There are 3 timesucks to this project: Step 4 (~6 hours), Step 5 (~2.5 hours), and Step 6 (~40 hours). *Note: I used an external hard drive, so the speed and time it has taken me to run my code is likely to be slower than average. However, every computer is different based on numerous factors such as memory and even internet speed, so always plan to budget more time than expected.* If you want to cut down on time, prepare to trade with money. As I will mention in Step 4, 5, and 6, there are alternatives to explore.

Setup
------
If you want to check out the chatbot that I have built, follow these steps. Note: to run this, you must still have all the prerequisites mentioned above! Otherwise, continue with the tutorial to build your own! 

1. Open up your terminal, and type: ```$ git clone --recursive https://github.com/mayli10/deep-learning-chatbot```
2. Make sure you are in the deep-learning-chatbot folder that you just created. 
   Then in your terminal, type: ```$ python3 hello_chatbot.py``` This will show questions/comments and its corresponding replies.
3. If you would like to talk to the chatbot live, then navigate out of the deep-learning-chatbot folder, and clone sentdex's helper utilities repository. 
   In your terminal, type: ```$ git clone --branch v0.1 --recursive https://github.com/daniel-kukiela/nmt-chatbot.git```
4. We will now use the inference utility. In your terminal, type: ```$ python3 inference.py < hello_chatbot.py```
5. If you are having trouble, use sentdex's demo [here](https://github.com/daniel-kukiela/nmt-chatbot/blob/master/README.md#demo-chatbot)

Datasets
--------- 
If you are new to machine learning, a good tip to remember is that the most important and difficult aspect of machine learning is finding enough of the correct training data to train the model on. Training the model could be expensive and time-consuming, and we also need to find the specific type of data to train with. Some good dataset sources for future projects can be found at r/data, uci repository, or kaggle. The larger the dataset, the more information the model will have to learn from, and (usually) the better your model will have learned. But, since we are constrained by the memory of our computers or the monetary cost of external storage, let’s build our chatbot with the minimal amount of data needed to train a decent model. 

Even this amount of data is not tiny. We will be using all the Reddit comments from May 2015, representing just 54 million rows of a dataset containing 1.7 billion Reddit comments. This month of data will be more than enough to train a decent model, but if you want to take it one step further beyond the basics, read more at sentdex’s tutorial here. Download the May 2015 data here, and if you want to view the full dataset, you can find it here and here. 

If you are unfamiliar with Reddit, the comments are structured in a non-linear tree structure. A Redditor makes a post, and other Redditors comment on the post. There can be:
1.	Comment with no replies
2.	Comment with a reply
3.	Comment with a reply that has a reply

Because we just need a comment (input) and reply (output) pair, we will be addressing how to filter out the data so that we pick comment-reply pairs. Furthermore, if there are multiple replies to the comment, we will pick the top-voted reply. We will address this issue at Step 5.

**How to download:**
Make sure that you have at least 50 GB of free space on your computer. Use software such as Application Deleter, clean my macbook, or the storage manager to do this, but if you still don’t have enough, a good way to address this issue would be to buy an external hard drive. The one I am using is ___, and it contains _GB of space for a relatively cheap price of __. I began with using software to make space for the data, but after multiple efforts and many hours of whittling down my Applications folder, it made sense to just use an external hard drive. It will definitely be slower to use the hard drive, but if it’s the last option for you, then it’s still a viable option.

If you’re using an external storage drive, plug in your drive and make sure to download your file directly into the drive.
If neither of these options work, another option is to use AWS or Paperspace. Skip down to step 5 to learn more about Paperspace if you choose this option.

Deep Learning vs. Machine Learning
--------------
Deep learning is a type of machine learning that uses feature learning to continuously and automatically analyze data to detect features or classify data. Essentially, deep learning uses a larger amount of layers of algorithms in models such as a recurrent neural network or deep neural network to take machine learning a step further. While machine learning learns using algorithms and makes informed decisions, deep learning is a type of machine learning that furthermore uses these algorithms with more networks of layers to make intelligent inferred decisions. While wit machine learning, the engineer needs to provide the features that the model needs for classification, deep learning automatically discovers these features itself.  Although deep learning generally needs much more data to train than machine learning, the results are often much more advanced than that of machine learning.

Chatbot with Neural Machine Translation (NMT)
--------------
*Note: the following section provides somewhat dense technical information to assist in your understanding towards the model that is used for the chatbot. However, if this is too difficult to follow, come back to this section later when you are about to train and use your model with [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot)*

Building a chatbot with deep learning is an exciting approach that is radically different than building a chatbot with machine learning. **We want to build a chatbot that can make its own inferences and detect features to use that we don’t explicitly define for them.** With a machine learning chatbot, we would give the bot a set of intents, which are the intentions of the user’s utterance to the bot, and entities, such as the descriptors the user utters. For example, a user could say to the bot, “Tell me your name,” and the engineer would have specified that “tell” is an intent and “name” is an entity.

However, in deep learning, the process is much different. We will not be specifying features to the bot, but will instead expect the bot to detect these features itself and respond appropriately. Particularly, we will be using **Neural Machine Translation (NMT),** which is a vast artificial neural network that uses deep learning and feature learning to model full sentences with machine translation. This approach specializes in producing continuous sequences of words better than the traditional approach of using a recurrent neural network (RNN) because it mimics how humans translate sentences. Essentially, when we translate, we read through the entire sentence, understand what the sentence means, and then we translate the sentence. NMT performs this process with an encoder-decoder architecture; the encoder transforms the sentence into a vector of numbers that represent the meaning of the sentence and then the decoder takes this meaning vector and constructs a translated sentence. Thus, at its core, an NMT model is a deep multi-layer with 2 Bi-Directional Recurrent Neural Networks: the encoder BRNN and the decoder BRNN. Here is an example from sentdex’s tutorial that shows this architecture:

This **sequence-to-sequence** model (colloquially referred to in the ML community as seq2seq) is often used for machine translation, text summarization, and speech recognition, and TensorFlow provides a tutorial on building your own NMT model [here](https://github.com/tensorflow/nmt). As a beginner, I found that this tutorial was a little too dense to understand, so I recommend using sentdex’s NMT [model](https://github.com/daniel-kukiela/nmt-chatbot) built specifically for this tutorial that includes additional utilities along with a pre-trained NMT model. (Aside: I intend to build my own seq2seq model after further self-learning, as my current attempts at building this model have been insufficient for use in building this deep learning chatbot).

It is essential that we use Bi-Directional Recurrent Neural Networks because with organic human language, there is value in understanding the context of the words or sentences in relation to other words and sentences. Because both the past, present, and future data in the sentence is important to remember and know to understand the sentence as a whole, we need a neural network that has an input sequence that can go both ways (forward and reverse) to understand a sentence. What differentiates the BRNN from a simple RNN is this ability, which is due to a hidden layer between the input and output layers of the network that can pass data both forwards and backwards, essentially giving the network the ability to understand previous, present, and incoming input as a cohesive whole.

Step 1: Structure the Database 
-------------------------------
Now that you have your data, let’s look at one row of JSON data:
```{"author":"Arve","link_id":"t3_5yba3","score":0,"body":"Can we please deprecate the word \"Ajax\" now? \r\n\r\n(But yeah, this _is_ much nicer)","score_hidden":false,"author_flair_text":null,"gilded":0,"subreddit":"reddit.com","edited":false,"author_flair_css_class":null,"retrieved_on":1427426409,"name":"t1_c0299ap","created_utc":"1192450643","parent_id":"t1_c02999p","controversiality":0,"ups":0,"distinguished":null,"id":"c0299ap","subreddit_id":"t5_6","downs":0,"archived":true}```

The most important fields that we will factor in are parent_id, comment_id, body, name, and score. 

Let’s talk about the paired comment-replies in more detail. 
Because we need an input and an output, we need to pick comments that have at least 1 reply as the input, and the most upvoted reply (or only reply) for the output. We also need to consider the third case, where the comment’s reply has a reply. In this case, we would need to consider the comment as the input, and the comment’s first reply as the output. But, the comment’s first reply would then also serve as a comment itself with its own paired reply, so essentially we would have two paired rows, one nested in another. This brings us to the logical conclusion that we must be careful not to only check if there is a reply for a comment, we must also see if the comment is itself a reply to another comment. Since every comment can potentially have a reply, every comment will be considered a parent.

Step 2: Clean the Data
------------------------

Step 3: Write SQL Insertions 
----------------------------------

Step 4: Build Paired Rows
--------------------------

Step 5: Partition the Data 
----------------------------

Step 6: Train with [nmt-chatbot](https://github.com/daniel-kukiela/nmt-chatbot) 
--------------------------------
I originally began with my mac, but it soon became clear that I wouldn't be able to even begin without using tensorflow-gpu, which isn't supported on OS operation systems anymore.
So, George Witteman graciously pulled out his Linux system 8gb of memory and 1.5 tb of storage to train on. Even then, it took 11 hours to train, and it hadn't even finished training.

So after getting Paperspace approved, I decided to spring $10 and go ahead. Use this link for $10 of free credit for a virtual environment. 
Problems with Ubuntu - you need to download tensorflow gpu with a series of other pieces of software. You can take this route, but it's time intensive.
I decided to just leave it to my sample data.

Step 7: Interact and Test
------------------------------

Results and Reflection
-----------------------

Acknowledgements
-----------------
Thank you to sentdex, pythonprogramming.net, George Witteman, and Professor Josh deLeeuw for all your help and support.  
