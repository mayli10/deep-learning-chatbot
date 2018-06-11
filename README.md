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
10. [Step 4: Build the Database - Paired Rows](#step-4-build-the-database---paired-rows)
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

I began by 

Prerequisites: Methods and Tools
---------------------------------
It's essential that you have these prerequisites to even be able to proceed with this tutorial. Otherwise, you can always find a solution, but this tutorial will mostly like not have the answers to those issues.
You must have:

1. **Latest versions of Python3 (the programming language we are using) and Pip3 (package management system for Python3) installed** 
2. **At least 50 GB of unused storage.** The dataset we'll be using is 33.5 GB, but you'll need even more (~8 GB) later on. Free up storage with your pc's Storage Management System or use an external hard drive. 
3. **An IDE, such as Atom or Sublime.** Any text editor will work, but it is recommended that you have access to a debugger.
4. **Enough time.** There are 3 timesucks to this project: Step 4 (~6 hours), Step 5 (~2.5 hours), and Step 6 (~40 hours). *Note: I used an external hard drive, so the speed and time it has taken me to run my code is likely to be slower than average. However, every computer is different based on numerous factors such as memory and even internet speed, so always plan to budget more time than expected.* If you want to cut down on time, prepare to trade with money. As I will mention in Step 4, 5, and 6, there are alternatives to explore.

Setup
------

Datasets
--------- 


What is Deep Learning and NMT?
--------------

Step 1: Structure the Database 
-------------------------------

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
I originally began with my mac, but it soon became clear that I wouldn't be able to even begin without using tensorflow-gpu, which isn't supported on os operation systems anymore.
so, my boyfriend @george witteman graciously pulled out his linux system 8gb of memory and 1.5 tb of storage to train on. even then, it took 11 hours to train, and it hadn't even finished training 

so after getting paperspace approved, i decided to spring $10 and go ahead. use this link for $10 of free credit for a virtual environment. we need
problems with ubuntu - you need to download tensorflow gpu with a series of other pieces of software. you can take this route, but it's time intensive.
I decided to just leave it to my sample data.

Step 7: Interact and Test
------------------------------

Results and Reflection
-----------------------

Acknowledgements
-----------------
Thank you to sentdex, pythonprogramming.net, and George Witteman for 
