# Readme
## Chess - Unsupervised Learning

### Description: 
#### Overview:
The goal of this project is to create chess hueristics using unsupervised learning. It is common knowledge that state-of-the-art chess engines easily dominate the best human players. This has been true for over a decade and is a major accomplishment of machine learning. Opinions on the effect of these engines on human play will vary depending on who you ask. Certainly, chess engines have accelerated the development of opening and endgame theory, but these developments are more relevant to the expert than the novice. I want to use machine learning to produce a set of principles of good play for the novice to use to advance to an above-average player. 
#### Process:

#### Results:



### Instructions:
1) Download my database containing games and features: https://drive.google.com/open?id=1JOy-OZgNoVw0q7Xk2BlZhLGkkMpIDHQY. 
2) Make sure MongoDB is running. Go to the chess_project directory in the terminal. Load the database from the command line: 
  mongorestore --db=chess_games --collection=pgns_2017_10 pgns_2017_10.bson
3) Run chess_project.py


### File Directory:
- chess_project.py: Contains scripts for loading data from database, performing clustering, and creating vizualizations
- chess_slides_v2.pdf: Slides prepared for Metis ~ 11/2019





