# Readme
## Improve Your Chess with Unsupervised Learning

### Description: 
#### Overview:
The goal of this project is to create chess hueristics using unsupervised learning. It is common knowledge that state-of-the-art chess engines easily dominate the best human players. This has been true for over a decade and is a major accomplishment of machine learning. Opinions on the effect of these engines on human play will vary depending on who you ask. Certainly, chess engines have accelerated the development of opening and endgame theory, but these developments are more relevant to the expert than the novice. I want to use machine learning to produce a set of principles of good play for the novice to use to advance to an above-average player. 

#### Process:
Focus is on the white player. All of the features are derived from his pieces and play. I developed two sets of features that are treated separately. The first set of features reflects the level of control the player has on each of the 64 squares of the chess board. I implemented a clustering algorithm which groups those games where the players' control of the board was similar. I then take the centriod as being representative of the group and use it to produce a heatmap which shows how the group controlled the chess board. I can see the distribution of the players' levels in each group, and I use this information to distinguish between how good players control the board versus how bad players control the board.

The second set of features reflects the placement of the players pieces. This set of features needs a little more set up. Basically, we count how many times each type of piece (Pawn, knight, etc.) sits on each square througout the game. For example, if a pawn sits on the square F2 for 20 moves in a particular game, the feature called "F2_Pawn" (or something similar) will have value 20. Long story short, the clustering algorithm will tend to group those games which have the same fixed-piece structures. These fixed structures mostly consist of pawns, thus we see what types of pawn structures good/bad players are using. 

#### Results:
In the images folder, you will find two subfolders PCA and NMF. These names refer to the dimension reduction method used. If you run chess_project.py, both folders will be populated with images. In my analysis, I focused on the results from PCA as that method seemed to work best. In the PCA folder, you have two more subfolders - 'sqr_attacks' and 'piece_locations'. The first contains the heat maps for each of the groups described in the previous section - red means more than average control and blue means less than average control. There is also a box plot and a line chart (average elo) which indicates the level of the white player in the games from each group. In the 'piece_locations' folder, you see the fixed-piece structures from each group represented on chess boards and graphs which indicate the chess-strength of the groups.



### Instructions:
1) Download my database containing games and features: https://drive.google.com/open?id=1JOy-OZgNoVw0q7Xk2BlZhLGkkMpIDHQY. 
2) Make sure MongoDB is running. Go to the chess_project directory in the terminal. Load the database from the command line: 
  mongorestore --db=chess_games --collection=pgns_2017_10 pgns_2017_10.bson
3) Run chess_project.py


### File Directory:
- chess_project.py: Contains scripts for loading data from database, performing clustering, and creating vizualizations
- chess_slides_v2.pdf: Slides prepared for Metis ~ 11/2019
- requirements.txt: Required software and packages





