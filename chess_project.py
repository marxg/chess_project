import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.svg
import cairosvg

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler,scale
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer 
from pymongo import MongoClient


# from chess.pgn import read_game
# import chess
# import chess.svg

def load_features(moves=80, exclude_elo=[1350,1750], Tfidf=False, features='sqr_attacks'):
    """
    Select games and features to analyse. My Mongodb database is in 
    the repo. The database name is "chess_games", and the collection 
    name is "pgns_2017_10". If you use different names, make the adjustment
    noted below. 
    Parameters:
    moves: Number of moves from the start to use for analysis. If a 
        game has less than this number, then the game is skipped. The 
        value should be <= 80 because the features were not collected 
        beyond the first 80 states.
    exclude_elo: ELO range to exclude. We want to keep good/bad 
        players and exclude average players.
    tfidf: Not sure about its effectiveness in this setting. For 
        more info, see https://en.wikipedia.org/wiki/Tf-idf. 
    features = 'sqr_attacks' or 'piece_locations'

    Returns: (feature_names, bag_of_features, game_list)
    bag_of_features : The dataset. Columns are features and rows
        are chess games. 
    feature_names : Name of columns / features
    games_list : List of games in same matching order of 
        bag_of_features. Entries are a list with game id, 
        white elo, and result.    
    """
    client=MongoClient()
    games=client.chess_games.pgns_2017_10
    #Adjust if you are using different names
    #e.g. games=client.your_database.your_collection
    games=list(games.find({}))
    game_dicts=[]
    game_list=[]
    for game in games:
        if len(game['game_states'])<moves:
            continue
        if exclude_elo[1]>int(game['WhiteElo'])>exclude_elo[0]:
            continue
        game_dicts.append({**game[features]})
        game_list.append([game['_id'], 
                          game['WhiteElo'], 
                          1 if game['Result']=='1-0' else 0 if game['Result']=='0-1' else .5])
    bag_of_features=pd.DataFrame(game_dicts)
    feature_names=bag_of_features.columns
    bag_of_features=bag_of_features.fillna(0)
    if Tfidf:
        TT=TfidfTransformer()
        bag_of_features=TT.fit_transform(bag_of_features)
        bag_of_features=sp.sparse.csr_matrix.todense(bag_of_features)
    return (feature_names, bag_of_features, game_list)

def reduce_cluster(reducer=PCA, components=30, clusters=10, features='sqr_attacks'):
    """
    Dimension reduction and clustering.
    features : 'sqr_attacks' or 'piece_locations' 
    reducer : Dimension reduction method. Viz looks best with PCA
    components = Reduce to this many principal components
    clusters = # of clusters
    """
    feature_names, bag_of_features, game_list = load_features(features = features)
    #REDUCTION STEP
    re_name=reducer.__name__
    re=reducer(n_components=components)
    reduced_bag=re.fit_transform(bag_of_features)
    #PRINT OUT OF METRICS 
    print(f"{re_name}:")
    if re_name=='PCA':
        print(f"Explained variance ratio sum {sum(re.explained_variance_ratio_)}")
        #print(f"Explained variance ratios {re.explained_variance_ratio_}")
        #print(f"Singular values {re.singular_values_}")
        #print("Top ten features of first component:")
        #pprint(sorted(list(zip(re.components_[0], feature_names)), reverse=True)[:10])
    if re_name=='NMF':
        print(f"Error-norm ratio {re.reconstruction_err_/np.linalg.norm(bag_of_features)}")
    #KMEANS
    ###################################################
    #INCLUDE SECTION BELOW TO TEST DIFFERENT K VALUES
#     inertia_list=[]
#     n=20
#     for i in range(1,n):
#         means = KMeans(n_clusters = i, random_state=20)
#         means.fit(reduced_bag)
#         inertia_list.append(means.inertia_)
#    sns.lineplot([i for i in range(1,n)], inertia_list, markers=True)
#    plt.show()
    ####################################################
    #print(f'KMEANS K={clusters}:')
    means = KMeans(n_clusters = clusters, random_state=40)
    cluster_fit=means.fit_predict(reduced_bag)
    # for i in range(clusters):
    #     print(f'Size of cluster {i}: {sum(1 for x in cluster_fit if x==i)}')
    km_groups=list(zip(game_list, cluster_fit, reduced_bag))
    #km_groups.sort(key=lambda x: x[1])
    #################### 
    plt.subplots(figsize=(4, 4))
    sns.boxplot([int(x[1]) for x in km_groups], [int(x[0][1]) for x in km_groups], color= "firebrick")
    plt.savefig(f"./images/{re_name}/{features}/group_elo_box.svg", pad_inches=0)
    plt.close()
    ####################
    sns.set() 
    plt.subplots(figsize=(4, 4))
    sns.lineplot([int(x[1]) for x in km_groups], [int(x[0][1]) for x in km_groups], ci=None)
    plt.xticks(np.arange(0, clusters, 1.0))
    plt.savefig(f"./images/{re_name}/{features}/group_elo_line.svg", pad_inches=0)
    plt.close()
    #################### 
    # plt.subplots(figsize=(8, 2))
    # sns.lineplot([int(x[1]) for x in km_groups], [int(x[0][2]) for x in km_groups])
    # plt.show()
    # #pprint(sorted(list(zip( means.cluster_centers_[4] @ re.components_, feature_names)), reverse=True))
    split_names=[x.split('_') for x in feature_names]
############################# Making HEAT MAPS ######################################
    if features=='sqr_attacks':
        for i in range(clusters):
            cluster_features=list(zip(means.cluster_centers_[i] @ re.components_, split_names))
            #print(cluster_features)
            atk_board=[]
            for j in range(64):
                #atk_board.append(sum(x for x,y in cluster_features if y[0]=='atk' and int(y[2])==j))
                atk_board.append(sum(x for x,y in cluster_features if y[0]=='atk' and int(y[2])==j))
            atk_board=np.array(atk_board).reshape(8,8)[::-1]
            #print(f'Attack board for cluster {i}:')
            plt.subplots(figsize=(4, 4))
            if re_name=="PCA":#Change the scaling
                cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
                sns.heatmap(atk_board, center=0, vmin=-40, vmax=40, cmap=cmap, 
                            xticklabels=False, yticklabels=False, cbar=False)
            else:
                sns.heatmap(atk_board, vmin=0, vmax=150, cmap='Reds', 
                            xticklabels=False, yticklabels=False, cbar=False)
            ### fix for mpl bug that cuts off top/bottom of seaborn viz ##
            b, t = plt.ylim() # discover the values for bottom and top
            b += 0.5 # Add 0.5 to the bottom
            t -= 0.5 # Subtract 0.5 from the top
            plt.ylim(b, t) # update the ylim(bottom, top) values
            plt.savefig(f"./images/{re_name}/{features}/group_{i}.svg", pad_inches=0)
            plt.close()
        #################################################################
    else:
        for i in range(clusters):
            cluster_features=list(zip(means.cluster_centers_[i] @ re.components_, split_names))
            location_board=[]
            location_power=[]
            for j in range(64):
                if re_name=='PCA':
                    lim=10
                else:
                    lim=15
                mx=lim
                piece=0
                for n in range(1,7):
                    mx_new=sum(x for x,y in cluster_features if len(y)==2 and int(y[0])==n and int(y[1])==j)
                    if mx_new > mx:
                        piece=n
                        mx=mx_new
                location_board.append(piece)
                location_power.append(mx)
            board=chess.Board('8/8/8/8/8/8/8/8')
            for k,x in enumerate(location_board):
                if not x:
                    continue
                piece=chess.Piece(x, True)
                board.set_piece_at(k, chess.Piece(x,True))
            bytestring=chess.svg.board(board=board)
            cairosvg.svg2png(bytestring, write_to=f"./images/{re_name}/{features}/group_{i}.png")

if __name__ == "__main__":
    reduce_cluster(features='sqr_attacks')
    reduce_cluster(features='piece_locations')
    reduce_cluster(reducer=NMF, features='sqr_attacks')
    reduce_cluster(reducer=NMF, features='piece_locations')