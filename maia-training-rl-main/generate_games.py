
import chess
import chess.pgn
import random
import numpy as np
from collections import defaultdict
import sys
import os
from shutil import copyfile
import multiprocessing
import pymongo
import pickle
import time

from dotenv import load_dotenv

# prod - training config to run on HPC
NUM_PROCESSORS = 8
GAMES_PER_PROCESSOR = 625
NUM_LC0_NODES = 1500

# testing - local config (uncomment to use)
# NUM_PROCESSORS = 2
# GAMES_PER_PROCESSOR = 5
# NUM_LC0_NODES = 100

# Note: for local testing, you would also wanna adjust the --backend-opts / --backend flags in the ageng initialisation under worker()
# for example, --backend=cpu or --backend=auto, or if you have a Macbook with M chip, --backend=metal

# Note: Calculation for total number of games
# total games would be NUM_PROCESSORS * GAMES_PER_PROCESSOR * 2 
# the *2 is because each iteration in worker() plays 2 games (team1 vs team2 and team2 vs team1)
# For example,
# with the prod config above, total games = 8 * 625 * 2 = 10000 games

# load environment variables from .env
load_dotenv()
myclient = pymongo.MongoClient(os.getenv('MONGODB_CONNECTION_STRING') or "INVALID")
mydb = myclient[os.getenv('DB_NAME') or "test"]
mycol = mydb[os.getenv('COLLECTION_NAME') or "training_games"]

# some config to change
name_of_run="genlogs/"+"name_of_run"
partner_names=['partner1','partner2']
weight_files=['../maia-partner/models/128x10-t60-2-5300.pb.gz',
            '../maia-partner/models/128x10-t60-2-5300.pb.gz']
MAIA_WEIGHTS="../maia-partner/models/maia-1100.pb.gz"

def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_
def argmax(L):
    return int(np.random.choice(allmax(L)))

class team:
    def __init__(self,player,partner,p,name=""):
        print(f"Creating team with player: {player.name}, partner: {partner.name}")
        self.name=name
        if(name==""):
            self.name=player.name+partner.name
        self.player=player
        self.partner=partner
        self.p=p
        teams_dic[self.name]=self
        print(f"Team {self.name} created successfully")
    def info(self):
        return {'name':self.name ,'player':self.player.info(), 'partner':self.partner.info(), 'p':self.p}
    def play(self,board,plystring=[], valstring=[]):
        help=random.uniform(0,1)
        if(help<self.p):
            plystring.append('1')
            y=self.partner.play(board)
            valstring.append(y.info['score'].white().score(mate_score=100000))
            return y
        else:
            plystring.append('0')
            valstring.append('No')
            return self.player.play(board)
            
    def completed(self,res=True):
        self.player.completed(res)
        self.partner.completed(res)
class agent:
    def __init__(self,name,config,nodes,reset_thresh=1):
        self.name=name
        self.config=config
        self.engine=chess.engine.SimpleEngine.popen_uci(config)
        self.reset_thresh=reset_thresh
        self.nodes=nodes
        self.counter=0
        self.reset_thresh=reset_thresh
    def info(self):
        return {'name': self.name, 'nodes':self.nodes, 'config':self.config}
    def play(self,board):
        try:
            result = self.engine.play(board, chess.engine.Limit(nodes=self.nodes), info=chess.engine.Info(2))
            # print(f"Agent {self.name} decided on move: {result.move}")
            return result
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            self.reset(True)
            print("ERRED")
            return self.play(board)
    def analyze(self,board):
        if(board.is_game_over()):
            ok=board.outcome().winner
            if(ok):
                return 99999
            else:
                return -99999
        r=self.engine.analyse(board,chess.engine.Limit(nodes=self.nodes))['score'].white().score(mate_score=100000)
        return r
    def reset(self,res):
        self.engine.quit()
        if(res):
            self.engine=chess.engine.SimpleEngine.popen_uci(self.config)
    def completed(self,res):
        self.counter+=1
        if(self.counter%self.reset_thresh==0):
            self.reset(res)


def play(white, black):
    curgame={}
    plystring=[]
    valstring=[]
    numplies=0
    board = chess.Board()
    game=chess.pgn.Game()
    t=0
    while not board.is_game_over():
        numplies+=1
        if(t%2==0):
            result = white.play(board,plystring,valstring)
        else:
            result = black.play(board,plystring,valstring)
        board.push(result.move)
        if(t==0):
            node=game.add_variation(chess.Move.from_uci(str(result.move)))
        else:
            node=node.add_variation(chess.Move.from_uci(str(result.move)))
        t+=1
    ok=board.outcome().winner
    plystring=''.join(plystring)
    if(ok==None):
        winner=0
    elif(ok):
        winner=1
    else:
        winner=-1
    curgame['winner']=winner
    curgame['length']=numplies
    curgame['move_identities']=plystring
    curgame['board_values']=valstring
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter)
    pgn_string=pgn_string.replace('\n'," ")
    curgame['pgn']=pgn_string
    coin=random.uniform(0,1)
    if(coin<0.8):
        settype='tr'
    elif(coin<0.9):
        settype='va'
    else:
        settype='te'
    curgame['set']=settype
    curgame['white']=white.info()
    curgame['black']=black.info()
    curgame['name']=curgame['white']['name']+'self'
    print(f"Game finished after {numplies} moves. Winner: {winner}")
    return curgame

teams_dic={}

#assumption of defaults
def worker(partner1_name, partner2_name, partner1_weights, partner2_weights,i,q):
    print(f"Worker {i} starting...")
    open(name_of_run+'/trash'+str(i), 'a').close()
    sys.stderr = open(name_of_run+'/trash'+str(i), 'w')
    maiay=agent('m11',["lc0" , "--temperature=1", "--weights="+MAIA_WEIGHTS, "--backend-opts=gpu="+str((i//2)%4)],1)
    partnery=agent(partner1_name,["lc0" , "--weights="+partner1_weights, "--backend-opts=gpu="+str((i//2)%4)], NUM_LC0_NODES) 
    team1=team(maiay,partnery,0.5)
    
    maiay2=agent('m11_2',["lc0", "--temperature=1", "--weights="+MAIA_WEIGHTS, "--backend-opts=gpu="+str((i//2)%4)],1)
    partnery2=agent(partner2_name+'_2',["lc0", "--weights="+partner2_weights, "--backend-opts=gpu="+str((i//2)%4)], NUM_LC0_NODES)   
    team2=team(maiay2,partnery2,0.5)

    print(f"Worker {i}: Both teams created, starting game loop...")
    for i in range(GAMES_PER_PROCESSOR):
        print(f"Worker {i}: About to start game {i+1}")
        print(f"Worker {i}: Calling play(team1, team2)...")
        r1=play(team1,team2)
        r2=play(team2,team1)
        print(f"Worker {i}: Game {i+1} completed")
        if(i%100==0 and i>0):
            team1.completed(True)
            team2.completed(True)    
        # print(r1,r2)
        q.put((r1,r2))
    team1.completed(False)
    team2.completed(False)
        

    return (r1,r2)
def choosenext(result_matrix,pre_totals_matrix):
    matchup_eligibility=np.zeros((n,n),dtype=int)
    for i in range(1,n):
            matchup_eligibility[0][i]=100000-pre_totals_matrix[0][i]
    selected=(0,np.argmax(matchup_eligibility[0]))
    pre_totals_matrix[selected]+=2
    return selected

def scorear():
    return [0,0,0]
def controller(num_proc):
    print(f"Starting controller with {num_proc} processes...")
    load_path="battle_results/"+"officialfocused"
    team_names=['m11'+x for x in partner_names]
    proc_ar=[]
    nowar=[]
    q=multiprocessing.Queue()
    
    for i in range(num_proc):
        print(f"Creating process {i}...")
        now=(0,1)
        p = multiprocessing.Process(target=worker,args=(partner_names[now[0]],partner_names[now[1]],weight_files[now[0]],weight_files[now[1]],i,q))
        proc_ar.append(p)
        nowar.append(now)
        p.start()

    games_processed = 0
    expected_games = num_proc * 5 * 2  # num_proc * games_per_worker * results_per_game
    print(f"Expecting {expected_games} total game results...")
    
    while games_processed < expected_games:
        if(not q.empty()):
            print(f"Processing result {games_processed + 1}/{expected_games}")
            res=q.get()
            # if(random.uniform(0,1)<0.001):
            #     print(res)
            mycol.insert_one(res[0])
            mycol.insert_one(res[1])
            games_processed += 2  # We process 2 games at once (res[0] and res[1])
    
    print("All games processed, waiting for processes to finish...")
    for p in proc_ar:
        p.join()  # Wait for all processes to complete
    
    print(f"Controller finished! Total games saved: {games_processed}")



if __name__=="__main__":
    os.makedirs(name_of_run,exist_ok=True)

    print("Testing MongoDB connection...")
    try:
        myclient.admin.command('ping')
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

    # Uncomment the following blocks to test Lc0 engine initialization (adjust backend flags as needed)

    # print("Testing Lc0 Maia engine creation...")
    # try:
    #     test_engine = chess.engine.SimpleEngine.popen_uci(["lc0", "--backend=metal", "--weights=../maia-partner/models/maia-1100.pb.gz"])
    #     print("Test engine created successfully")
    #     test_engine.quit()
    #     print("Test engine closed successfully")
    # except Exception as e:
    #     print(f"Engine creation failed: {e}")
    #     sys.exit(1)

    # print("Testing Lc0 Leela engine creation...")
    # try:
    #     test_engine = chess.engine.SimpleEngine.popen_uci(["lc0", "--backend=metal", "--weights=../maia-partner/models/128x10-t60-2-5300.pb.gz"])
    #     print("Test engine 2 created successfully")
    #     test_engine.quit()
    #     print("Test engine 2 closed successfully")
    # except Exception as e:
    #     print(f"Engine creation failed: {e}")
    #     sys.exit(1)

    start_time = time.time()
    controller(NUM_PROCESSORS)
    end_time = time.time()
    print(f"Controller finished in {end_time - start_time} seconds")