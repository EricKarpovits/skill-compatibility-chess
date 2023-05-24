
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
        self.name=name
        if(name==""):
            self.name=player.name+partner.name
        self.player=player
        self.partner=partner
        self.p=p
        teams_dic[self.name]=self
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
            return self.engine.play(board,chess.engine.Limit(nodes=self.nodes),info=chess.engine.Info(2))
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
    return curgame

teams_dic={}

#assumption of defaults
def worker(partner1_name, partner2_name, partner1_weights, partner2_weights,i,q):
    open(name_of_run+'/trash'+str(i), 'a').close()
    sys.stderr = open(name_of_run+'/trash'+str(i), 'w')
    maiay=agent('m11',["lc0" , "--temperature=1", "--weights=maia-1100.pb.gz","--backend-opts=gpu="+str(i%4) ],1)
    partnery=agent(partner1_name,["lc0" , "--weights="+partner1_weights,"--backend-opts=gpu="+str((i+1)%4)],1500)   
    team1=team(maiay,partnery,0.5)
    maiay=agent('m11',["lc0" , "--temperature=1", "--weights=maia-1100.pb.gz","--backend-opts=gpu="+str((i+2)%4) ],1)
    partnery=agent(partner2_name,["lc0" , "--weights="+partner2_weights,"--backend-opts=gpu="+str((i+3)%4)],1500)   
    team2=team(maiay,partnery,0.5)
    for i in range(1500):
        r1=play(team1,team2)
        r2=play(team2,team1)
        if(i%100==0 and i>0):
            team1.completed(True)
            team2.completed(True)    
        print(r1,r2)
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
    load_path="battle_results/"+"officialfocused"
    team_names=['m11'+x for x in partner_names]
    proc_ar=[]
    nowar=[]
    q=multiprocessing.Queue()
    
    for i in range(num_proc):
        now=(0,1)
        p = multiprocessing.Process(target=worker,args=(partner_names[now[0]],partner_names[now[1]],weight_files[now[0]],weight_files[now[1]],i,q))
        proc_ar.append(p)
        nowar.append(now)
        p.start()
    while(True):
        if(not q.empty()):
            print("done")
            res=q.get()
            if(random.uniform(0,1)<0.001):
                print(res)
            mycol.insert(res[0])
            mycol.insert(res[1])




name_of_run="genlogs/"+"name_of_run"
os.makedirs(name_of_run,exist_ok=True)
partner_names=['name','name']
myclient = pymongo.MongoClient("HOST NAME")
mydb = myclient["DATABASE NAME"]
mycol=mydb["TRAINING SET NAME"]
weight_files=['path_to_weights',
'path_to_weights']
n=len(partner_names)
controller(12)
