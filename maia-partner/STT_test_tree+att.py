
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
import math
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
    def __init__(self,player,partner,name=""):
        self.name=name
        if(name==""):
            self.name=player.name+partner.name
        self.player=player
        self.partner=partner
        teams_dic[self.name]=self
    def info(self):
        return {'name':self.name ,'player':self.player.info(), 'partner':self.partner.info()}
    def play(self,board,bit):
        if(bit=='1'):
            return self.partner.play(board)
        else:
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
            return self.engine.play(board,chess.engine.Limit(nodes=self.nodes), info=2)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            self.reset(True)
            print("ERRED")
            return self.play(board, info=2)
    def reset(self,res):
        self.engine.quit()
        if(res):
            self.engine=chess.engine.SimpleEngine.popen_uci(self.config)
    def completed(self,res):
        self.counter+=1
        if(self.counter%self.reset_thresh==0):
            self.reset(res)

def gbitstring(n):
    return ''.join(random.choice('01') for _ in range(n))
def cptowl(feval):
    return 0.5+math.atan(feval/290.680623072)/3.096181612
def freestuff(bitstring,infoarray,van,res):
    agrtot_en=0
    agr_en=0
    agrtot_van=0
    agr_van=0
    delt_van=0
    deltot_van=0
    delt_en=0
    deltot_en=0
    for i in range(len(infoarray)):
        if(i>=2 and bitstring[i]=='1' and bitstring[i-2]=='1' and bitstring[i-1]=='0'):
            if(i%2==van):
                delt_van+=(cptowl(infoarray[i])-cptowl(infoarray[i-2]))*(1 if van==0 else -1)
                deltot_van+=1
            else:
                delt_en+=(cptowl(infoarray[i])-cptowl(infoarray[i-2]))*(1 if van==1 else -1)
                deltot_en+=1
        if(bitstring[i]=='1' and i%2==van):
            agrtot_van+=1
            agr_van+=np.sign(infoarray[i])==res
        if(bitstring[i]=='1' and i%2!=van):
            agrtot_en+=1
            agr_en+=np.sign(infoarray[i])==res
    #print(van,res,wpldelt,wpltot)
    return {'leela_right':(agr_van+1)/(agrtot_van+1),'engine_right':(agr_en+1)/(agrtot_en+1),'loss_van':(delt_van)/(deltot_van+1),'loss_en':(delt_en)/(deltot_en+1)}

        




def play(white, black,plystring):
    curgame={}
    values=[]
    numplies=0
    board = chess.Board()
    game=chess.pgn.Game()
    t=0
    while not board.is_game_over():
        numplies+=1
        if(t%2==0):
            result = white.play(board,plystring[t])
        else:
            result = black.play(board,plystring[t])
        board.push(result.move)
        if(t==0):
            node=game.add_variation(chess.Move.from_uci(str(result.move)))
        else:
            node=node.add_variation(chess.Move.from_uci(str(result.move)))
        t+=1
        values.append(result.info['score'].white().score(mate_score=100000))
    ok=board.outcome().winner
    plystring=plystring[0:numplies]
    if(ok==None):
        winner=0
    elif(ok):
        winner=1
    else:
        winner=-1
    curgame['winner']=winner
    curgame['length']=numplies
    curgame['move_identities']=plystring
    curgame['values']=values
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter)
    pgn_string=pgn_string.replace('\n'," ")
    curgame['pgn']=pgn_string
    curgame['byproducts']=freestuff(plystring,values,white.partner.name!='s53',winner)
    #print(pgn_string)
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
    curgame['name']=curgame['white']['name']+'x'+curgame['black']['name']+"_5" #REMOVEEE
    return curgame

teams_dic={}
#assumption of defaults
def worker(partner1_name, partner2_name, partner1_weights, partner2_weights,i,q,bitstring):
    random.seed()
    plystring=bitstring
    #print(plystring)
    #print(plystring[0:10])
    #open(name_of_run+'/trash'+str(i), 'a').close()
    #sys.stderr = open(name_of_run+'/trash'+str(i), 'w')
    maiay=agent('m11',["lc0" , "--temperature=0", "--weights=models/maia-1100.pb.gz","--backend-opts=gpu="+str((i)%4) ],1)
    partnery=agent(partner1_name,["lc0" , "--weights="+partner1_weights,"--backend-opts=gpu="+str((i+1)%4)],1500)   
    team1=team(maiay,partnery)

    maiay=agent('m11',["lc0" , "--temperature=0", "--weights=models/maia-1100.pb.gz","--backend-opts=gpu="+str((i+2)%4) ],1)
    partnery=agent(partner2_name,["lc0" , "--weights="+partner2_weights,"--backend-opts=gpu="+str((i+3)%4)],1500)   
    team2=team(maiay,partnery)

    #neutral=play(team1,team1,plystring)
    #team1.completed()

    r1=play(team1,team2,plystring)
    team1.completed()
    team2.completed()
    r2=play(team2,team1,plystring)
    team1.completed(False)
    team2.completed(False)
    
    #make sure same place
    r2['set']=r1['set']
    #r3['set']=r1['set']
    q.put((r1,r2))
    return (r1,r2)
def choosenext(result_matrix,pre_totals_matrix):
    matchup_eligibility=np.zeros((n,n),dtype=int)
    for i in range(1,n):
        matchup_eligibility[0][i]=100000-pre_totals_matrix[0][i]
    selected=(0,np.argmax(matchup_eligibility[0]))
    ind=int(pre_totals_matrix[selected]/2)
    pre_totals_matrix[selected]+=2
    return (selected,ind)

def scorear():
    return [0,0,0]
def controller(num_proc):
    load_path="battle_results/"+"officialfocused"
    pos_path="ranbits_5000.pkl"
    with open(pos_path, 'rb') as f:
        bitstrings = pickle.load(f)
    team_names=['m11'+x for x in partner_names]
    result_dic=defaultdict(scorear)
    totals_matrix=np.zeros((n,n),dtype=int)
    pre_totals_matrix=np.zeros((n,n),dtype=int)
    perc1=[[] for i in range(n)]
    perc2=[[] for i in range(n)]
    resmat=[[2 for j in range(2*len(bitstrings))] for i in range(n)]
    correctness=[[] for i in range(n)]
    value_after_van=[[] for i in range(n)]
    value_after_en=[[] for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            totals_matrix[j][i]=totals_matrix[i][j]
            pre_totals_matrix[j][i]=pre_totals_matrix[i][j]
    result_matrix=np.zeros((n,n),dtype=int)
    result_matrix2=np.zeros((n,n,3,3),dtype=int)
    proc_ar=[]
    nowar=[]
    q=multiprocessing.Queue()
    deadproc=np.ones(n)
    
    for i in range(num_proc):
        now,ind=choosenext(result_matrix,pre_totals_matrix)
        p = multiprocessing.Process(target=worker,args=(partner_names[now[0]],partner_names[now[1]],weight_files[now[0]],weight_files[now[1]],i,q,bitstrings[ind]))
        proc_ar.append(p)
        nowar.append((now,ind))
        p.start()
    while(True):
        for j in range(num_proc):
            if(not proc_ar[j].is_alive()):
                res=q.get()
                if(random.uniform(0,1)<0.01):
                    print(res)
                #print(res[0]['pgn'])
                ii=nowar[j][0][0]
                jj=nowar[j][0][1]
                totals_matrix[nowar[j][0]]+=2
                totals_matrix[jj][ii]+=2
                result_matrix[ii][jj]+=res[0]['winner']
                result_matrix[ii][jj]-=res[1]['winner']
                result_matrix[jj][ii]-=res[0]['winner']
                result_matrix[jj][ii]+=res[1]['winner']
                result_dic[res[0]['name']][res[0]['winner']]+=1
                result_dic[res[1]['name']][res[1]['winner']]+=1
                perc1[jj].append(result_matrix[jj][ii]/totals_matrix[jj][ii])
                perc2[jj].append((result_matrix2[jj][ii][2][2]-result_matrix2[jj][ii][1][1]+1)/(result_matrix2[jj][ii][2][2]+result_matrix2[jj][ii][1][1]+1))
                result_matrix2[ii][jj][res[0]['winner']][-res[1]['winner']]+=1
                resmat[jj][2*nowar[j][1]+1]=res[1]['winner']
                resmat[jj][2*nowar[j][1]]=-res[0]['winner']
                correctness[ii].append(res[0]['byproducts']['leela_right'])
                correctness[ii].append(res[1]['byproducts']['leela_right'])
                correctness[jj].append(res[0]['byproducts']['engine_right'])
                correctness[jj].append(res[1]['byproducts']['engine_right'])
                value_after_van[jj].append(res[0]['byproducts']['loss_van'])
                value_after_van[jj].append(res[1]['byproducts']['loss_van'])
                value_after_en[jj].append(res[0]['byproducts']['loss_en'])
                value_after_en[jj].append(res[1]['byproducts']['loss_en'])
                meancorr=[np.mean(el) for el in correctness]
                meanvalvan=[np.mean(el) for el in value_after_van]
                meanvalen=[np.mean(el) for el in value_after_en]
                #if(res[0]['winner']==-res[1]['winner']):
                #    print(res[0]['winner'])
                #    print(res[0]['move_identities'])
                #    print(res[0]['pgn'])
                #    print(res[1]['move_identities'])
                #    print(res[1]['pgn'])
                ff=open(name_of_run+'/results.txt','w')
                ff.write("teams: "+str(team_names)+'\n')
                ff.write("expanded results: "+str(result_dic)+'\n')
                ff.write("matrix results:"+'\n'+str(result_matrix2)+'\n')
                ff.write("simple results:"+'\n'+str(result_matrix)+'\n')
                ff.write("game_totals: "+'\n'+str(totals_matrix)+'\n')
                ff.write("pre_totals:"+'\n'+str(pre_totals_matrix)+'\n')
                ff.write("metrics:"+'\n'+str(meancorr)+"valaf: "+str(meanvalvan)+" valafen: "+str(meanvalen))
                with open(name_of_run+'/result_dic.pkl',"wb") as g:
                    pickle.dump(result_dic, g)
                with open(name_of_run+'/result_matrix.pkl',"wb") as g:
                    pickle.dump(result_matrix,g)
                with open(name_of_run+'/result_matrix2.pkl',"wb") as g:
                    pickle.dump(result_matrix2,g)
                with open(name_of_run+'/totals_matrix.pkl',"wb") as g:
                    pickle.dump(totals_matrix, g)
                with open(name_of_run+'/perc1.pkl', "wb") as g:
                    pickle.dump(perc1,g)
                with open(name_of_run+'/perc2.pkl', "wb") as g:
                    pickle.dump(perc2,g)
                for w in range(n):
                    with open(name_of_run+'/'+pos_path+'_'+partner_names[w]+'.pkl', 'wb') as g:
                        pickle.dump(resmat[w],g)
                ff.close()
                now,ind=choosenext(result_matrix,pre_totals_matrix)
                p = multiprocessing.Process(target=worker,args=(partner_names[now[0]],partner_names[now[1]],weight_files[now[0]],weight_files[now[1]],j,q,bitstrings[ind]))
                proc_ar[j]=p
                nowar[j]=(now,ind)
                p.start()




name_of_run="battle_results/"+"stt_tree+att"
os.makedirs(name_of_run,exist_ok=True)
partner_names=['s53','tree','att']
#for i in range(1,2):
#    for j in range(8):
#        partner_names.append('rigdb'+str(1+i)+'g1rate4'+'v'+str(j+1))
print(partner_names)

weight_files=[
'models/128x10-t60-2-5300.pb.gz',
'models/maia-1100.pb.gz',
'models/att_t.gz'
]





n=len(partner_names)
controller(4)