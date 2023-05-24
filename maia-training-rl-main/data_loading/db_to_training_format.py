import argparse
import tempfile
import os
os.environ['MAIA_DISABLE_TF'] = 'true'
import maia_rl

import os.path
import shutil
import multiprocessing

import pymongo
from shutil import copyfile
def main():
    parser = argparse.ArgumentParser(description='Take a players PGN and create training data formatted file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input table')
    parser.add_argument('output', help='output data directory name')
    parser.add_argument('gen_number', help='generation subdirectory')
    parser.add_argument('game_name')
    parser.add_argument('--cut', default=10000)
    parser.add_argument('--games_per_file', type=int, help='max games per training data file', default=128)
    parser.add_argument('--pool_size', type=int, help='pool size', default=4)
    parser.add_argument('--onlyme',default=0)
    parser.add_argument('--vals',default=0)
    
    args = parser.parse_args()
    
    
    
    maia_rl.printWithDate(f"Reading games from: {args.input}")
    i = 0
    fullpath='path_to_data'+args.output+'/'+args.gen_number #CHANGE THIS TO WHATEVER YOU NEED TO
    os.makedirs(fullpath, exist_ok=True)
    copyfile('db_to_training_format.py', fullpath+'/db2train.py')
    f=open(fullpath+'/configinfo','w')
    f.write(args.input + " "+args.output +" "+ args.gen_number + " " +args.game_name+" "+ str(args.cut))
    
    def create_file(name,settype,cut,onlyme, vals):
        games = mongo_iter(args.input, args.game_name, settype,cut,onlyme,vals)
        testpath=fullpath+'/'+name
        print(testpath)
        with tempfile.TemporaryDirectory(prefix=os.path.basename(testpath).replace('.zip', '') + '_', dir=os.path.dirname(os.path.abspath(testpath))) as tmpdir, multiprocessing.Pool(args.pool_size) as pool:
            batch = []
            processes = {}
            for i, (game_pgn, result,bitstring,valar) in enumerate(games):
                #print("wow",bitstring)
                batch.append((game_pgn, result,bitstring,valar))
                if len(batch) >= args.games_per_file:
                    processes[i] = pool.apply_async(maia_rl.write_games_batch_ignore, args = (os.path.join(tmpdir, f'games_data_{i}.pb'), batch))
                    #write_batch(os.path.join(tmpdir, f'games_data_{i}.pb'), batch)
                    if len(processes) > args.pool_size + 100:
                        while len(processes) > args.pool_size + 20:
                            j = 0
                            for j in list(processes.keys()):
                                if processes[j].ready():
                                    processes[j].get()
                                    del processes[j]
                        maia_rl.printWithDate(f"{os.path.basename(args.input)} done runner {j} of {i} ({len(processes)}+)", end = '\r')
                    batch = []
            if len(batch) > 0:
                processes[i] = pool.apply_async(maia_rl.write_games_batch_ignore, args = (os.path.join(tmpdir, f'games_data_{i}.pb'), batch))
            for j in sorted(processes.keys()):
                processes[j].get()
                maia_rl.printWithDate(f"{os.path.basename(args.input)} done runner {j} of {i} ({len(processes)}-complete)", end = '\r')
            maia_rl.printWithDate(f"\n{os.path.basename(args.input)} loading to zip")
            shutil.make_archive(
                testpath.replace('.zip', ''),
                'zip',
                tmpdir,
            )
            maia_rl.printWithDate(f"{os.path.basename(args.input)} Done")
    create_file('/train.zip','tr',int(0.8*int(args.cut)),bool(int(args.onlyme)),bool(int(args.vals)))
    create_file('/test.zip','te',int(0.1*int(args.cut)),bool(int(args.onlyme)),bool(int(args.vals)))
    create_file('/validate.zip','va',int(0.1*int(args.cut)),bool(int(args.onlyme)),bool(int(args.vals)))

def mongo_iter(table_name,gamename,settype,cut,onlyme,vals):
    print(table_name,gamename,settype,cut)
    tbl = pymongo.MongoClient().NAME_OF_DATABASE[table_name] #CHANGE THIS TO WHATEVER YOU NEED TO
    c = tbl.aggregate([{ '$match': {'$and':[ { 'set': settype },{ 'name': {'$in':[gamename]} }]}},{'$project':{'_id' : 0, 'pgn':1 ,'winner' : 1,'move_identities':1, 'board_values':1}}])
    t=0
    for e in c:
        if(t==cut):
            break
        if(onlyme and vals):
            yield e['pgn'], e['winner'],e['move_identities'], e['board_values']
        elif(onlyme):
            yield e['pgn'], e['winner'],e['move_identities'],[]
        elif(vals):
            yield e['pgn'], e['winner'],"",e['board_values']
        else:
            yield e['pgn'], e['winner'],"",[]
        t+=1
if __name__ == "__main__":

    main()
