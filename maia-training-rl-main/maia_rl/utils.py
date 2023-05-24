import datetime
import sys
import glob
import os.path

import yaml
import pytz
import pygit2

tz = pytz.timezone('######')

colours = {
    'blue' : '\033[94m',
    'green' : '\033[92m',
    'yellow' : '\033[93m',
    'red' : '\033[91m',
    'pink' : '\033[95m',
}
endColour = '\033[0m'

def current_datetime():
    return datetime.datetime.now(tz)

def printWithDate(s, colour = None, **kwargs):
    if colour is None:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')} {s}", **kwargs)
    else:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}{colours[colour]} {s}{endColour}", **kwargs)

def get_config_blocks(conf_path):
    with open(conf_path) as f:
        cfg = yaml.safe_load(f.read())
        return cfg['model']['residual_blocks']

def get_config_filters(conf_path):
    with open(conf_path) as f:
        cfg = yaml.safe_load(f.read())
        return cfg['model']['filters']

def get_config_board_stack_size(conf_path):
    with open(conf_path) as f:
        cfg = yaml.safe_load(f.read())
        return cfg['model']['board_stack_size']

def get_commit_info(repo_path = '.'):
    repository_git_path = pygit2.discover_repository(repo_path)
    repo = pygit2.Repository(repository_git_path)
    last = repo[repo.head.target]
    return {
        'date' : datetime.datetime.utcfromtimestamp(last.commit_time).strftime('%Y-%m-%d %H:%M:%S'),
        'hex' : last.hex,
        'message' : last.message.strip(),
        'author' : last.author.name,
        'author_email' : last.author.email,
        'link' : f"x/{last.hex}"
    }

def get_model_name(path):
    if path[-1] == '/':
        path = path[:-1]
    targets = glob.glob(path+"-*")
    if len(targets) > 0:
        return os.path.basename(max(targets, key = lambda x : int(x.split('-')[-1])))
    else:
        return os.path.basename(path)

class Tee(object):
    #Based on https://stackoverflow.com/a/616686
    def __init__(self, fname, is_err = False):
        self.file = open(fname, 'a')
        self.is_err = is_err
        if is_err:
            self.stdstream = sys.stderr
            sys.stderr = self
        else:
            self.stdstream = sys.stdout
            sys.stdout = self
    def __del__(self):
        if self.is_err:
            sys.stderr = self.stdstream
        else:
            sys.stdout = self.stdstream
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdstream.write(data)
    def flush(self):
        #self.file.flush()
        self.stdstream.flush()
