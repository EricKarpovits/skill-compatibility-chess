import os
import os.path
import yaml
import glob
import shutil

import numpy as np

from .file_handling.proto_reader import sampling_types
from .file_handling.data_file_reader import ZipDataFileReaderSingle, ZipDataFileReaderMulti
from .file_handling.data_file_reader import move_time_splits
from .utils import get_commit_info

def get_data_loaders(data_files, batch_size, num_workers, shuffle_size, sampling_type, sampling_options, single_thread = False, board_stack_size = 8, backend = 'tensorflow', random_seed = 123):
    loaders = {}
    if sampling_type not in sampling_types:
        raise NotImplementedError(f"{sampling_type} is not a known sampling function, {sampling_types.keys().join(', ')} are the valid options")
    for set_name in ['train', 'validate']:
        if single_thread:
            loader_class = ZipDataFileReaderSingle
        else:
            loader_class = ZipDataFileReaderMulti
        loader = loader_class(
                os.path.join(data_files, f"{set_name}.zip"),
                batch_size,
                num_workers = num_workers,
                shuffle_size = shuffle_size,
                sampling_type = sampling_type,
                sampling_options = sampling_options,
                board_stack_size = board_stack_size,
                rng = np.random.default_rng(random_seed),
                verbose=False,
                )
        #import pdb; pdb.set_trace()
        if backend == 'tensorflow':
            import tensorflow as tf
            dataset = tf.data.Dataset.from_generator(
                    loader.sampling_iter,
                    output_types=(
                        tf.float32,tf.float32,tf.float32,tf.float32
                        ),
                    output_shapes=(
                        tf.TensorShape([batch_size,(board_stack_size * 13) + 8,8,8]), #board 
                        #tf.TensorShape([batch_size, 1]), #time avail
                        tf.TensorShape([batch_size, 1858]), #moves
                        tf.TensorShape([batch_size, 3]), #result
                        #tf.TensorShape([batch_size, len(move_time_splits) + 1]), #move time
                        tf.TensorShape([batch_size, 3]) #Q
                    )
                    ).prefetch(4)
        else:
            raise NotImplementedError(f"Backend {backend} is not known, only tensorflow and pytorch are defined")
        loaders[set_name] = (loader, dataset)
    for v in loaders.values():
        v[0].check_workers()

    return loaders['train'][0], loaders['train'][1], loaders['validate'][0], loaders['validate'][1]

def make_model_files(cfg, model_name, data_name, save_dir, models_dir,args=None):
    output_name = os.path.join(save_dir, data_name)
    os.makedirs(output_name, exist_ok = True)
    ckpt_dir = os.path.join(output_name, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok = True)
    model_saves_dir = os.path.join(models_dir, data_name)
    models = [(int(p.name.split('-')[1]), p.name, p.path) for p in os.scandir(model_saves_dir) if p.name.endswith('.pb.gz')]
    model_file_name = None
    if len(models) > 0:
        top_model = max(models, key = lambda x : x[0])
        model_file_name = top_model[1].replace('ckpt', data_name)
        shutil.copyfile(top_model[2], os.path.join(output_name, model_file_name))

    last_chkpt = "ckpt----"
    with open(os.path.join(model_saves_dir, 'checkpoint')) as f:
        for line in f:
            if line.startswith("model_checkpoint_path:"):
                last_chkpt =line.split('"')[1]
                break

    shutil.copyfile(os.path.join(model_saves_dir, 'checkpoint'), os.path.join(ckpt_dir, 'checkpoint'))

    for ckpt_path in glob.glob(os.path.join(model_saves_dir, f"{last_chkpt}*")):
        shutil.copyfile(ckpt_path, os.path.join(ckpt_dir, os.path.basename(ckpt_path)))

    with open(os.path.join(output_name, "config.yaml"), 'w') as f:
        out_dict = {
            'name' : data_name,
            'display_name' : f"{model_name}-{data_name}",
            'engine' : 'lc0_23',
            'options': {'weightsPath': model_file_name},
            'full_config' : cfg,
            'run_parameters': args,
            'git_info' : get_commit_info(),
        }
        yaml.dump(out_dict, f)
