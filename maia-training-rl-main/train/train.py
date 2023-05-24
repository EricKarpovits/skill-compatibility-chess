import os
import os.path
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.random.set_seed(12345)

import yaml
import wandb

import maia_rl

def main():
    parser = argparse.ArgumentParser(description='Training script for Maias', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', help='config file for model parameters')
    parser.add_argument('data_name', help='Name of sub-dataset, e.g. player or rating')
    parser.add_argument('model_name',help='Name of model')
    parser.add_argument('--starting_model_path', help='path to loaded checkpoint', default=None)
    parser.add_argument('--dataset_path', help='Base path to files', default = 'data', type = str)
    parser.add_argument('--gpu', help='gpu to use', default = 0, type = int)
    parser.add_argument('--num_workers', help='number of worker threads to use', default = 6, type = int)
    parser.add_argument('--copy_dir', help='dir to save final models in', default = 'final_models')
    parser.add_argument('--log_dir', help='log files dir', default = 'logs')
    parser.add_argument('--tb_logs_dir', help = 'directory to store tensorboard run info', default = 'runs')
    parser.add_argument('--models_dir', help = 'directory to store checkpoint model files', default = 'models')
    parser.add_argument('--single_thread', help = 'Disable multithreaded loading', action = 'store_true')

    args = parser.parse_args()

    #model_name = os.path.basename(args.config).split('.')[0]
    data_name = args.data_name
    model_name=args.model_name.split('/')[0]+args.model_name.split('/')[1]+args.model_name.split('/')[2]
    print(model_name)
    data_path = os.path.join(args.dataset_path, data_name)
    log_dirname = os.path.join(args.log_dir, args.model_name)
    print(log_dirname)
    os.makedirs(log_dirname, exist_ok=True)

    std_log = os.path.join(log_dirname, f'{model_name}_stdout.log')
    err_log = os.path.join(log_dirname, f'{model_name}_stderr.log')
    maia_rl.printWithDate(f"Starting log files: '{std_log}', '{err_log}'", flush = True)

    tee_out = maia_rl.Tee(std_log, is_err = False)
    tee_err = maia_rl.Tee(err_log, is_err = True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    random_seed = 1234
    if 'random_seed' in cfg['training']:
        random_seed = cfg['training']['random_seed']
    tf.random.set_seed(random_seed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    maia_rl.printWithDate(yaml.dump(cfg, default_flow_style=False), flush = True)

    maia_rl.printWithDate(f"Starting training: {model_name}-{data_name}")
    maia_rl.printWithDate(f"Dataset: {data_path}")

    train_handler, train_dataset, val_handler, val_dataset = maia_rl.get_data_loaders(
            data_path,
            cfg['training']['batch_size'],
            args.num_workers,
            cfg['training']['shuffle_size'],
            cfg['training']['sampling'],
            cfg['training'].get('sampling_options', None),
            single_thread = args.single_thread,
            board_stack_size= cfg['model'].get('board_stack_size', 8),
            random_seed = random_seed,
            )

    maia_rl.printWithDate(f"Train loader: {train_handler.sampling_type} config {train_handler.sampling_options}")

    current_index = 1
    original_name = data_name
    while os.path.isdir(os.path.join(args.tb_logs_dir, data_name)) or os.path.isdir(os.path.join(args.models_dir, data_name)):
        current_index += 1
        data_name = f"{original_name}-{current_index}"

    tf_proc = maia_rl.TFProcess(
            cfg,
            model_name,
            data_name,
            train_handler,
            val_handler,
            tb_logs_dir = args.tb_logs_dir,
            models_dir = args.models_dir,
            )
    tf_proc.init_v2(train_dataset, val_dataset, starting_model = args.starting_model_path)
    tf_proc.restore_v2()

    maia_rl.printWithDate("Using {} evaluation batches".format(cfg['training']['num_test_evals']), flush = True)

    try:
        tf_proc.process_loop_v2(
            cfg['training']['batch_size'],
            cfg['training']['num_test_evals'],
            batch_splits=cfg['training']['num_batch_splits'],
            )
        maia_rl.make_model_files(
            cfg,
            model_name,
            data_name,
            args.copy_dir,
            models_dir = args.models_dir,
            args=args
            )
    except KeyboardInterrupt:
        maia_rl.printWithDate("\rKeyboardInterrupt: Stopping")
        return
    finally:
        maia_rl.printWithDate("Starting shutdown")
        train_handler.shutdown()
        val_handler.shutdown()
        tee_out.flush()
        tee_err.flush()
        maia_rl.printWithDate("Shutdown complete", flush = True)
    

if __name__ == "__main__":
    main()
