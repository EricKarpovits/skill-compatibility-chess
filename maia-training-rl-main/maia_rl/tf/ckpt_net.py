import os.path

import yaml
import tensorflow as tf
import numpy as np
import chess

from ..leela_board import LeelaBoard
from .net import ApplyPolicyMap
from .tfprocess import TFProcess
from ..file_handling import max_clock

LC0_MAJOR = 0
LC0_MINOR = 21
LC0_PATCH = 0
WEIGHTS_MAGIC = 0x1c0

class CKPT_Net(TFProcess):

    def __init__(self, target_path, filters = 64, blocks = 6, se_ratio = 8, board_stack_size = 8, move_time_head = True, gpu_id = 0):
        self.file_path = target_path
        self.gpu_id = gpu_id

        self.virtual_batch_size = 1
        self.config = None
        try:
            with open(os.path.join(os.path.dirname(os.path.normpath(target_path)), 'config.yaml')) as f:
                cfg = yaml.safe_load(f)
            model_dict = cfg['full_config']['model']
            self.RESIDUAL_FILTERS = model_dict['filters']
            self.RESIDUAL_BLOCKS = model_dict['residual_blocks']
            self.SE_ratio = model_dict['se_ratio']
            self.board_stack_size = model_dict.get('board_stack_size', 8)
            self.move_time_head = model_dict.get('move_time_head', False)
            self.config = model_dict
        except FileNotFoundError:
            self.RESIDUAL_FILTERS = filters
            self.RESIDUAL_BLOCKS = blocks
            self.SE_ratio = se_ratio
            self.board_stack_size = board_stack_size
            self.move_time_head = move_time_head
            self.renorm_enabled = True
            self.renorm_max_r = 1
            self.renorm_max_d = 0
            self.renorm_momentum = 0.99
            self.wdl = True
            self.moves_left = False

        if not os.path.isfile(os.path.join(target_path, 'checkpoint')):
            raise FileNotFoundError(f"{os.path.join(target_path, 'checkpoint')} not found, this isn't a checkpoint directory")

        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError:
            # This sometimes causes issues TODO: figure it out
            pass


        self.model_init()
        self.checkpoint  = tf.train.Checkpoint(model = self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.file_path, max_to_keep = 10)
        self.checkpoint.restore(self.manager.latest_checkpoint)

    #@profile
    def __call__(self, board, avail_time = 60):
        if isinstance(board, str) or isinstance(board, chess.Board):
            board = LeelaBoard(board).lcz_features(move_time = avail_time, board_stack_size = self.board_stack_size).reshape(1, (self.board_stack_size * 13) + 9, 64).astype('float32')
            #board = np.stack([board] * 64)
        elif isinstance(board, LeelaBoard):
            board = board.lcz_features(move_time = avail_time, board_stack_size = self.board_stack_size).reshape(1, (self.board_stack_size * 13) + 9, 64).astype('float32')
            #board = np.stack([board] * 64)
        elif isinstance(board, np.ndarray):
            board = board.reshape(-1, (self.board_stack_size * 13) + 9, 64).astype('float32')
        if isinstance(avail_time, int) or isinstance(avail_time, float):
            avail_time = np.array([avail_time] * board.shape[0]).reshape(-1, 1).astype('float32') / max_clock
        return self.model(board)

    #@profile
    def eval_boards(self, boards, policy_softmax_temp = 1, avail_time = 60):
        boards = [self.board_convert(b) for b in boards]
        boards_arr = np.stack([b.lcz_features(move_time = avail_time,board_stack_size = self.board_stack_size) for b in boards])

        pols, vals, mvts = self(boards_arr)
        ret_vals = []
        for policy, val, mvt, board in zip(pols.cpu().numpy(), vals.cpu().numpy(), mvts.cpu().numpy(), boards):
            ret_vals.append(self.make_outputs(policy, val, mvt, board, policy_softmax_temp = policy_softmax_temp))
        return ret_vals

    #@profile
    def eval_board(self, board, policy_softmax_temp = 1, avail_time = 60):
        board = self.board_convert(board)
        pol, val, mvt = self(board, avail_time = avail_time)
        policy = pol.cpu().numpy()[0]
        return self.make_outputs(
                    pol.cpu().numpy()[0],
                    val.cpu().numpy()[0],
                    mvt.cpu().numpy()[0],
                    board,
                    policy_softmax_temp = policy_softmax_temp
                    )

    #@profile
    def board_convert(self, board, avail_time = 60):
        if isinstance(board, str) or isinstance(board, chess.Board):
            return LeelaBoard(board)
        return board

    #@profile
    def make_outputs(self, policy, val, move_times, board, policy_softmax_temp = 1):
        return convert_outputs(policy, val, move_times, board, policy_softmax_temp = policy_softmax_temp)

    def make_net(self, x_planes):
        flow = self.conv_block_v2(x_planes, filter_size=3, output_channels=self.RESIDUAL_FILTERS, bn_scale=True, name = 'input_cnn')

        for i in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block_v2(flow, self.RESIDUAL_FILTERS, name = f'res_block_{i}')

        #policy
        conv_pol = self.conv_block_v2(flow, filter_size=3, output_channels=self.RESIDUAL_FILTERS, name = 'policy_cnn_1')
        conv_pol2 = tf.keras.layers.Conv2D(80, 3, use_bias=True, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, data_format='channels_first', name = 'policy_cnn_2')(conv_pol)
        h_fc1 = ApplyPolicyMap()(conv_pol2)

        #value
        conv_val = self.conv_block_v2(flow, filter_size=1, output_channels=32, name = 'value_cnn')
        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(128, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, activation='relu', name = 'value_dense_1')(h_conv_val_flat)
        h_fc3 = tf.keras.layers.Dense(3, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, name = 'value_dense_2')(h_fc2)
        return h_fc1, h_fc3

def _softmax(x, softmax_temp):
    e_x = np.exp((x - np.max(x))/softmax_temp)
    return e_x / e_x.sum(axis=0)

def weight_to_np(weight):
    params = np.frombuffer(weight.params, dtype = np.uint16).astype(np.float32)
    params /= 0xffff
    if weight.max_val == weight.min_val:
        params = (params + weight.min_val)
    else:
        params = (weight.max_val - weight.min_val) * params + weight.min_val
    return params

def name_to_proto_name(name):
    if 'input_cnn' in name:
        return ['input', mapped_tf(name, cnn_mapping)]
    elif 'res_block_' in name:
        return ['residual', int(name.split('_')[2])] + res_tf(name)
    elif 'value_cnn' in name:
        return ['value', mapped_tf(name, cnn_mapping)]
    elif 'value_dense' in name:
        return [mapped_tf(name, value_mapping)]
    elif 'policy_cnn_1' in name:
        return ['policy1', mapped_tf(name, cnn_mapping)]
    elif 'policy_cnn_2' in name:
        return ['policy', mapped_tf(name, cnn_mapping)]
    raise KeyError(f"{name} not found")


def res_tf(name):
    if 'cnn_1' in name:

        return ['conv1', mapped_tf(name, cnn_mapping)]
    elif 'cnn_2' in name:
        return ['conv2', mapped_tf(name, cnn_mapping)]
    else:
        return ['se', mapped_tf(name, se_mapping)]

value_mapping = {
        'value_dense_1/kernel' : 'ip1_val_w',
        'value_dense_1/bias' : 'ip1_val_b',
        'value_dense_2/kernel' : 'ip2_val_w',
        'value_dense_2/bias' : 'ip2_val_b',
}

cnn_mapping = {
    'kernel' : 'weights',
    'gamma' : 'bn_gammas',
    'beta' : 'bn_betas',
    'moving_mean' : 'bn_means',
    'moving_variance' : 'bn_stddivs',
    'bias' : 'biases',
}

se_mapping = {
    'squeeze/kernel' : 'w1',
    'squeeze/bias' : 'b1',
    'excited/kernel' : 'w2',
    'excited/bias' : 'b2',
}

def mapped_tf(name, map_dict):
    for n, v in map_dict.items():
        if n in name:
            return v
    raise KeyError(f"{name} not in bn_mapping")

def proto_tf_weights(proto_weights, tf_model):
    new_weights = []
    for w in tf_model.weights:
        pn = name_to_proto_name(w.name)
        w_p = getattr(proto_weights, pn.pop(0))
        while len(pn) > 0:
            pvn = pn.pop(0)
            if isinstance(pvn , str):
                w_p = getattr(w_p, pvn)
            else:
                w_p = w_p[pvn]
        a = weight_to_np(w_p)
        new_weights.append(a.reshape(w.shape))
    return new_weights

def convert_outputs(policy, val, move_times, board, policy_softmax_temp = 1):
    legal_uci = [m.uci() for m in board.generate_legal_moves()]
    legal_indexes = board.lcz_uci_to_idx(legal_uci)
    softmaxed = _softmax(policy[legal_indexes], policy_softmax_temp)
    return { m : float(v) for v, m in sorted(zip(softmaxed, legal_uci), reverse = True)}, float(val[0]) /2 + 0.5, _softmax(move_times, policy_softmax_temp)
