import io
import os
import os.path
import typing

import numpy as np
import tensorflow as tf

from ..shared import MaiaNet_Base, Net, make_map

LC0_MAJOR = 0
LC0_MINOR = 21
LC0_PATCH = 0
WEIGHTS_MAGIC = 0x1C0


class MaiaNet(MaiaNet_Base):
    def __init__(self, target_path: str, gpu_id: typing.Optional[int] = None) -> None:
        self.file_path = target_path
        if gpu_id is None and len(tf.config.list_physical_devices("GPU")) > 0:
            gpu_id = 0
        elif gpu_id is None:
            gpu_id = -1
        self.gpu_id = gpu_id
        self.cpu = False
        self.device_name = None
        if self.gpu_id >= 0:
            gpus = tf.config.list_physical_devices("GPU")
            if not isinstance(target_path, tf.keras.Model):
                #pass
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            self.device_name = f"gpu:{self.gpu_id}"
        else:
            self.device_name = "cpu"
            self.cpu = True
        input_var = tf.keras.Input(shape=(112, 8 * 8))
        x_planes = tf.keras.layers.Reshape([112, 8, 8])(input_var)
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))

        proto_net = Net()
        proto_net.parse_proto(self.file_path)
        self.RESIDUAL_FILTERS = proto_net.filters()
        self.RESIDUAL_BLOCKS = proto_net.blocks()
        self.SE_ratio = proto_net.se_ratio()
        self._partial_models = None
        with tf.device(self.device_name):  # type: ignore
            self.model = tf.keras.Model(
                inputs=input_var,
                outputs=self.make_net(
                    x_planes, virtual_batch_size=1, cpu_mode=self.gpu_id < 0
                ),
            )
            self.replace_weights_v2(proto_net.get_weights(), cpu_mode=self.gpu_id < 0)
        super().__init__(self.file_path)

    def __str__(self) -> str:
        return f"{os.path.basename(self.file_path).replace('.pb.gz', '')}"

    def __repr__(self) -> str:
        return f"<maia_lib.MaiaNet[{self.device_name}] {os.path.join(os.path.basename(os.path.dirname(self.file_path)), os.path.basename(self.file_path).replace('.pb.gz', ''))}>"

    def __getState__(self) -> typing.Dict[str, typing.Any]:
        state = self.__dict__.copy()
        del state["model"]
        del state["l2reg"]
        state["model_state"] = self.model.get_weights()
        return state

    def __setState__(self, state: typing.Dict[str, typing.Any]) -> None:
        model_state = state.pop("model_state")
        state["l2reg"] = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        self.__dict__.update(state)
        input_var = tf.keras.Input(shape=(112, 8 * 8))
        x_planes = tf.keras.layers.Reshape([112, 8, 8])(input_var)
        self.model = tf.keras.Model(
            inputs=input_var,
            outputs=self.make_net(
                x_planes, virtual_batch_size=1, cpu_mode=self.gpu_id < 0
            ),
        )
        self.model.set_weights(model_state)

    def model_eval(self, board) -> typing.Tuple[np.ndarray, np.ndarray]:
        pol, val = self.model(board)
        return pol.cpu().numpy(), val.cpu().numpy()

    def model_layers_eval(self, board):
        if self._partial_models is None:
            self._partial_models = {}
            for l in self.model.layers:
                if l.name.endswith("cnn_2_batchnorm") or l.name.startswith(
                    "policy_cnn"
                ):
                    self._partial_models[l.name] = tf.keras.Model(
                        inputs=self.model.input,
                        outputs=self.model.get_layer(l.name).output,
                    )
        ret_dct = {}
        for k, v in self._partial_models.items():
            ret_dct[k] = v.predict(board)
        return ret_dct

    def replace_weights_v2(self, new_weights_orig, cpu_mode=False):
        new_weights = [w for w in new_weights_orig]
        # self.model.weights ordering doesn't match up nicely, so first shuffle the new weights to match up.
        # input order is (for convolutional policy):
        # policy conv
        # policy bn * 4
        # policy raw conv and bias
        # value conv
        # value bn * 4
        # value dense with bias
        # value dense with bias
        #
        # output order is (for convolutional policy):
        # value conv
        # policy conv
        # value bn * 4
        # policy bn * 4
        # policy raw conv and bias
        # value dense with bias
        # value dense with bias
        new_weights[-5] = new_weights_orig[-10]
        new_weights[-6] = new_weights_orig[-11]
        new_weights[-7] = new_weights_orig[-12]
        new_weights[-8] = new_weights_orig[-13]
        new_weights[-9] = new_weights_orig[-14]
        new_weights[-10] = new_weights_orig[-15]
        new_weights[-11] = new_weights_orig[-5]
        new_weights[-12] = new_weights_orig[-6]
        new_weights[-13] = new_weights_orig[-7]
        new_weights[-14] = new_weights_orig[-8]
        new_weights[-15] = new_weights_orig[-16]
        new_weights[-16] = new_weights_orig[-9]

        if cpu_mode:
            new_new_weights = [w for w in new_weights]
            # new_weights[-15] = new_weights_orig[-9]
            new_new_weights[-16] = new_weights[-15]
            new_new_weights[-15] = new_weights[-16]
            new_new_weights[-13] = new_weights[-9]
            new_new_weights[-12] = new_weights[-8]
            new_new_weights[-11] = new_weights[-7]
            new_new_weights[-9] = new_weights[-13]
            new_new_weights[-8] = new_weights[-12]
            new_new_weights[-7] = new_weights[-11]

            new_weights = new_new_weights
        all_evals = []
        offset = 0
        last_was_gamma = False
        for e, weights in enumerate(self.model.weights):
            source_idx = e + offset
            if weights.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if e == 0:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weights[source_idx])):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weights[source_idx][i] = new_weights[source_idx][i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Can't populate renorm weights, but the current new_weight will need using elsewhere.
                if "renorm" in weights.name:
                    offset -= 1
                    continue
                # betas without gamms need to skip the gamma in the input.
                if "beta:" in weights.name and not last_was_gamma:
                    source_idx += 1
                    offset += 1
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[source_idx], shape=weights.shape)
                if "stddev:" in weights.name:
                    weights.assign(tf.math.sqrt(new_weight + 1e-5))
                else:
                    weights.assign(new_weight)

            last_was_gamma = "gamma:" in weights.name

    def make_net(self, x_planes: tf.Tensor, virtual_batch_size=None, cpu_mode=False):

        flow = self.conv_block_v2(
            x_planes,
            filter_size=3,
            output_channels=self.RESIDUAL_FILTERS,
            bn_scale=True,
            name="input_cnn",
            virtual_batch_size=virtual_batch_size,
            cpu_mode=cpu_mode,
        )
        for i in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block_v2(
                flow,
                self.RESIDUAL_FILTERS,
                name=f"res_block_{i}",
                virtual_batch_size=virtual_batch_size,
                cpu_mode=cpu_mode,
            )
        # policy
        conv_pol = self.conv_block_v2(
            flow,
            filter_size=3,
            output_channels=self.RESIDUAL_FILTERS,
            name="policy_cnn_1",
            virtual_batch_size=virtual_batch_size,
            cpu_mode=cpu_mode,
        )
        conv_pol2 = conv_2d_cpu_shim(
            conv_pol,
            80,
            3,
            cpu_mode=cpu_mode,
            use_bias=True,
            padding="same",
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            bias_regularizer=self.l2reg,
            name="policy_cnn_2",
        )
        h_fc1 = ApplyPolicyMap()(conv_pol2)

        # value
        conv_val = self.conv_block_v2(
            flow,
            filter_size=1,
            output_channels=32,
            name="value_cnn",
            virtual_batch_size=virtual_batch_size,
            cpu_mode=cpu_mode,
        )
        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(
            128,
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            activation="relu",
            name="value_dense_1",
        )(h_conv_val_flat)
        h_fc3 = tf.keras.layers.Dense(
            3,
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            bias_regularizer=self.l2reg,
            name="value_dense_2",
        )(h_fc2)
        return h_fc1, h_fc3

    def batch_norm_v2(
        self,
        input: tf.Tensor,
        scale: bool = False,
        renorm_enabled: bool = False,
        renorm_max_r: float = 1,
        renorm_max_d: float = 0,
        renorm_momentum: float = 0.99,
        name: typing.Optional[str] = None,
        virtual_batch_size: typing.Optional[int] = None,
    ) -> tf.Tensor:
        if renorm_enabled:
            clipping = {
                "rmin": 1.0 / renorm_max_r,
                "rmax": renorm_max_r,
                "dmax": renorm_max_d,
            }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=renorm_momentum,
                name=name,
            )(input)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                virtual_batch_size=virtual_batch_size,
                name=name,
            )(input)

    def conv_block_v2(
        self,
        inputs: tf.Tensor,
        filter_size: int,
        output_channels: int,
        bn_scale: bool = False,
        name: typing.Optional[str] = None,
        virtual_batch_size: typing.Optional[int] = None,
        cpu_mode: bool = False,
    ) -> tf.Tensor:
        conv = conv_2d_cpu_shim(
            inputs,
            output_channels,
            filter_size,
            cpu_mode=cpu_mode,
            use_bias=False,
            padding="same",
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            name=name,
        )
        return tf.keras.layers.Activation("relu")(
            self.batch_norm_v2(
                conv,
                scale=bn_scale,
                name=None if name is None else f"{name}_batchnorm",
                virtual_batch_size=virtual_batch_size,
            )
        )

    def squeeze_excitation_v2(
        self,
        inputs: tf.Tensor,
        channels: int,
        name: typing.Optional[str] = None,
        cpu_mode: bool = False,
    ) -> tf.Tensor:
        pooled = tf.keras.layers.GlobalAveragePooling2D(
            data_format="channels_first",  # if not cpu_mode else "channels_last"
        )(inputs)
        squeezed = tf.keras.layers.Activation("relu")(
            tf.keras.layers.Dense(
                channels // self.SE_ratio,
                kernel_initializer="glorot_normal",
                kernel_regularizer=self.l2reg,
                name=None if name is None else f"{name}_squeeze",
            )(pooled)
        )
        excited = tf.keras.layers.Dense(
            2 * channels,
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            name=None if name is None else f"{name}_excited",
        )(squeezed)
        return ApplySqueezeExcitation(cpu_mode=cpu_mode)([inputs, excited])

    def residual_block_v2(
        self,
        inputs,
        channels,
        name=None,
        virtual_batch_size=None,
        cpu_mode=False,
    ):

        conv1 = conv_2d_cpu_shim(
            inputs,
            channels,
            3,
            cpu_mode=cpu_mode,
            use_bias=False,
            padding="same",
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            name=None if name is None else f"{name}_cnn_1",
        )
        out1 = tf.keras.layers.Activation("relu")(
            self.batch_norm_v2(
                conv1,
                scale=False,
                name=None if name is None else f"{name}_cnn_1_batchnorm",
                virtual_batch_size=virtual_batch_size,
            )
        )
        conv2 = conv_2d_cpu_shim(
            out1,
            channels,
            3,
            cpu_mode=cpu_mode,
            use_bias=False,
            padding="same",
            kernel_initializer="glorot_normal",
            kernel_regularizer=self.l2reg,
            name=None if name is None else f"{name}_cnn_2",
        )
        out2 = self.squeeze_excitation_v2(
            self.batch_norm_v2(
                conv2,
                scale=True,
                name=None if name is None else f"{name}_cnn_2_batchnorm",
                virtual_batch_size=virtual_batch_size,
            ),
            channels,
            name=name,
            cpu_mode=cpu_mode,
        )
        return tf.keras.layers.Activation("relu")(tf.keras.layers.add([inputs, out2]))


def conv_2d_cpu_shim(input_t, *args, cpu_mode=False, **kwargs):
    if cpu_mode:
        in_a = tf.keras.layers.Permute(
            dims=[2, 3, 1], name=f'cpu_transpose_{kwargs["name"]}'
        )(input_t)
        ret_t = tf.keras.layers.Conv2D(*args, data_format="channels_last", **kwargs)(
            in_a
        )
        out_a = tf.keras.layers.Permute(
            dims=[3, 1, 2], name=f'cpu_transpose_inverse_{kwargs["name"]}'
        )(ret_t)
        return out_a
    else:
        return tf.keras.layers.Conv2D(
            *args,
            data_format="channels_first",
            **kwargs,
        )(input_t)


class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, cpu_mode=False, **kwargs):
        self.cpu_mode = cpu_mode
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens) -> None:

        self.reshape_size = input_dimens[1][1]

    def call(self, inputs: typing.List[tf.Tensor]) -> tf.Tensor:
        x = inputs[0]
        excited = inputs[1]
        # if self.cpu_mode:
        #    gammas, betas = tf.split(tf.reshape(excited, [-1, 1, 1, self.reshape_size]), 2, axis=3)
        # else:
        gammas, betas = tf.split(
            tf.reshape(excited, [-1, self.reshape_size, 1, 1]), 2, axis=1
        )
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat, tf.cast(self.fc1, h_conv_pol_flat.dtype))


def is_checkpoint_dir(target_path):
    if os.path.isdir(target_path):
        return os.path.isfile(os.path.join(target_path, "checkpoint"))
    return False
