import logging
import os
import warnings


def _quiet_tf():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)


try:
    _quiet_tf()
except ImportError as e:

    def MaiaNet(*args, **kwargs):
        raise e

else:
    from .net import MaiaNet
