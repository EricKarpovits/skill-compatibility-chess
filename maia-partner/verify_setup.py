# Test script to verify setup
import tensorflow as tf
import chess
import numpy as np
import os

def check_setup():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Chess library version: {chess.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if required directories exist
    dirs_to_check = [
        "battle_results/hb_tree+att",
        "models"
    ]
    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        print(f"Directory '{dir_path}' exists: {exists}")

    # Check required model files
    model_files = [
        "models/maia-1100.pb.gz",
        "models/128x10-t60-2-5300.pb.gz",
        "models/att_h.gz",
        "models/att_t.gz"
    ]
    print("\nChecking model files:")
    for model_file in model_files:
        exists = os.path.exists(model_file)
        print(f"Model '{model_file}' exists: {exists}")

if __name__ == "__main__":
    check_setup()