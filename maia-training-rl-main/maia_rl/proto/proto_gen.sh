#!/usr/bin/env bash

protoc --proto_path=. --python_out=. net.proto
protoc --proto_path=. --python_out=. chunk.proto
protoc --proto_path=. --python_out=. move_tree.proto
protoc --proto_path=. --python_out=. data_files.proto
protoc --proto_path=. --python_out=. data_files_ignorable.proto

echo "You also need to edit chunk_pb2.py add:"
echo "
from .net_pb2 import DESCRIPTOR as net_pb2_DESCRIPTOR, _ENGINEVERSION as net_pb2_ENGINEVERSION

class names():
    DESCRIPTOR = net_pb2_DESCRIPTOR
    _ENGINEVERSION = net_pb2_ENGINEVERSION

net__pb2 = names()
"
echo "This is the least bad option and but I still don't trust it enough to make it fully automated"
