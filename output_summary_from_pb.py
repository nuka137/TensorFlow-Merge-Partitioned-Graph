import os.path
import argparse

import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef


def load_graphdef_from_pb(pb_file):
    graph = GraphDef()
    with open(pb_file, 'rb') as f:
        content = f.read()
        try:
            graph.ParseFromString(content)
        except Exception as e:
            raise IOError("Can't parse file {}: {}".format(pb_file, str(e)))
    return graph


def load_graphdef_from_pbtxt(pbtxt_file):
    graph = GraphDef()
    with open(pbtxt_file, 'rb') as f:
        content = f.read()
        from google.protobuf import text_format
        try:
            text_format.Parse(content.decode('UTF-8'), graph,
                              allow_unknown_extension=True)
        except Exception as e:
            raise IOError("Can't parse file {}: {}".format(pbtxt_file, str(e)))
    return graph


parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs=1, type=str, action="store")
parser.add_argument("--output_dir", nargs=1, type=str, action="store")

args = parser.parse_args()

if (args.input is None) or (len(args.input) == 0):
    raise RuntimeError("option '--input' must be specified")

if (args.output_dir is None) or (len(args.output_dir) == 0):
    raise RuntimeError("option '--output_dir' must be specified")

_, ext = os.path.splitext(args.input[0])


graph = None
if ext == ".pbtxt":
    graph = load_graphdef_from_pbtxt(args.input[0])
elif ext == ".pb":
    graph = load_graphdef_from_pb(args.input[0])
else:
    raise RuntimeError("Input file must be .pb or .pbtxt")


tf.summary.FileWriter(args.output_dir[0], graph)

