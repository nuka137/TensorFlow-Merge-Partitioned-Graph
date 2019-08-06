from tensorflow.core.framework.graph_pb2 import GraphDef

import os.path
import argparse


def merge_partitioned_graphs(partitioned_graphs):
    merged_graph = GraphDef()
    # TODO: for now we use first partitioned graph for version 
    merged_graph.versions.CopyFrom(partitioned_graphs[0].versions)
    merged_graph.library.CopyFrom(partitioned_graphs[0].library)

    send_nodes = []
    recv_nodes = []
    for pg in partitioned_graphs:
        for node in pg.node:
            if node.op == "_Send" or node.op == "_HostSend":
                send_nodes.append(node)
            elif node.op == "_Recv" or node.op == "_HostRecv":
                recv_nodes.append(node)
            else:
                merged_graph.node.extend([node])

    # build _Send/_Recv pairs
    send_recv_pairs = []
    for snode in send_nodes:
        for rnode in recv_nodes:
            if not "tensor_name" in snode.attr:
                raise RuntimeError("_Send node {} must have tensor_name"
                                   .format(snode.name))
            if not "tensor_name" in rnode.attr:
                raise RuntimeError("_Recv node {} must have tensor_name"
                                   .format(rnode.name))
            if snode.attr["tensor_name"] == rnode.attr["tensor_name"]:
                send_recv_pairs.append([snode, rnode])
                break
        else:
            raise RuntimeError(
                    "_Send node '{}' does not match any _Recv node (tensor_name={})"
                    .format(snode.name, snode.attr["tensor_name"]))

    # build source/destination node pairs
    rewrite_node_pairs = []
    for pair in send_recv_pairs:
        src_node_and_port = None
        dst_node_and_port = None
        dst_node_and_port_list = []
        for node in merged_graph.node:
            for i, input_full_name in enumerate(pair[0].input):
                str_list = input_full_name.split(":")
                if len(str_list) == 2:
                    input_node_name = str_list[0]
                    input_port = str_list[1]
                elif len(str_list) == 1:
                    input_node_name = str_list[0]
                    input_port = None
                else:
                    raise RuntimeError(
                            "Node '{}' input '{}' does not match the proper format."
                            .format(pair[0].name, input_full_name))
                if input_node_name == node.name:
                    src_node_and_port = {
                        "node": node,
                        "port": input_port,
                        "index": i
                    }
            for i, input_full_name in enumerate(node.input):
                str_list = input_full_name.split(":")
                if len(str_list) == 2:
                    input_node_name = str_list[0]
                    input_port = str_list[1]
                elif len(str_list) == 1:
                    input_node_name = str_list[0]
                    input_port = None
                else:
                    raise RuntimeError(
                            "Node '{}' input '{}' does not match the proper format."
                            .format(node.name, input_full_name))
                if input_node_name == pair[1].name:
                    dst_node_and_port = {
                        "node": node,
                        "port": input_port,
                        "index": i
                    }
                    dst_node_and_port_list.append(dst_node_and_port)

        if src_node_and_port is None:
            raise RuntimeError(
                    "_Send input node '{}' is not found. (Node name: {})"
                    .format(pair[0].input, pair[0].name))
        if not dst_node_and_port_list:
            raise RuntimeError(
                    "_Recv output is not found. (Node name: {})"
                    .format(pair[1].name))

        for dst in dst_node_and_port_list:
            rewrite_node_pairs.append({
                "src": src_node_and_port,
                "dst": dst
            })

    # rewrite destination node's input
    for pair in rewrite_node_pairs:
        src = pair["src"]
        dst = pair["dst"]
        if src["port"] is not None:
            dst["node"].input[dst["index"]] = "{}:{}".format(src["node"].name, src["port"])
        else:
            dst["node"].input[dst["index"]] = "{}".format(src["node"].name)

    return merged_graph


def merge_partitioned_graphs_from_pb(pb_files):
    graphs = []
    for pb_file in pb_files:
        graph = GraphDef()
        with open(pb_file, 'rb') as f:
            content = f.read()
        try:
            graph.ParseFromString(content)
            graphs.append(graph)
        except Exception as e:
            raise IOError("Can't parse file {}: {}.".format(pb_file, str(e)))

    return merge_partitioned_graphs(graphs)


def merge_partitioned_graphs_from_pbtxt(pbtxt_files):
    graphs = []
    for pbtxt_file in pbtxt_files:
        graph = GraphDef()
        with open(pbtxt_file, 'rb') as f:
            content = f.read()

        from google.protobuf import text_format
        try:
            text_format.Parse(
                    content.decode('UTF-8'), graph, allow_unknown_extension=True)
            graphs.append(graph)
        except Exception as e:
            raise IOError("Can't parse file {}: {}.".format(pbtxt_file, str(e)))

    return merge_partitioned_graphs(graphs)


parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+",  type=str, action="store")
parser.add_argument("--output", nargs=1,  type=str, action="store")

args = parser.parse_args()

if (args.inputs is None) or (len(args.inputs) == 0):
    raise RuntimeError("option '--inputs' must be specified")

if (args.output is None) or (len(args.inputs) == 0):
    raise RuntimeError("option '--output' must be specified")

exts = []
for input_file in args.inputs:
    root, ext = os.path.splitext(input_file)
    exts.append(ext)

merged_graph = None
if all([ext == ".pbtxt" for ext in exts]):
    merged_graph = merge_partitioned_graphs_from_pbtxt(args.inputs)
elif all([ext == ".pb" for ext in exts]):
    merged_graph = merge_partitioned_graphs_from_pb(args.inputs)
else:
    raise RuntimeError("All input files must be .pb or .pbtxt")

output_file = args.output[0]
root, ext = os.path.splitext(output_file)
if ext == ".pbtxt":
    with open(output_file, "w") as f:
        f.write(repr(merged_graph))
elif ext == ".pb":
    with open(output_file, "wb") as f:
        f.write(merged_graph.SerializeToString())
else:
    raise RuntimeError("Output file must be .pb or .pbtxt")



