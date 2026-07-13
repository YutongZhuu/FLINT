#!/usr/bin/env python3
"""Convert all float16 tensors and type annotations in an ONNX model to float32."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper


def convert_type(type_proto: onnx.TypeProto) -> None:
    if type_proto.HasField("tensor_type"):
        if type_proto.tensor_type.elem_type == TensorProto.FLOAT16:
            type_proto.tensor_type.elem_type = TensorProto.FLOAT
    elif type_proto.HasField("sequence_type"):
        convert_type(type_proto.sequence_type.elem_type)
    elif type_proto.HasField("optional_type"):
        convert_type(type_proto.optional_type.elem_type)
    elif type_proto.HasField("map_type"):
        convert_type(type_proto.map_type.value_type)


def convert_tensor(tensor: TensorProto) -> None:
    if tensor.data_type != TensorProto.FLOAT16:
        return
    replacement = numpy_helper.from_array(
        numpy_helper.to_array(tensor).astype(np.float32), name=tensor.name
    )
    tensor.CopyFrom(replacement)


def convert_graph(graph: onnx.GraphProto) -> None:
    for value in (*graph.input, *graph.output, *graph.value_info):
        convert_type(value.type)
    for tensor in graph.initializer:
        convert_tensor(tensor)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                convert_tensor(attr.t)
            elif attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    convert_tensor(tensor)
            elif attr.type == AttributeProto.GRAPH:
                convert_graph(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for nested_graph in attr.graphs:
                    convert_graph(nested_graph)
            elif node.op_type == "Cast" and attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    convert_graph(model.graph)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
