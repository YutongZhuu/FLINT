#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "w")
b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), "b")
node = helper.make_node("Conv", ["x", "w", "b"], ["y"],
                        dilations=[1, 1], pads=[0, 0, 0, 0],
                        strides=[1, 1], group=1)
graph = helper.make_graph([node], "conv_test", [x], [y], [w, b])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.checker.check_model(model)
onnx.save(model, "build/conv-test.onnx")
