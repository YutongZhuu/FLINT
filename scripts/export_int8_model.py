from ultralytics import YOLO

model = YOLO("models/yolov5n-ultralytics-standard.pt")
model.export(
    format="onnx",
    int8=True,
    data="coco128.yaml",
    fraction=0.25,
)