import cv2
from pathlib import Path
from ultralytics import YOLO
from boxmot import StrongSort
import numpy as np
import yaml
import time

model = YOLO("best.pt")
model.fuse()

with open("/boxmot/configs/strongsort.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


tracker = StrongSort(
    reid_weights=Path("boxmot/osnet_x0_25_imagenet.pth"),
    device="cpu",  
    half=True,
    per_class=False,
    min_conf=0.6,
    max_cos_dist=0.2,
    max_iou_dist=0.7,
    max_age=100,
    n_init=3,
    nn_budget=100,
    mc_lambda=0.98,
    ema_alpha=0.9,
)

cap = cv2.VideoCapture("15sec_input_720p.mp4")

last_outputs = []
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_inference.mp4', fourcc, fps, (width, height))


while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (640, 384))  
    results = model(resized, verbose=False)[0]
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 384

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clses = results.boxes.cls.cpu().numpy().astype(int)

    mask = clses == 2
    boxes = boxes[mask]
    confs = confs[mask]
    clses = clses[mask]

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    boxes = boxes.astype(int)

    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_mask]
    confs = confs[valid_mask]
    clses = clses[valid_mask]

    detections = np.concatenate([boxes, confs[:, None], clses[:, None]], axis=1)  # shape (N, 6)
    dets = np.array(detections)

    if dets.shape[0] > 0:
        outputs = tracker.update(dets, frame)
    else:
        print('empty detections, skipping update')
        outputs = []

    for output in outputs:
        x1, y1, x2, y2, track_id, conf, cls_id, _ = output
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # cv2.imshow("YOLOv11 + StrongSORT (Fast)", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
