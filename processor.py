from ultralytics import YOLO
import cv2
import pandas as pd
import os

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")

def process_video(input_path, output_video_path, output_csv_path):

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    data = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Skip frames for speed (process 1 out of 5)
        if frame_id % 5 != 0:
            frame_id += 1
            continue

    # 🔥 Resize frame to reduce computation
        frame = cv2.resize(frame, (640, 480))

        results = model(frame)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                data.append({
                    "frame": frame_id,
                    "vehicle_type": model.names[cls],
                    "confidence": conf,
                    "vehicle_id": f"{frame_id}_{cls}",
                    "in_queue": False,
                    "red_light_violation": False,
                    "rash_driving": False
                })

            annotated_frame = r.plot()
            out.write(annotated_frame)

        frame_id += 1

    cap.release()
    out.release()

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)