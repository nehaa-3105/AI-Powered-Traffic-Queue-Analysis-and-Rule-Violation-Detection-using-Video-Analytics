from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import os

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

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
    stop_line_y = int(height * 0.6)
    signal_cycle = 150
    previous_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # CPU optimization
        if frame_id % 5 != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                vehicle_type = model.names[cls]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                w = x2 - x1
                h = y2 - y1

                detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        signal_red = (frame_id // signal_cycle) % 2 == 0
        color = (0, 0, 255) if signal_red else (0, 255, 0)

        cv2.line(frame, (0, stop_line_y), (640, stop_line_y), color, 3)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x_center = (l + r) / 2
            y_center = (t + b) / 2

            in_queue = y_center > 400
            red_light_violation = False
            rash_driving = False

            if signal_red and y_center < stop_line_y:
                red_light_violation = True

            if track_id in previous_positions:
                prev_y = previous_positions[track_id]
                speed = abs(prev_y - y_center)
                if speed > 40:
                    rash_driving = True

            previous_positions[track_id] = y_center

            cv2.putText(frame, f"ID {track_id}", (int(l), int(t)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            data.append({
                "frame": frame_id,
                "vehicle_id": track_id,
                "vehicle_type": vehicle_type,
                "confidence": 0.9,
                "in_queue": in_queue,
                "red_light_violation": red_light_violation,
                "rash_driving": rash_driving
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)