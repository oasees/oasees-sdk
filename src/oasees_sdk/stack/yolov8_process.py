import argparse
import os


def process_video(input_path, output_path, model_name="yolov8n.pt", conf=0.3):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from exc

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required. Install with `pip install opencv-python`.") from exc

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 video processing for COGNETS demo")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model (e.g., yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    frames = process_video(args.input, args.output, args.model, args.conf)
    print(f"Processed {frames} frames -> {args.output}")


if __name__ == "__main__":
    main()
