from flask import Flask, request, jsonify
from keras.models import model_from_json
import numpy as np
import cv2
import os
import tempfile
from function import (
    mediapipe_detection,
    extract_keypoints,
    actions,
    mp_hands,
    draw_styled_landmarks,
)
import traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # Set limit to 100MB

# Load model
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("prototype.h5")


@app.route("/")
def home():
    return "Welcome to ASL-Model"


@app.route("/predict", methods=["POST"])
def predict():
    temp_video = None
    try:
        # Save the uploaded video to a temporary file
        video = request.files["video"]
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        video.save(temp_video.name)
        temp_video.close()  # Ensure the file is closed after saving

        predictions = []
        interval_predictions = []
        interval_duration = 3  # 3 seconds interval
        fps = 30  # Assuming video is 30 frames per second
        frames_per_interval = fps * interval_duration

        cap = cv2.VideoCapture(temp_video.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_intervals = frame_count // frames_per_interval

        # Set mediapipe model
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            for interval in range(total_intervals):
                interval_predictions = []
                sequence = []  # Reset sequence for each interval
                for _ in range(frames_per_interval):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Crop frame to top left corner with a taller rectangle
                    height, width, _ = frame.shape
                    crop_height = int(height * 0.75)  # 75% of the height
                    crop_width = int(width * 0.5)  # 50% of the width
                    crop_x_start = 0
                    crop_y_start = 0
                    cropframe = frame[
                        crop_y_start : crop_y_start + crop_height,
                        crop_x_start : crop_x_start + crop_width,
                    ]

                    # Process frame
                    image, results = mediapipe_detection(cropframe, hands)
                    keypoints = extract_keypoints(results)

                    if keypoints is not None and len(keypoints) > 0:
                        sequence.append(keypoints)
                        sequence = sequence[
                            -29:
                        ]  # Adjust to match the expected input shape

                        if (
                            len(sequence) == 29
                        ):  # Adjust to match the expected input shape
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            interval_predictions.append(np.argmax(res))

                if interval_predictions:
                    most_frequent_prediction = max(
                        set(interval_predictions), key=interval_predictions.count
                    )
                    predictions.append(actions[most_frequent_prediction])
                else:
                    predictions.append("No action detected")

        cap.release()

        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)

        return jsonify({"actions": predictions})

    except Exception as e:
        print("Error occurred: ", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_video and os.path.exists(temp_video.name):
            try:
                os.remove(temp_video.name)
            except Exception as e:
                print("Error removing temp file: ", str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
