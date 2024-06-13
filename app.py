from flask import Flask, request, jsonify, send_file
from keras.models import model_from_json
import numpy as np
import cv2
import os
import tempfile
from function import (
    mediapipe_detection,
    extract_keypoints,
    asl_actions,
    psl_actions,
    mp_hands,
    draw_styled_landmarks,
)
import traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # Set limit to 100MB

# Load ASL model
with open("asl-model.json", "r") as json_file:
    asl_model_json = json_file.read()
asl_model = model_from_json(asl_model_json)
asl_model.load_weights("asl-model.h5")

# Load PSL model
with open("psl-model.json", "r") as json_file:
    psl_model_json = json_file.read()
psl_model = model_from_json(psl_model_json)
psl_model.load_weights("psl-model.h5")


@app.route("/")
def home():
    return "Welcome to the Models"


def predict_action(model, video, actions):
    temp_video = None
    try:
        # Save the uploaded video to a temporary file
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

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

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

                    # Define a wider and taller bounding box in the top right corner
                    box_width = int(width * 0.5)  # 60% of the width
                    box_height = int(height * 0.5)  # 80% of the height
                    crop_x_start = 0
                    crop_y_start = 0

                    cropframe = frame[
                        crop_y_start : crop_y_start + box_height,
                        crop_x_start : crop_x_start + box_width,
                    ]

                    # Draw the bounding box on the original frame
                    cv2.rectangle(
                        frame,
                        (crop_x_start, crop_y_start),
                        (crop_x_start + box_width, crop_y_start + box_height),
                        (0, 255, 0),
                        2,
                    )

                    # Process frame
                    image, results = mediapipe_detection(cropframe, hands)
                    draw_styled_landmarks(
                        image, results
                    )  # Draw landmarks on the cropped frame
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

                    # Display the frame with the bounding box and landmarks
                    cv2.imshow("Processed Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if interval_predictions:
                    most_frequent_prediction = max(
                        set(interval_predictions), key=interval_predictions.count
                    )
                    predictions.append(actions[most_frequent_prediction])
                else:
                    predictions.append("No action detected")

        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)

        return {"actions": predictions}

    except Exception as e:
        print("Error occurred: ", str(e))
        print(traceback.format_exc())
        return {"error": str(e)}, 500

    finally:
        if temp_video and os.path.exists(temp_video.name):
            try:
                os.remove(temp_video.name)
            except Exception as e:
                print("Error removing temp file: ", str(e))


@app.route("/predict_asl", methods=["POST"])
def predict_asl():
    video = request.files["video"]
    result = predict_action(asl_model, video, asl_actions)
    return jsonify(result)


@app.route("/predict_psl", methods=["POST"])
def predict_psl():
    video = request.files["video"]
    result = predict_action(psl_model, video, psl_actions)
    if "actions" in result:
        urdu_predictions = [
            psl_actions[psl_actions.tolist().index(action)]
            for action in result["actions"]
        ]
        result["actions"] = urdu_predictions
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
