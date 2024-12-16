# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Check if webcam is working
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Updated label dictionary with your actual signs
# labels_dict = {0: 'hello', 1: 'I love you', 2: 'Fuck you'}

# # Main loop
# while True:
#     # Capture frame from the webcam
#     ret, frame = cap.read()

#     # If frame is not read correctly
#     if not ret:
#         print("Error: Failed to capture image")
#         break

#     # Get the shape of the frame
#     H, W, _ = frame.shape

#     # Convert frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame to detect hands
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         bounding_boxes = []  # List to store bounding boxes for both hands

#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 hand_landmarks, 
#                 mp_hands.HAND_CONNECTIONS, 
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#             # Ensure there are exactly 21 landmarks (the hand keypoints)
#             if len(hand_landmarks.landmark) < 21:
#                 print("Warning: Not enough landmarks detected")
#                 continue  # Skip this hand if fewer than 21 landmarks are detected

#             # Get x and y coordinates for each landmark
#             x_ = [landmark.x for landmark in hand_landmarks.landmark]
#             y_ = [landmark.y for landmark in hand_landmarks.landmark]

#             # Normalize features using min-max scaling
#             x_min, x_max = min(x_), max(x_)
#             y_min, y_max = min(y_), max(y_)

#             x_range = x_max - x_min if x_max - x_min != 0 else 1e-6
#             y_range = y_max - y_min if y_max - y_min != 0 else 1e-6

#             data_aux = [
#                 (landmark.x - x_min) / x_range for landmark in hand_landmarks.landmark
#             ] + [
#                 (landmark.y - y_min) / y_range for landmark in hand_landmarks.landmark
#             ]

#             # Compute distances between landmarks (e.g., wrist to fingertips)
#             distances = []
#             for i in range(1, len(hand_landmarks.landmark)):
#                 for j in range(i):
#                     x1, y1 = x_[j], y_[j]
#                     x2, y2 = x_[i], y_[i]
#                     distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#                     distances.append(distance)

#             # Ensure that we are adding only 42 distance features (total 42 distances)
#             if len(distances) > 42:
#                 distances = distances[:42]  # Truncate to 42 distances if necessary

#             # Append the distances to the feature vector
#             data_aux.extend(distances)

#             # Ensure the feature vector is of the correct size (84 features)
#             if len(data_aux) != 84:
#                 print(f"Warning: Incorrect number of features. Expected 84, got {len(data_aux)}")
#                 continue  # Skip this frame if features don't match

#             # Debugging features and prediction
#             print(f"Feature vector (length {len(data_aux)}): {data_aux}")
#             prediction = model.predict([np.asarray(data_aux)])
#             print(f"Model prediction: {prediction}")

#             # Define bounding box around the hand
#             x1 = max(int(x_min * W) - 10, 0)
#             y1 = max(int(y_min * H) - 10, 0)
#             x2 = min(int(x_max * W) + 10, W - 1)
#             y2 = min(int(y_max * H) + 10, H - 1)

#             bounding_boxes.append((x1, y1, x2, y2))

#             # Predict the character based on the features
#             predicted_character = labels_dict.get(int(prediction[0]), "Unknown")  # Safe get method

#             # Draw the bounding box and the predicted character on the frame
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Draw bounding boxes for both hands if detected
#         for (x1, y1, x2, y2) in bounding_boxes:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Display the frame with the predictions
#     cv2.imshow('frame', frame)

#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()


import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Define labels
labels_dict = {0: 'Hello', 1: 'I love you', 2: 'Fuck you'}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Cannot access webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    H, W, _ = frame.shape  # Get frame dimensions
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize landmarks
            normalized_x = [x - min(x_) for x in x_]
            normalized_y = [y - min(y_) for y in y_]

            # Combine normalized features
            data_aux = normalized_x + normalized_y

            # Handle case for only one hand detected
            if len(data_aux) < 84:
                data_aux.extend([0] * (84 - len(data_aux)))

            # Ensure the input size matches 84
            if len(data_aux) == 84:
                # Predict gesture
                prediction = model.predict([np.array(data_aux)])
                predicted_label = labels_dict[int(prediction[0])]

                # Calculate bounding box
                x_min = int(min(x_) * W) - 10
                y_min = int(min(y_) * H) - 10
                x_max = int(max(x_) * W) + 10
                y_max = int(max(y_) * H) + 10

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)

                # Display prediction above the bounding box
                text_x = x_min
                text_y = y_min - 10 if y_min - 10 > 0 else y_min + 20  # Ensure text is within frame
                cv2.putText(
                    frame, predicted_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA
                )
    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
