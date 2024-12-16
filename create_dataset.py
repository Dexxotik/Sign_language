# import os
# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt

# mp_hands= mp.solutions.hands
# mp_drawing=mp.solutions.drawing_utils
# mp_drawing_styles= mp.solutions.drawing_styles

# hands= mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATADIR='./data'

# for dir in os.listdir(DATADIR):
#     for img_path in os.listdir(os.path.join(DATADIR,dir))[:1]:
#         img= cv2.imread(os.path.join(DATADIR,dir,img_path))
#         img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     img_rgb,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )
#         plt.figure()
#         plt.imshow(img_rgb)
        
# plt.show()


# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         x_ = []
#         y_ = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

# import os
# import pickle
# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []

# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         # Read and process the image
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)

#         if results.multi_hand_landmarks:
#             hand_data = []

#             # Process each detected hand
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_ = [landmark.x for landmark in hand_landmarks.landmark]
#                 y_ = [landmark.y for landmark in hand_landmarks.landmark]

#                 # Normalize landmarks relative to the hand's minimum x and y
#                 normalized_x = [x - min(x_) for x in x_]
#                 normalized_y = [y - min(y_) for y in y_]

#                 # Combine normalized x and y coordinates
#                 hand_data.extend(normalized_x + normalized_y)

#                 # Draw landmarks on the image
#                 mp_drawing.draw_landmarks(
#                     img_rgb,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style(),
#                 )

#             # Ensure the feature vector has 84 features (42 for each hand)
#             if len(results.multi_hand_landmarks) == 1:
#                 # If only one hand is detected, pad the second hand's data with zeros
#                 hand_data.extend([0] * 42)

#             data_aux.extend(hand_data[:84])  # Limit to 84 features (in case of more hands)

#             data.append(data_aux)
#             labels.append(dir_)


# # Save the data and labels to a pickle file
# with open('data.pickle', 'wb') as f:
#     pickle.dump({'data': data, 'labels': labels}, f)

import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'
output_file = 'data.pickle'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]
                normalized_x = [x - min(x_) for x in x_]
                normalized_y = [y - min(y_) for y in y_]
                hand_data.extend(normalized_x + normalized_y)

            # Pad data if only one hand is detected
            if len(results.multi_hand_landmarks) == 1:
                hand_data.extend([0] * 42)

            data.append(hand_data[:84])  # Ensure exactly 84 features
            labels.append(int(dir_))

# Save dataset
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved to {output_file}.")

