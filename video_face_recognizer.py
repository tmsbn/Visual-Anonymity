import dlib
from os.path import join
import cv2
import os
import numpy as np

# Base directory for models
MODEL_BASE_DIR = 'models'

# Load Face Recognition Model
FACE_MODEL_PATH = join(MODEL_BASE_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
LANDMARK_DETECTOR_PATH = join(MODEL_BASE_DIR, 'shape_predictor_5_face_landmarks.dat')

# Get the front face detector from dlib
face_detector = dlib.get_frontal_face_detector()
face_recognizer_model = dlib.face_recognition_model_v1(FACE_MODEL_PATH)
landmark_detector = dlib.shape_predictor(LANDMARK_DETECTOR_PATH)

DEFAULT_FRAME_WIDTH = 256


def load_image(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	# return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image


def get_face_locations(img, sample_rate=1):
	return face_detector(img, sample_rate)


def get_encodings_from_input(face_files_path):
	face_encodings = []
	for file_name in os.listdir(face_files_path):

		file_path = os.path.join(face_files_path, file_name)
		if os.path.isfile(file_path) and file_path.endswith('.jpg'):
			face_image = load_image(file_path)
			face_location = get_face_locations(face_image)[0]
			face_landmarks = landmark_detector(face_image, face_location)
			face_encoding = get_face_encoding(face_image, face_landmarks)
			# print('face encoding', face_encoding)
			face_encodings.append(face_encoding)

	return face_encodings


def get_face_encoding(img, face_location):
	face_encoding = face_recognizer_model.compute_face_descriptor(img, face_location)
	return np.array(face_encoding)


def get_scaled_frame(frame):
	width, height = frame.shape[:2]
	scale_factor = DEFAULT_FRAME_WIDTH / width
	return cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)


def enter_to_continue():
	dlib.hit_enter_to_continue()


def get_landmark_shape(img, face_location):
	return landmark_detector(img, face_location)


def plot_landmarks(img, landmarks):
	for idx in range(landmarks.num_parts):
		x, y = landmarks.part(idx).x, landmarks.part(idx).y
		cv2.circle(img, (x, y), 1, (0, 0, 255), -1)  # -1 for filled circles


def plot_rectangle(img, face_location):
	left, top, right, bottom = face_location.left(), face_location.top(), face_location.right(), face_location.bottom()
	cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)


def get_key(wait=1):
	return cv2.waitKey(wait) & 0xFF


def read_video(media_file):
	return cv2.VideoCapture(media_file)


def play_video(video, total_frames):
	while video.isOpened() and total_frames >= 0:
		ret, frame = video.read()
		frame = get_scaled_frame(frame)
		total_frames -= 1
		yield total_frames, frame


def blur_frame_location(frame, face_location, padding=15):

	left, top, right, bottom = face_location.left() - padding, face_location.top() - padding, face_location.right() + padding, face_location.bottom() + padding
	cropped_frame = frame[top: bottom, left: right]
	median__blur = cv2.medianBlur(cropped_frame, 27)
	frame[top: bottom, left: right] = median__blur


def check_for_match(face_encoding, input_encodings, update_encodings=True):
	contains_match = False
	contains_close_match = False

	for input_encoding in input_encodings:
		distance = np.linalg.norm(input_encoding - face_encoding)

		if distance <= 0.65:
			contains_match = True
			# print(distance)

		if distance < 0.6:
			contains_close_match = True
			break

	if update_encodings and contains_match and not contains_close_match:
		input_encodings.append(face_encoding)
		print('Encoding updated, length is now', len(input_encodings))

	return contains_match
