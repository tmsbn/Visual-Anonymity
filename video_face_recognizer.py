import os
import time
from os.path import join

import cv2
import dlib
import numpy as np
import copy

# Base directory for models
MODEL_BASE_DIR = 'models'

# Load Face Recognition Model
FACE_MODEL_PATH = join(MODEL_BASE_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
LANDMARK_DETECTOR_PATH = join(MODEL_BASE_DIR, 'shape_predictor_5_face_landmarks.dat')

# Get the front face detector from dlib
face_detector = dlib.get_frontal_face_detector()
face_recognizer_model = dlib.face_recognition_model_v1(FACE_MODEL_PATH)
landmark_detector = dlib.shape_predictor(LANDMARK_DETECTOR_PATH)
# aligner = face_aligner.AlignDlib(LANDMARK_DETECTOR_PATH)

DEFAULT_FRAME_WIDTH = 256
MISSING_COUNT_TOLERANCE = 20
TOLERANCE_DISTANCE = 0.6
MEDIAN_BLUR = 27
FONT_SCALE, FONT_THICKNESS = 1, 1

previous_face_measurement = None
missing_count = MISSING_COUNT_TOLERANCE

distance_diff_set = set()

LOG = True


def log(*args):

	if LOG:
		print(*args)

class Color:
	red = (255, 0, 0)
	green = (0, 255, 0)


def load_image(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
			face_encodings.append(face_encoding)

	return face_encodings


def get_face_encoding(img, face_location):
	face_encoding = face_recognizer_model.compute_face_descriptor(img, face_location)
	return np.array(face_encoding)


def get_scaled_frame(frame):
	width, height = frame.shape[:2]
	scale_factor = DEFAULT_FRAME_WIDTH / width
	# print(scale_factor)
	return cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)


def enter_to_continue():
	dlib.hit_enter_to_continue()


def get_landmark_shape(img, face_location):
	return landmark_detector(img, face_location)


def plot_landmarks(img, landmarks):
	for idx in range(landmarks.num_parts):
		x, y = landmarks.part(idx).x, landmarks.part(idx).y
		cv2.circle(img, (x, y), 1, Color.red, -1)  # -1 for filled circles


def plot_rectangle(img, face_location):
	left, top, right, bottom = face_location.left(), face_location.top(), face_location.right(), face_location.bottom()
	cv2.rectangle(img, (left, top), (right, bottom), Color.green, 2)


def get_key(wait=1):
	return cv2.waitKey(wait) & 0xFF


def add_delay(start_time, end_time):

	diff = end_time - start_time
	if diff < 0.05:
		time.sleep(0.05 - diff)


def read_video(media_file):
	return cv2.VideoCapture(media_file)


def get_frames_gen(video, total_frames):
	while video.isOpened() and total_frames >= 0:
		ret, frame = video.read()
		frame = get_scaled_frame(frame)
		total_frames -= 1
		yield total_frames, frame


def get_frames(video, num_frames):

	frames = []
	for frame_count, frame in get_frames_gen(video, num_frames):

		log('read frame', frame_count)
		frames.append(frame)

	return frames


def process_frame(frames, face_input_encodings, sampling_rate=1):

	for frame_count, frame in enumerate(frames):

		log('Processing frame ', frame_count)

		if frame_count % sampling_rate == 0:

			face_locations = get_face_locations(frame)

			for face_location in face_locations:

				# Display the resulting frame
				face_landmarks = get_landmark_shape(frame, face_location)
				face_encoding = get_face_encoding(frame, face_landmarks)
				match_found = check_for_match(face_encoding, face_input_encodings)

				if match_found:

					plot_landmarks(frame, face_landmarks)
					blur_frame_location(frame, face_location)
					plot_rectangle(frame, face_location)

					break


def read_and_process_frame(video, face_input_encodings, queue, num_frames=100, sampling_rate=1):

	log('Started Process!')
	frames = get_frames(video, num_frames)
	original_frames = copy.deepcopy(frames)
	process_frame(frames, face_input_encodings, sampling_rate)
	queue.put((frames, original_frames))
	log('Ended Process!')


def blur_frame_location(frame, face_location, padding=15):

	left, top, right, bottom = face_location.left() - padding, face_location.top() - padding, face_location.right() + padding, face_location.bottom() + padding
	cropped_frame = frame[top: bottom, left: right]
	median__blur = cv2.medianBlur(cropped_frame, MEDIAN_BLUR)
	frame[top: bottom, left: right] = median__blur


def update_previous_location(measurement):

	global previous_face_measurement, missing_count

	previous_face_measurement = measurement
	missing_count = MISSING_COUNT_TOLERANCE


def has_previous_measurements():

	global previous_face_measurement

	return previous_face_measurement is not None


def get_previous_measurements():

	global previous_face_measurement, missing_count

	temp = previous_face_measurement

	if missing_count == 0:
		previous_face_measurement = None
	else:
		missing_count -= 1

	return temp


def find_center(face_location):
	left, top, right, bottom = face_location.left(), face_location.top(), face_location.right(), face_location.bottom()
	return ((left + right) + (top + bottom)) / 2


def get_sampling_rate(current_face_location):

	global previous_face_measurement
	previous_face_location = previous_face_measurement[0]
	previous_center, current_center = find_center(previous_face_location), find_center(current_face_location)
	diff = abs(previous_center - current_center)

	if diff <= 3:
		return 4
	elif 3 < diff <= 10:
		return 3
	else:
		return 2


def add_text(frame, text, position=(10, 30)):

	cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS, cv2.LINE_AA)


def copy_frame(frame):
	return np.copy(frame)


def merge_frames(frame1, frame2):
	return np.concatenate((frame1, frame2), axis=1)


def check_for_match(face_encoding, input_encodings, update_encodings=False):

	if face_encoding.size == 0:
		return False

	contains_match = False
	contains_close_match = False

	for input_encoding in input_encodings:
		distance = np.linalg.norm(input_encoding - np.array(face_encoding))

		if distance <= TOLERANCE_DISTANCE:
			contains_match = True

		if distance < TOLERANCE_DISTANCE * 0.95:
			contains_close_match = True
			break

	if update_encodings and contains_match and not contains_close_match:
		input_encodings.append(face_encoding)
		print('Encoding updated, length is now', len(input_encodings))

	return contains_match


class Stats:

	def __init__(self):
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0
		self.start = 0
		self.end = 0

	def start_timer(self):
		self.start = time.time()

	def end_timer(self):
		self.start = time.time()

	def update_delay(self):
		self.end = time.time()
		return self.start - self.end

	def get_precision(self):

		if self.TP + self.FP == 0:
			return 0
		else:
			return self.TP / (self.TP + self.FP)

	def get_recall(self):

		if self.TP + self.FN == 0:
			return 0
		else:
			return self.TP / (self.TP + self.FN)

	def get_f1_score(self):
		precision, recall = self.get_precision(), self.get_recall()

		if precision + recall == 0:
			return 0
		else:
			return 2 * (precision * recall) / (precision + recall)

	def print_confusion_matrix(self):
		print('TP:', self.TP, 'FP:', self.FP, 'TN:', self.TN, 'FN:', self.FN)



