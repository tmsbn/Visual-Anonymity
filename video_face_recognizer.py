import os
import datetime
from os.path import join
import copy
import time

import cv2
import dlib
import numpy as np
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt
from contextlib import contextmanager


# Base directory for models
MODEL_BASE_DIR = 'models'

# Load Face Recognition Model
FACE_MODEL_PATH = join(MODEL_BASE_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
LANDMARK_DETECTOR_PATH = join(MODEL_BASE_DIR, 'shape_predictor_5_face_landmarks.dat')

# Get the front face detector from dlib
face_detector = dlib.get_frontal_face_detector()
face_recognizer_model = dlib.face_recognition_model_v1(FACE_MODEL_PATH)
landmark_detector = dlib.shape_predictor(LANDMARK_DETECTOR_PATH)

DEFAULT_FRAME_WIDTH = 300
MISSING_COUNT_TOLERANCE = 10
TOLERANCE_DISTANCE = 0.6
MEDIAN_BLUR = 27
FONT_SCALE, FONT_THICKNESS = 0.8, 2

previous_face_measurement = None
missing_count = MISSING_COUNT_TOLERANCE
sampling_rate = 2

distance_diff_set = set()

LOG = False

frame_no, frame_count_per_second, previous_second, fps = 0, 0, 0, 0

fps_lst = []


class Stats:
	def __init__(self):
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0
		self.start = 0
		self.end = 0

	def total_frames(self):
		return self.TP + self.FP + self.TN + self.FN

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


stats = Stats()


def print_fps_graph():

	avg_fps = sum(fps_lst) // len(fps_lst)
	avg_fps = round(avg_fps, 2)
	plt.plot([1, len(fps_lst)], [avg_fps, avg_fps], 'r--')
	frame_nos = [x for x in range(1, len(fps_lst) + 1)]
	plt.plot(frame_nos, fps_lst, 'k--', label='avg fps:' + str(avg_fps))
	plt.xlabel('time')
	plt.legend()
	plt.ylabel('frames per second')
	plt.show()


def get_current_second():
	return datetime.datetime.now().second


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
	# print('stuck here')

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


def close_video(video):
	cv2.destroyAllWindows()
	video.release()


@contextmanager
def open_video(media_file):
	video = cv2.VideoCapture(media_file)
	yield video
	video.release()
	cv2.destroyAllWindows()


def read_frame(video):
	global frame_no

	ret, frame = video.read()
	if ret:
		frame = get_scaled_frame(frame)
		frame_no += 1
		return frame
	else:
		return None


def play_video(video, total_frames=1000):
	log('started reading ', total_frames)

	while video.isOpened() and total_frames > 0:
		ret, frame = video.read()
		frame = get_scaled_frame(frame)
		total_frames -= 1
		log('read frame', total_frames)
		yield total_frames, frame


def get_frames(video, num_frames=1000):
	frames = []
	for frame_count, frame in play_video(video, num_frames):
		frames.append(frame)

	return frames


def update_previous_measurement(measurement):
	global previous_face_measurement, missing_count, sampling_rate

	if previous_face_measurement:

		current_face_location = measurement[1]
		previous_face_location = previous_face_measurement[1]

		previous_center, current_center = find_center(previous_face_location), find_center(current_face_location)
		diff = abs(previous_center - current_center)

		if diff <= 3:
			sampling_rate = 4
		elif 3 < diff <= 10:
			sampling_rate = 3
		else:
			sampling_rate = 2
	else:
		sampling_rate = 2

	previous_face_measurement = measurement
	missing_count = MISSING_COUNT_TOLERANCE


def get_previous_measurements():
	global previous_face_measurement, missing_count

	if missing_count == 0:
		previous_face_measurement = None
	else:
		missing_count -= 1

	return previous_face_measurement


def get_number_of_cores():
	return cv2.getNumberOfCPUs()


def process_frames(frames, face_input_encoding):
	final_frames = []
	for idx, frame in enumerate(frames):
		final_frames.append(process_frame(frame, face_input_encoding))

	return final_frames


def process_frame(frame, face_input_encodings):
	global frame_no, fps

	original_frame = frame.copy()
	add_text(original_frame, 'original')
	add_text(frame, 'blur')

	# print('fps', fps)
	add_text(frame, 'SR:' + str(sampling_rate), (340, 30))

	log('sampling rate', sampling_rate)

	match_found = False

	if frame_no % sampling_rate == 0:

		face_locations = get_face_locations(frame)

		for face_location in face_locations:

			face_landmarks = get_landmark_shape(frame, face_location)
			face_encoding = get_face_encoding(frame, face_landmarks)
			match_found = check_for_match(face_encoding, face_input_encodings)

			if match_found:
				measurement = (face_landmarks, face_location)
				update_previous_measurement(measurement)

				break

	measurements = get_previous_measurements()
	if measurements:
		match_found = True
		(face_landmarks, face_location) = measurements
		blur_frame_location(frame, face_location)
		plot_landmarks(frame, face_landmarks)
		plot_rectangle(frame, face_location)

	final_frame = merge_frames(original_frame, frame)
	return final_frame, match_found


def play_frames(frames):
	for frame in frames:
		play_frame(frame)


def calculate_fps():

	global previous_second, fps, frame_count_per_second

	current_second = get_current_second()

	if current_second - previous_second != 0:
		fps = frame_count_per_second
		fps_lst.append(frame_count_per_second)
		frame_count_per_second = 0
		previous_second = current_second
	else:
		frame_count_per_second += 1


def calculate_stats(is_match, label):

	target_in_frame, target_not_in_frame = False, False

	key = get_key()

	label = True if label == 'Y' else False

	# Press C on keyboard to detect face
	if key == ord('c'):
		target_in_frame = True
	elif key == ord('x'):
		target_not_in_frame = True
	elif key == ord('q'):
		return False

	if target_in_frame or target_not_in_frame:
		if target_in_frame:
			if is_match:
				stats.TP += 1
				print('TP')
			else:
				stats.FP += 1
				print('FP')
		elif target_not_in_frame:
			if is_match:
				stats.FN += 1
				print('FN')
			else:
				stats.TN += 1
				print('TN')

	return True


def play_frame(frame):

	calculate_fps()
	cv2.imshow('Frame', frame)



def blur_frame_location(frame, face_location, padding=15):
	left, top, right, bottom = face_location.left() - padding, face_location.top() - padding, face_location.right() + padding, face_location.bottom() + padding
	cropped_frame = frame[top: bottom, left: right]
	median__blur = cv2.medianBlur(cropped_frame, MEDIAN_BLUR)
	frame[top: bottom, left: right] = median__blur


def has_previous_measurements():
	global previous_face_measurement

	return previous_face_measurement is not None


def find_center(face_location):
	left, top, right, bottom = face_location.left(), face_location.top(), face_location.right(), face_location.bottom()
	return ((left + right) + (top + bottom)) / 2


def add_text(frame, text, position=(10, 30), thickness=FONT_THICKNESS):
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



