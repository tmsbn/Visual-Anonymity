import dlib
from os.path import join
import cv2


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


def get_face_locations(img, sample_rate=1):
	return face_detector(img, sample_rate)


def get_face_encoding(img, face_location):
	landmark_shape = get_landmark_shape(img, face_location)
	face_encoding = face_recognizer_model.compute_face_descriptor(img, landmark_shape)
	return face_encoding


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
