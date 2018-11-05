import face_recognition
import cv2
import os
from pykalman import KalmanFilter
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sf", help="how much to scale the image down", type=int)
parser.add_argument("--person", help="name of person")
parser.add_argument("--video", help="video path")
parser.add_argument("--skip", help="skip ever x frame", default=2, type=int)
parser.add_argument("--tolerance", help="tolerance for detection", default=0.6, type=float)


def main():
	args = parser.parse_args()

	person_name = args.person
	video_path = args.video
	frames_to_skip = args.skip

	tolerance = args.tolerance

	media_file = os.path.join('media', person_name, 'videos', video_path)
	face_files = os.path.join('media', person_name, 'images')

	known_face_encodings = []

	for file_name in os.listdir(face_files):

		file_path = os.path.join(face_files, file_name)
		if os.path.isfile(file_path) and file_path.endswith('.jpg'):
			face_image = face_recognition.load_image_file(file_path)
			face_encoding = face_recognition.face_encodings(face_image, num_jitters=5)[0]
			known_face_encodings.append(face_encoding)

	media_file = os.path.join(media_file)
	cap = cv2.VideoCapture(media_file)

	# Initialize some variables
	face_locations = []
	face_target_lst = []

	frame_no = 0

	missing_measurement_since_threshold = 21

	previous_face_location = None

	padding = 20

	previous_second = -1

	fps = 0

	# Do measurement
	TP, FP, FN, TN = 0, 0, 0, 0

	target_in_frame = False

	print('Starting...')

	total_frames = 0
	in_frames, out_frames = [], []
	while cap.isOpened() and total_frames < 700:
		# Capture frame-by-frame
		ret, frame = cap.read()
		in_frames.append(frame)
		total_frames += 1

	# print(len(frames))

	scale_factor = args.sf

	# Read until video is completed
	for frame in in_frames:

		# Capture frame-by-frame
		# ret, frame = cap.read()

		# Measure frames per second
		if previous_second != - 1:
			second = datetime.datetime.now().second
			if second - previous_second != 0:
				# print(fps)
				fps = 0
			previous_second = second
		else:
			previous_second = datetime.datetime.now().second

		target_in_frame = False

		key = cv2.waitKey(1) & 0xFF

		# Press C on keyboard to detect face
		if key == ord('c'):
			target_in_frame = True
		elif key == ord('q'):
			break

		# Only process every other frame of video to save time
		if frame_no % frames_to_skip == 0:

			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rgb_small_frame = cv2.resize(rgb_frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)

			frame_no = 0
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

			face_target_lst = []

			# print(face_encodings)

			for face_encoding in face_encodings:

				# See if the face is a match for the known face(s)

				distances = face_recognition.face_distance(known_face_encodings, face_encoding)

				should_add = True
				found = False

				for distance in distances:

					if distance <= tolerance:

						found = True

						# if we find an encoding that is very close the actual face, don't add the encoding as a part of known encodings
						if distance < (tolerance - 0.05):
							should_add = False
							break

				face_target_lst.append(found)

				if found and should_add:
					known_face_encodings.append(face_encoding)
					print(list(distances))
					print('Known face encodings added, length is now', len(known_face_encodings))

		# Display the results
		for face_location, face_target in zip(face_locations, face_target_lst):

			if face_target:
				top, right, bottom, left = face_location
				previous_face_location = (
											 int(top * scale_factor), int(right * scale_factor),
											 int(bottom * scale_factor),
											 int(left * scale_factor)), 0
				break

		if previous_face_location:

			if target_in_frame:
				TP += 1
			else:
				FP += 1

			(top, right, bottom, left), count = previous_face_location

			if count == missing_measurement_since_threshold:
				previous_face_location = None
				# print('no match found!!')
				continue

			top, right, bottom, left = top - padding, right + padding, bottom + padding, left - padding

			cropped_frame = frame[top: bottom, left: right]
			median__blur = cv2.medianBlur(cropped_frame, 27)
			frame[top: bottom, left: right] = median__blur

			# cv2.rectangle(frame,(top,left),(bottom,right),(0,255,0),3)

			previous_face_location = (previous_face_location[0], count + 1)
		else:

			if target_in_frame:
				FN += 1
			else:
				TN += 1

		face_locations = []
		frame_no += 1
		fps += 1

		out_frames.append(frame)
		# Display the resulting frame
		cv2.imshow('Frame', frame)

	# output_file = os.path.join('media', person_name, 'videos', 'output.mp4')
	# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	# out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

	# for frame in out_frames:
	# 	out.write(frame)
	#
	# cap.release()
	# out.release()


	if False:
		print('Confusion Matrix')
		print('TP', TP, 'FP', FP, 'TN', TN, 'FN', FN)
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		f1_score = 2 * (precision * recall) / (precision + recall)
		print('precision:', precision, 'Recall', recall, 'f1 score', f1_score)


if __name__ == '__main__':
	main()
