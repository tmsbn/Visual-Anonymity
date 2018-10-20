import face_recognition
import cv2
import os
from pykalman import KalmanFilter
import numpy as np


def main():

	media_file = os.path.join('media', 'bale', 'videos', 'bale_1.mp4')
	face_files = os.path.join('media', 'bale', 'images')

	known_face_encodings = []

	for file_name in os.listdir(face_files):

		file_path = os.path.join(face_files, file_name)
		if os.path.isfile(file_path):
			face_image = face_recognition.load_image_file(file_path)
			face_encoding = face_recognition.face_encodings(face_image)[0]
			known_face_encodings.append(face_encoding)


	media_file = os.path.join(media_file)
	cap = cv2.VideoCapture(media_file)

	# Initialize some variables
	face_locations = []
	face_target_lst = []

	frame_no = 0

	missing_measurement_since_threshold = 20

	previous_face_location = None

	padding = 10

	# Read until video is completed
	while cap.isOpened():

		# Capture frame-by-frame
		ret, frame = cap.read()

		if ret:

			# Press Q on keyboard to  exit
			if cv2.waitKey(100) & 0xFF == ord('q'):
				break

			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Only process every three frame of video to save time
			if not previous_face_location or frame_no % 3 == 0:

				# Find all the faces and face encodings in the current frame of video
				face_locations = face_recognition.face_locations(rgb_frame)
				face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

				face_target_lst = []

				# print(face_encodings)

				for face_encoding in face_encodings:

					# See if the face is a match for the known face(s)
					matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.50)
					# If a match was found in known_face_encodings, just use the first one.
					if True in matches:
						face_target_lst.append(True)
					else:
						face_target_lst.append(False)

			# Display the results
			for face_location, face_target in zip(face_locations, face_target_lst):

				if face_target:

					previous_face_location = (face_location, 0)
					break

			if previous_face_location:

				(top, right, bottom, left), count = previous_face_location

				if count == missing_measurement_since_threshold:
					previous_face_location = None
					continue

				top, right, bottom, left = top - padding, right + padding, bottom + padding, left - padding

				cropped_frame = frame[top: bottom, left: right]
				median__blur = cv2.medianBlur(cropped_frame, 25)
				frame[top: bottom, left: right] = median__blur

				previous_face_location = (previous_face_location[0], count + 1)

			face_locations = []
			face_target_lst = []
			frame_no += 1

			# Display the resulting frame
			cv2.imshow('Frame', frame)

		# Break the loop
		else:
			break


if __name__ == '__main__':
	main()
