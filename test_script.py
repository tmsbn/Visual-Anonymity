import cv2
import os
import video_face_recognizer as recognizer
import argparse
from video_face_recognizer import Stats

parser = argparse.ArgumentParser()


parser.add_argument("--person", help="name of person", required=True)
parser.add_argument("--video", help="video file name", required=True)
parser.add_argument("--sample", help="skip ever x frame", default=2, type=int)
parser.add_argument("--tolerance", help="tolerance for detection", default=0.6, type=float)

args = parser.parse_args()

person_name = args.person
video_path = args.video
sampling_rate = args.sample

media_file_path = os.path.join('media', person_name, 'videos', video_path)
face_files_path = os.path.join('media', person_name, 'images')


video_file = recognizer.read_video(media_file_path)

face_input_encodings = recognizer.get_encodings_from_input(face_files_path)

stats = Stats()

target_in_frame = False


for frame_count, frame in recognizer.play_video(video_file, 1000):

	key = recognizer.get_key()

	# Press C on keyboard to detect face
	if key == ord('c'):
		target_in_frame = True
	elif key == ord('q'):
		break

	match_found = False

	if frame_count % sampling_rate == 0:

		face_locations = recognizer.get_face_locations(frame)

		for face_location in face_locations:

			# Display the resulting frame
			face_landmarks = recognizer.get_landmark_shape(frame, face_location)
			face_encoding = recognizer.get_face_encoding(frame, face_landmarks)
			match_found = recognizer.check_for_match(face_encoding, face_input_encodings)

			if match_found:
				measurement = (face_location, face_landmarks)

				if recognizer.has_previous_measurements():
					sampling_rate = recognizer.get_sampling_rate(face_location)

				recognizer.update_previous_location(measurement)

				break

	# Check if previous measurement exists
	if recognizer.has_previous_measurements():
		previous_measurement = recognizer.get_previous_measurements()
		current_face_location, current_face_landmarks = previous_measurement
		recognizer.blur_frame_location(frame, current_face_location)
		recognizer.plot_landmarks(frame, current_face_landmarks)
		recognizer.plot_rectangle(frame, current_face_location)

		if target_in_frame:
			stats.TP += 1
		else:
			stats.FP += 1
	else:

		# Reset sampling rate
		sampling_rate = 2

		if target_in_frame:
			stats.FN += 1
		else:
			stats.TN += 1

	print('Sampling Rate', sampling_rate)

	cv2.imshow('Frame', frame,)

# Measurement
print('Precision:', stats.get_precision(), 'Recall:', stats.get_recall(), 'F1 Score:')
print('Confusion Matrix', stats.print_confusion_matrix())

