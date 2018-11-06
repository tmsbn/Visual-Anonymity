import cv2
import os
import video_face_recognizer as recognizer
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--person", help="name of person", required=True)
parser.add_argument("--video", help="video file name", required=True)
parser.add_argument("--skip", help="skip ever x frame", default=2, type=int)
parser.add_argument("--tolerance", help="tolerance for detection", default=0.6, type=float)

args = parser.parse_args()

person_name = args.person
video_path = args.video
skip_count = args.skip

media_file_path = os.path.join('media', person_name, 'videos', video_path)
face_files_path = os.path.join('media', person_name, 'images')


video_file = recognizer.read_video(media_file_path)

face_input_encodings = recognizer.get_encodings_from_input(face_files_path)
# print(len(face_encodings))


frames = []
for frame_count, frame in recognizer.play_video(video_file, 1000):

	key = recognizer.get_key()

	# Press C on keyboard to detect face
	if key == ord('c'):
		target_in_frame = True
	elif key == ord('q'):
		break

	if frame_count % skip_count == 0:

		face_locations = recognizer.get_face_locations(frame)

		for face_location in face_locations:

			# Display the resulting frame
			face_landmarks = recognizer.get_landmark_shape(frame, face_location)

			face_encoding = recognizer.get_face_encoding(frame, face_landmarks)

			match_found = recognizer.check_for_match(face_encoding, face_input_encodings)

			if match_found:
				recognizer.blur_frame_location(frame, face_location)
				recognizer.plot_landmarks(frame, face_landmarks)
				recognizer.plot_rectangle(frame, face_location)
				break

		cv2.imshow('Frame', frame,)


print(len(frames))


