import cv2
import os
import video_face_recognizer as recognizer

media_file = os.path.join('media', 'arnold', 'videos', 'arnold_480.mp4')

media_file = os.path.join(media_file)
video_file = recognizer.read_video(media_file)


frames = []
for frame_count, frame in recognizer.play_video(video_file, 1000):

	key = recognizer.get_key()

	# Press C on keyboard to detect face
	if key == ord('c'):
		target_in_frame = True
	elif key == ord('q'):
		break

	if frame_count % 2 == 0:

		face_locations = recognizer.get_face_locations(frame)

		for face_location in face_locations:

			# Display the resulting frame
			landmarks = recognizer.get_landmark_shape(frame, face_location)

			recognizer.plot_landmarks(frame, landmarks)
			recognizer.plot_rectangle(frame, face_location)

		cv2.imshow('Frame', frame)


print(len(frames))


