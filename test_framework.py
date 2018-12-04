import os
import video_face_recognizer as recognizer
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--person", help="name of person", required=True)
	parser.add_argument("--video", help="video file name", required=True)

	args = parser.parse_args()

	person_name = args.person
	video_name = args.video

	media_file_path = os.path.join('media', person_name, 'videos', video_name)
	face_files_path = os.path.join('media', person_name, 'images')
	label_file_path = os.path.join('media', person_name, video_name[0:-4] + '_label.txt')

	if not os.path.isfile(label_file_path):
		with open(label_file_path, 'w') as label_file:

			with recognizer.open_video(media_file_path) as video_file:

				for frame_count, frame in recognizer.play_video(video_file):
					key = recognizer.get_key(100)
					if key == ord('c'):
						label_file.write('Y\n')
					elif key == ord('q'):
						break
					else:
						label_file.write('N\n')

					recognizer.play_frame(frame)


	else:
		print('File already exists!')



