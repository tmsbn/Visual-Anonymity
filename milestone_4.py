import os
import video_face_recognizer as recognizer
import argparse
from video_face_recognizer import Stats
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import multiprocessing
from collections import deque
import time

if __name__ == '__main__':

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
	queue = deque()

	buffer_size = 100

	buffered = False

	flag = True

	context = multiprocessing

	if "forkserver" in multiprocessing.get_all_start_methods():
		context = multiprocessing.get_context("forkserver")

	number_cores = recognizer.get_number_of_cores()
	pool = context.Pool(processes=number_cores)

	frame_no = 0

	is_playing = True

	while is_playing:

		start = time.time()
		while queue and queue[0].ready():

			task = queue.popleft()
			frame = task.get()
			recognizer.play_frame(frame)
			# print('playing frame')

		if len(queue) < number_cores:

			frame = recognizer.read_frame(video_file)

			if frame is not None:
				task = pool.apply_async(recognizer.process_frame2, (frame.copy(), face_input_encodings.copy()))
				queue.append(task)
			else:
				is_playing = False
		else:
			#print('Buffer full')
			pass

		key = recognizer.get_key()

		# Press C on keyboard to detect face
		if key == ord('c'):
			target_in_frame = True
		elif key == ord('q'):
			break

		# end = time.time()
		# print('buffering', end - start)
		# recognizer.add_delay(start, end)

	print('Precision:', stats.get_precision(), 'Recall:', stats.get_recall(), 'F1 Score:')
	print('Confusion Matrix', stats.print_confusion_matrix())

