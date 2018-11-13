import cv2
import os
import video_face_recognizer as recognizer
import argparse
from video_face_recognizer import Stats
import multiprocessing as mp
import time
from queue import Queue
import threading


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
	queue = mp.Queue()

	recognizer.read_and_process_frame(video_file, face_input_encodings, queue)


	# print('Queue size', queue.qsize())
	flag = True

	while flag:

		#recognizer.play_frames(queue)
		process = mp.Process(target=recognizer.play_frames, args=(queue, ))
		process.start()

		flag = recognizer.read_and_process_frame(video_file, face_input_encodings, queue)
		# queue.task_done()

	# Measurement

	print('Precision:', stats.get_precision(), 'Recall:', stats.get_recall(), 'F1 Score:')
	print('Confusion Matrix', stats.print_confusion_matrix())

