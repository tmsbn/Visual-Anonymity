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
	label_file_path = os.path.join('media', person_name, video_path[0:-4] + '_label.txt')

	face_input_encodings = recognizer.get_encodings_from_input(face_files_path)

	target_in_frame, is_match = False, False
	queue = deque()

	buffer_size = 100

	buffered = False

	flag = True

	context = multiprocessing

	with recognizer.open_video(media_file_path) as video_file:

		if "forkserver" in multiprocessing.get_all_start_methods():
			context = multiprocessing.get_context("forkserver")

		number_cores = recognizer.get_number_of_cores() - 2
		pool = context.Pool(processes=number_cores)

		frame_no = 0

		is_playing = True

		with open(label_file_path) as fp:

			label = fp.readline()

			while is_playing and label:

				start = time.time()
				while queue and queue[0].ready():

					task = queue.popleft()
					frame, is_match = task.get()

					recognizer.add_text(frame, 'fps:' + str(recognizer.fps), (10, 300))
					recognizer.play_frame(frame)
					frame_played = True

				is_playing = recognizer.calculate_stats(is_match, label)

				if len(queue) < number_cores:

					frame = recognizer.read_frame(video_file)

					if frame is not None:
						task = pool.apply_async(recognizer.process_frame, (frame.copy(), face_input_encodings.copy()))
						queue.append(task)
					else:
						is_playing = False

	stats = recognizer.stats
	print('Precision:', stats.get_precision(), 'Recall:', stats.get_recall(), 'F1 Score:', stats.get_f1_score())
	print('Total frames:', stats.total_frames())
	print('Confusion Matrix', stats.print_confusion_matrix())

	recognizer.print_fps_graph()

