import numpy as np
from tqdm import tqdm
import os
import glob
from argparse import ArgumentParser
import cv2
import torch
from skimage.transform import resize

from libs.utilities import make_path, _parse_metadata_file, crop_box, read_image_opencv
from libs.ffhq_cropping import crop_using_landmarks
from libs.landmarks_estimation import LandmarksEstimation

"""
If chunk videos have already been generated using download_voxCeleb.py:

1. Extract frames from chunk videos
2. Preprocess extracted frames by cropping them around the detected faces

Arguments:
	root_path: 				path where chunk videos are saved
	metadata_path: 			txt files from VoxCeleb
	dataset:				dataset name: vox1 or vox2
	delete_videos:			select to delete all videos
	delete_or_frames:		select to delete the original extracted frames

python preprocess_voxCeleb.py --root_path ./VoxCeleb1_test --metadata_path ./vox1_txt_test --dataset vox1

"""

REF_FPS = 25 		# fps to extract frames
REF_SIZE = 360 		# Height
LOW_RES_SIZE = 400 	


parser = ArgumentParser()
parser.add_argument("--root_path", default='videos', required = True, help='Path to youtube videos')
parser.add_argument("--metadata_path", default='metadata', required = True, help='Path to metadata')
parser.add_argument("--dataset", required = True, type = str, choices=('vox1', 'vox2'), help="Download vox1 or vox2 dataset")

parser.add_argument("--delete_videos", action='store_true', help='Delete chunk videos')
parser.set_defaults(delete_videos=False)
parser.add_argument("--delete_or_frames", dest='delete_or_frames', action='store_true', help="Delete original frames and keep only the cropped frames")
parser.set_defaults(delete_or_frames=False)

	
def get_frames(video_path, frames_path, video_index, fps):
	cap = cv2.VideoCapture(video_path)
	counter = 0
	# a variable to set how many frames you want to skip
	frame_skip = fps
	while cap.isOpened():
		ret, frame = cap.read()	
		if not ret:
			break		
		if counter % frame_skip == 0:
			cv2.imwrite(os.path.join(frames_path, '{:02d}_{:06d}.png'.format(video_index, counter)), frame)
		counter += 1

	cap.release()
	cv2.destroyAllWindows()


def extract_frames_opencv(videos_tmp, fps, frames_path):
	
	print('1. Extract frames')
	make_path(frames_path)
	for i in tqdm(range(len(videos_tmp))):
		get_frames(videos_tmp[i], frames_path, i, fps)
	

def preprocess_frames(dataset, output_path_video, frames_path, image_files, save_dir, txt_metadata, landmark_est = None):
	
	if dataset == 'vox2':
		image_ref = read_image_opencv(image_files[0])
		mult = image_ref.shape[0] / REF_SIZE
		image_ref = resize(image_ref, (REF_SIZE, int(image_ref.shape[1] / mult)), preserve_range=True)
	else:
		image_ref = None

	info_metadata = _parse_metadata_file(txt_metadata, dataset = dataset, frame = image_ref)

	errors = []
	chunk_id = 0
	frame_i = 0
	print('2. Preprocess frames')
	for i in tqdm(range(len(image_files))):

		# Check from which chunk video each frame is extracted.
		# Frames are saved as chunkid_index.png 
		image_file = image_files[i]
		image_name = image_file.split('/')[-1]
		image_chunk_id = image_name.split('.')[0]
		image_chunk_id = int(image_chunk_id.split('_')[0])
		bbox = None
		if chunk_id != image_chunk_id:
			chunk_id += 1
			frame_i = 0
			#########################################
		if chunk_id < len(info_metadata):
			frames = info_metadata[chunk_id]['frames']
			bboxes_metadata =  info_metadata[chunk_id]['bboxes']
			# print('Index with chunk videos every REF_FPS frames..')
			index = frame_i+1 + frame_i*(REF_FPS-1)
			if index < len(bboxes_metadata):
				bbox = bboxes_metadata[index]
				frame = frames[index]				
		
		if bbox is not None:
			image = read_image_opencv(image_file)
			frame = image.copy()
			(h, w) = image.shape[:2]	

			scale_res = REF_SIZE / float(h)
			bbox_new = bbox.copy()
			bbox_new[0] = bbox_new[0] / scale_res
			bbox_new[1] = bbox_new[1] / scale_res
			bbox_new[2] = bbox_new[2] / scale_res
			bbox_new[3] = bbox_new[3] / scale_res
			
			cropped_image, bbox_scaled = crop_box(frame, bbox_new, scale_crop = 2.0)
			filename = os.path.join(save_dir, image_name)
			cv2.imwrite(filename,  cv2.cvtColor(cropped_image.copy(), cv2.COLOR_RGB2BGR))
			h, w, _ = cropped_image.shape
			image_tensor = torch.tensor(np.transpose(cropped_image, (2,0,1))).float().cuda()
			
			if landmark_est is not None:
				with torch.no_grad():
					landmarks = landmark_est.detect_landmarks( image_tensor.unsqueeze(0))
					landmarks = landmarks[0].detach().cpu().numpy()
					landmarks = np.asarray(landmarks)
					condition = np.any(landmarks > w) or np.any(landmarks < 0)
					if (condition == False) :
						img = crop_using_landmarks(cropped_image, landmarks)
						if img is not None:
							filename = os.path.join(save_dir, image_name)
							cv2.imwrite(filename,  cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
		frame_i += 1

if __name__ == "__main__":

	
	args = parser.parse_args()
	root_path = args.root_path
	if not os.path.exists(root_path):
		print('Videos path {} does not exist'.format(root_path))

	metadata_path = args.metadata_path
	delete_videos = args.delete_videos
	delete_or_frames = args.delete_or_frames
	dataset = args.dataset

	if not os.path.exists(metadata_path):
		print('Please download the metadata for {} dataset'.format(dataset))
		exit()
	landmark_est = LandmarksEstimation(type = '2D')

	print('--Delete chunk videos: \t\t\t{}'.format(delete_videos))
	print('--Delete original frames: \t\t{}'.format(delete_or_frames))
	
	ids_path = glob.glob(os.path.join(root_path, '*/'))
	ids_path.sort()
	print('Dataset has {} identities'.format(len(ids_path)))

	data_csv = []
	data_low_res = []
	for i, id_path in enumerate(ids_path):
		id_index = id_path.split('/')[-2]
		videos_path = glob.glob(os.path.join(id_path, '*/'))
		videos_path.sort()
		print('*********************************************************')
		print('Identity {}/{}: {} videos for {} identity'.format(i, len(ids_path), len(videos_path), id_index))

		count = 0
		for j, video_path in enumerate(videos_path):
			video_id = video_path.split('/')[-2]

			print('{}/{} videos'.format(j, len(videos_path)))

			output_path_video = os.path.join(root_path, id_index, video_id)
			output_path_chunk_videos = os.path.join(output_path_video, 'chunk_videos')
			if not os.path.exists(output_path_chunk_videos):
				print('path {} does not exist.'.format(output_path_chunk_videos))
			else:
				
				txt_metadata = glob.glob(os.path.join(metadata_path, id_index, video_id, '*.txt'))
				txt_metadata.sort()
				
				############################################################
				###                   Frame extraction 					 ###
				############################################################
				videos_tmp = glob.glob(os.path.join(output_path_chunk_videos, '*.mp4'))
				videos_tmp.sort()
				extracted_frames_path = os.path.join(output_path_video, 'frames')
				if len(videos_tmp) > 0:					
					extract_frames_opencv(videos_tmp, REF_FPS, extracted_frames_path)
				else:
					print('No videos in {}'.format(output_path_video))
					count += 1
					continue
				
				
				############################################################
				###                   Preprocessing 					 ###
				############################################################
				image_files = glob.glob(os.path.join(extracted_frames_path, '*.png'))
				image_files.sort()
				if len(image_files) > 0:
					save_dir = os.path.join(output_path_video, 'frames_cropped')
					make_path(save_dir)
					preprocess_frames(dataset, output_path_video, extracted_frames_path, image_files, save_dir, txt_metadata, landmark_est)
				else:
					print('No frames in {}'.format(extracted_frames_path))

				# Delete all chunk videos
				if delete_videos:
					command_delete = 'rm -rf {}'.format(os.path.join(output_path_video, '*.mp4'))
					os.system(command_delete)
				# Delete original frames
				if delete_or_frames:
					command_delete = 'rm -rf {}'.format(os.path.join(output_path_video, frames_folder_name))
					os.system(command_delete)	
				################################################
			count += 1
		print('*********************************************************')
