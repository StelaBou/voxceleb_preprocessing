import os
import cv2
import numpy as np

def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok= True)

def _parse_metadata_file(metadata, dataset = 'vox1', frame = None):

	if dataset == 'vox1':
		metadata_info = []
		for path in metadata:
			frames = []
			bboxes = []
			with open(path) as f:
				segment_desc = f.read().strip()

			header, lines = segment_desc.split("\n\n")
			lines = lines.split("\n")

			for i in range(1,len(lines)):
				
				info = lines[i].split("\t")			
				frames.append(int(info[0]))
				x1 = int(info[1])
				y1 = int(info[2])
				x2 = int(info[1]) + int(info[3])
				y2 = int(info[2]) + int(info[4])
				bboxes.append([x1, y1, x2, y2])
				
			info = {
				'frames': 	frames,
				'bboxes': 	bboxes
			}
			metadata_info.append(info)
	elif dataset == 'vox2':
		metadata_info = []
		for path in metadata:
			frames = []
			bboxes = []
			with open(path) as f:
				segment_desc = f.read().strip()

			header, lines = segment_desc.split("\n\n")
			lines = lines.split("\n")

			for i in range(1,len(lines)):
				
				info = lines[i].split("\t")			
				frames.append(int(info[0]))
				x1 = int(float(info[1]) * frame.shape[1])
				y1 = int(float(info[2]) * frame.shape[0])
				x2 = int(float(info[3]) * frame.shape[1]) + x1
				y2 = int(float(info[4]) * frame.shape[0]) + y1
				bboxes.append([x1, y1, x2, y2])
				
			info = {
				'frames': 	frames,
				'bboxes': 	bboxes
			}
			metadata_info.append(info)

	else:
		print('Specify correct dataset')
		exit()

	return metadata_info

def crop_box(image, bbox, scale_crop = 1.0):
	
	h_im, w_im, c = image.shape
	
	y1_hat = bbox[1]
	y2_hat = bbox[3]
	x1_hat = bbox[0]
	x2_hat = bbox[2]

	new_w = x2_hat - x1_hat
	w = x2_hat - x1_hat
	h = y2_hat - y1_hat		
	cx = int(x1_hat + w/2)
	cy = int(y1_hat + h/2)

	w_hat = int(w * scale_crop) 
	h_hat = int(h * scale_crop) 
	x1_hat = cx - int(w_hat/2)
	if x1_hat < 0:
		x1_hat = 0
	y1_hat = cy - int(h_hat/2)
	if y1_hat < 0:
		y1_hat = 0
	x2_hat = x1_hat + w_hat
	y2_hat = y1_hat + h_hat
	
	if x2_hat > w_im:
		x2_hat = w_im
	if y2_hat > h_im:
		y2_hat = h_im

	if (y2_hat - y1_hat) > 20 and (x2_hat - x1_hat) > 20:
		crop = image[y1_hat:y2_hat, x1_hat:x2_hat, :]	
	else:
		crop = image

	bbox_caled = [x1_hat, y1_hat, x2_hat, y2_hat]
	return crop, bbox_caled


" Read image from path"
def read_image_opencv(image_path):
	img = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR order
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
	return img.astype('uint8')