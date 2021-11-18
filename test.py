import os
import sys
import cv2
import pickle
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from model import InceptionresnetV1
from facenet_pytorch import MTCNN

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
image_size = 299
req_margin = 38
threshold = 0.93


def compare_and_score(model, p, q):
	if p is None or q is None:
		return 0
    
	model.eval()
	p = torch.Tensor(p).unsqueeze(0).permute(0, 3, 1, 2).to(device)
	q = torch.Tensor(q).unsqueeze(0).permute(0, 3, 1, 2).to(device)
	p = (p - 127.5) / 128
	q = (q - 127.5) / 128

	with torch.no_grad():
	    p = model(p).squeeze(0)
	    q = model(q).squeeze(0)

	return p.dot(q).item()


def load_data(data_path, image_size, req_margin):
    data = []

    for root, subdirs, filename in os.walk(data_path, topdown=True):
        if len(filename):
            image_name = [f for f in filename if 'bbox' not in f]
            data.append([])

            for i, img_name in enumerate(image_name):
                bbox = img_name.split('.')[0] + '_bbox'
                if bbox not in filename:
                    print('not found', bbox)
                    continue
                
                bbox_path = os.path.join(root, bbox)
                with open(bbox_path, 'rb') as f:
                    Map = pickle.load(f)
                
                arr = cv2.imread(os.path.join(root, img_name))
                img = arr[:,:,::-1]
                
                out = Map['box']
                h = out[3] - out[1] + 1
                w = out[2] - out[0] + 1
                
                r_h = h / image_size
                r_w = w / image_size
                size_y, size_x, _ = img.shape
                
                margin = min(min(out[1] / r_h, req_margin), min(out[0] / r_w, req_margin), min((size_x - out[2]) / r_w, req_margin), min((size_y - out[3]) / r_h, req_margin))
                
                if margin <= 0:
                    print('margin not positive', margin)
                    continue
                
                margin = int(margin)
                offset_h = margin * r_h
                offset_w = margin * r_w
                
                x = img[round(out[1] - offset_h) : round(out[3] + offset_h) + 1, round(out[0] - offset_w) : round(out[2] + offset_w) + 1]
                y = cv2.resize(x, (image_size + 2 * margin, image_size + 2 * margin))

                origin = np.array([out[0], out[1]], dtype=np.float32)
                lmark = np.array([Map['left_eye'], Map['right_eye'], Map['nose'], Map['left_lips'], Map['right_lips']], dtype=np.float32)
                lmark = (lmark - origin) / np.array([r_w, r_h], dtype=np.float32)
                lmark = lmark[:,::-1]
                
                data[-1].append([y, margin, lmark])
            
            if len(data[-1]) <= 1:
                data.pop()
            print('processed', root, len(data[-1]))

    return data

def get_image(sample):
    image = sample[0]
    margin = sample[1]
    final_img = image[margin:-margin, margin:-margin]
    
    return final_img


def get_accuracy_plot(model, data, val_index, val_batch_size, max_size=2000):
    space = np.arange(-1, 1.01, 0.1)
    plots = []
    model.eval()

    for threshold in space:
        with torch.no_grad():
            num_batches = int(np.ceil(len(val_index) / val_batch_size))
            end = 0
            div = 0
            val_acc = 0
            embeddings = []
            stride = []

            for i in range(num_batches):
                start = end
                end = min(end + val_batch_size, len(val_index))
                batch = []

                if sum(stride) >= 2000:
                    print('breaking due to oversized matrix')
                    break

                for pos_index in range(start, end):
                    req = val_batch_size - len(batch)
                    if req <= 0:
                        end = pos_index
                        break
                    
                    stride.append(len(data[val_index[pos_index]]))

                    for indice in range(len(data[val_index[pos_index]])):
                        batch.append(get_image(data[val_index[pos_index]][indice]))

                batch = torch.Tensor(batch).permute(0, 3, 1, 2).to(device)
                batch = (batch - 127.5) / 128
                batch_embeddings = model(batch)
                embeddings.append(batch_embeddings)

            embeddings = torch.cat(embeddings, axis=0)
            cosine = embeddings @ embeddings.T
            target = torch.zeros_like(cosine).long()
            curr = 0
            
            for k in range(len(stride)):
                target[curr : curr + stride[k], curr : curr + stride[k]] = 1
                curr += stride[k]
            
            target = target.to(device)
            cosine[cosine > threshold] = 1
            cosine[cosine <= threshold] = 0
            cosine = cosine.long()

            total_case = (cosine.shape[0]**2)
            total_pp = torch.sum(cosine).item()
            total_ap = torch.sum(target).item()
            total_pn = total_case - total_pp
            total_an = total_case - total_ap
            
            tp = torch.sum(cosine * target).item()
            fp = total_pp - tp
            tn = total_an - fp
            fn = total_ap - tp
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            acc = (tp + tn) / (total_case)
            plots.append((tp, fp, fn, tn))
            
            print('......threshold %f......' % threshold)
            print('.... val accuracy %f .... ' % acc)
            print('\n tp %d fp %d\n fn %d tn %d\n' % (tp, fp, fn, tn))
            print('tpr %f fpr %f' % (tpr, fpr))
            print('.....................')

    return plots


def read_image(mtcnn, path):
    arr = cv2.imread(path)
    img = arr[:,:,::-1].copy()
    images = torch.Tensor(img).unsqueeze(0).to(device)
    out, prob, landmark = mtcnn.detect(images, landmarks=True)

    if out[0] is None:
    	return 0

    out = out[0][0]
    h = out[3] - out[1] + 1
    w = out[2] - out[0] + 1
    
    r_h = h / image_size
    r_w = w / image_size
    size_y, size_x, _ = img.shape
    
    margin = min(min(out[1] / r_h, req_margin), min(out[0] / r_w, req_margin), min((size_x - out[2]) / r_w, req_margin), min((size_y - out[3]) / r_h, req_margin))
    
    if margin <= 0:
        print('margin not positive', margin)
        return None
    
    margin = int(margin)
    offset_h = margin * r_h
    offset_w = margin * r_w
    
    x = img[round(out[1] - offset_h) : round(out[3] + offset_h) + 1, round(out[0] - offset_w) : round(out[2] + offset_w) + 1]
    y = cv2.resize(x, (image_size + 2 * margin, image_size + 2 * margin))
    final_img = y[margin:-margin, margin:-margin]
    
    return final_img

def load_model(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        print('file not found', checkpoint_path)
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('loaded model with %d epoch' % checkpoint['epoch'])


def main():
	root_path = os.getcwd()
	data_path = os.path.join(root_path, 'trainset')
	model_path = os.path.join(root_path, 'model_weights/checkpoint_8.pth')
	index_path = os.path.join(root_path , 'model_weights/index')

	parser = argparse.ArgumentParser(description='test script')
	parser.add_argument('--gpu', default=False, action='store_true')
	parser.add_argument('--benchmark', default=False, action='store_true')
	parser.add_argument('--image_path', metavar='path', default=['',''], nargs=2, help='two image path(seperated by spaces)')
	parser.add_argument('--model_path', default=model_path, nargs=1, help='.pt or .pth file')
	parser.add_argument('--index_path', default=index_path, nargs=1, help='validation index file')
	parser.add_argument('--data_path', default=data_path, nargs=1, help='path to parsed data')
	args = parser.parse_args()

	if not args.gpu:
		global use_cuda, device
		use_cuda = False
		device = torch.device('cpu')

	model = InceptionresnetV1()

	if use_cuda:
	    model.cuda()

	load_model(model, args.model_path)

    # plots a tpr vs fpr curve on val data(requires val index file)
    # assumes data is preprocessed by MTCNN
	if args.benchmark:
	    with open(args.index_path, 'rb') as f:
	        M = pickle.load(f)
	        train_index, val_index = M['train_index'], M['val_index']
	
	    data = load_data(args.data_path, image_size, req_margin)
	    plots = get_accuracy_plot(model, data, val_index, 32)
	    x, y = [], []
	    for i in range(len(plots)):
	        tp = plots[i][0]
	        fp = plots[i][1]
	        tn = plots[i][3]
	        fn = plots[i][2]
	        fpr = fp / (fp + tn)
	        tpr = tp / (tp + fn)
	        x.append(fpr)
	        y.append(tpr)

	    fig, axs = plt.subplots(figsize=(15, 10))
	    axs.plot(x, y, color='green')
	    axs.plot(np.arange(0,1.01,1), color='red')
	    plt.ylabel('true positive rate')
	    plt.xlabel('false positive rate')
	else:
		mtcnn = MTCNN(image_size=200, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=False, device=device)
		score = compare_and_score(model, read_image(mtcnn, args.image_path[0]), read_image(mtcnn, args.image_path[1]))
		score = (score + 1) / 2
		if score >= threshold:
			print('match, confidence score=%f' % score)
		else:
			print('no match, confidence score=%f' % score)

if __name__ == '__main__':
	main()
