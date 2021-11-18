import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import InceptionresnetV1

import os
import cv2
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from prettytable import PrettyTable


use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# randomly rotates or flips or crop head, chin, left-right face from the image
def random_augment(sample, chance, top_offset=-15, bottom_offset=-15, left_offset=20, right_offset=-20):
    image = sample[0]
    margin = sample[1]
    lmark = sample[2]
    
    isChange = np.random.choice(2, p=[1-chance, chance])
    final_img = image[margin:-margin, margin:-margin]
    if not isChange:
        return final_img
    
    choice = np.random.randint(8)
    size_y, size_x, _ = final_img.shape  
    
    if choice == 0: # crop head
        if lmark[1,0] >= lmark[0,0]:
            top = lmark[0,0] + top_offset
        else:
            top = lmark[1,0] + top_offset
        top = round(min(max(top, 0), size_y-1))
        top_cut = final_img[top:].copy()
        top_cut = cv2.resize(top_cut, (image_size, image_size))
        return top_cut
    elif choice == 1: # crop chin
        if lmark[3,0] >= lmark[4,0]:
            bottom = lmark[4,0] + bottom_offset
        else:
            bottom = lmark[3,0] + bottom_offset
        bottom = round(min(max(bottom, 1), size_y))
        bottom_cut = final_img[:bottom].copy()    
        bottom_cut = cv2.resize(bottom_cut, (image_size, image_size))
        return bottom_cut
    elif choice == 2: # crop left side
        if lmark[3,1] >= lmark[0,1]:
            left = lmark[3,1] + left_offset
        else:
            left = lmark[0,1] + left_offset
        left = round(min(max(left, 0), size_x-1))
        left_cut = final_img[:, left:].copy()
        left_cut = cv2.resize(left_cut, (image_size, image_size))
        return left_cut
    elif choice == 3: # crop right side
        if lmark[1,1] >= lmark[4,1]:
            right = lmark[4,1] + right_offset
        else:
            right = lmark[1,1] + right_offset
        right = round(min(max(right, 1), size_x))
        right_cut = final_img[:, :right].copy()
        right_cut = cv2.resize(right_cut, (image_size, image_size))
        return right_cut
    elif choice == 4: # flip
        pil_image = Image.fromarray(image)
        mirror_image = np.array(ImageOps.mirror(pil_image))[margin:-margin, margin:-margin]
        return mirror_image
    else: # rotate
        pil_image = Image.fromarray(image)
        rotation_choice = np.concatenate([[a, -a] for a in np.arange(10, 90, 10)])
        angle = np.random.choice(rotation_choice)
        rot = np.array(pil_image.rotate(angle))
        rotation_img = rot[margin:-margin, margin:-margin]
        return rotation_img

def get_squared_distance(embeddings):
    cosine = torch.matmul(embeddings, embeddings.T)
    squared_distance = 2 * (1 - cosine)
    x = torch.max(squared_distance, torch.zeros_like(squared_distance).to(device))
    return x

# applied online triplet loss between all possible embedding triplets (A, P, N) from the batch
# A anchor embedding
# P embedding from same class as A
# N embedding from differnet class than A
# loss = max(|| A - P ||^2 - || A - N ||^2 + margin, 0)
# stride contains the length of same class examples
def all_pair_batch_loss(strides, embeddings, margin):
    distance = get_squared_distance(embeddings)
    triplet_distance = distance.unsqueeze(2) - distance.unsqueeze(1) + margin
    curr = 0
    mask = torch.zeros(len(embeddings), len(embeddings)).to(device)
    
    for i in range(len(strides)):
        mask[curr : curr + strides[i], curr : curr + strides[i]] = 1.0
        curr += strides[i]
    
    n_mask = 1 - mask
    mask.fill_diagonal_(0)
    valid_triplets = mask.unsqueeze(2) * n_mask.unsqueeze(1)
    all_distance = triplet_distance * valid_triplets
    valid_distance = all_distance[all_distance > 0] 
    pos_valid_distance = torch.max(valid_distance, torch.zeros_like(valid_distance).to(device))
    new_valid_distance = pos_valid_distance[pos_valid_distance > 0]
    
    if len(new_valid_distance):
        return new_valid_distance.mean()
    
    return torch.Tensor(0)


# train loop
# evaluate after every epoch
def train(model, data, train_index, val_index, optimizer, scheduler, logger, threshold=0.85, epoch=10, max_batch_size=16,\
          val_batch_size=32, start_epoch=0, grad_step=0, save_every=2, path_to_save='', save_prefix=''):
    step_per_epoch = max(len(train_index) // max_batch_size, 1)
    val_acc = 0
    epoch += start_epoch
    model.train()
    
    for i in range(start_epoch, epoch):
        end = 0
        model.train()
        
        for j in range(step_per_epoch):
            start = end
            end = min(end + max_batch_size, len(train_index))
            batch = []
            stride = []
            
            for pos_index in range(start, end):
                req = max_batch_size - len(batch)
                if req <= 1:
                    end = pos_index
                    break
                
                stride.append(min(4, len(data[train_index[pos_index]]), req))
                index = np.random.choice(len(data[train_index[pos_index]]), size=stride[-1], replace=False)
                
                for indice in index:
                    batch.append(random_augment(data[train_index[pos_index]][indice], 0.6))
            
            # normalize data
            batch = torch.Tensor(batch).permute(0, 3, 1, 2).to(device)
            batch = (batch - 127.5) / 128
            
            embeddings = model(batch)
            loss = all_pair_batch_loss(stride, embeddings, 0.8)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.add_scalar('train_loss', loss.item(), grad_step)
                print('step[%d/%d] epoch=%d loss=%f ' % (j+1, step_per_epoch, i, loss.item()))
                grad_step += 1
            else:
                print('no hard or semi-hard negatives found!')
        
        # validate model after epoch
        with torch.no_grad():
            model.eval()
            num_batches = len(val_index) // val_batch_size
            end = 0
            div = 0
            val_acc = 0
            
            for j in range(num_batches):
                start = end
                end = min(end + val_batch_size, len(val_index))
                batch = []
                stride = []
                
                for pos_index in range(start, end):
                    req = val_batch_size - len(batch)
                    if req <= 1:
                        end = pos_index
                        break
                    
                    stride.append(min(4, len(data[val_index[pos_index]]), req))
                    index = np.random.choice(len(data[val_index[pos_index]]), size=stride[-1], replace=False)

                    for indice in index:
                        batch.append(random_augment(data[val_index[pos_index]][indice], 0))
                
                batch = torch.Tensor(batch).permute(0, 3, 1, 2).to(device)
                batch = (batch - 127.5) / 128
                embeddings = model(batch)
                # based on cosine similarity and threshold, determine match or no match 
                cosine = torch.matmul(embeddings, embeddings.T)
                target = torch.zeros_like(cosine).long()
                curr = 0
                
                for k in range(len(stride)):
                    target[curr : curr + stride[k], curr : curr + stride[k]] = 1
                    curr += stride[k]
                
                cosine[cosine > threshold] = 1
                cosine[cosine <= threshold] = 0
                cosine = cosine.long()
                
                val_acc += (cosine == target).sum() 
                div += (cosine.shape[0]**2)
            
            val_acc = val_acc / max(div, 1)
            logger.add_scalar('val_loss', val_acc, i)
            print('....epoch complete, val accuracy %f.... ' % val_acc)
        
        # save model
        if (i+1) % save_every == 0:
            torch.save({
                'epoch': i,
                'grad_step' : grad_step,
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'val_acc' : val_acc,
                }, path_to_save + '/checkpoint_' + save_prefix + str(i) + '.pth')
        
        # shuffle index for stable training
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)


# preprocessed data loaded 
# data was earlier processed using MTCNN and the bounding stored in a corresponding bbox file
# in loading this we ensure some extra width is given so that rotation of the image doesn't give black patch on the side
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



def main():
    root_path = os.getcwd() + '/'
    data_path = root_path + 'trainset/'
    image_size = 299
    req_margin = 38  # extra margin for augmentation. Later the image is cropped to image_size * image_size
    data = []

    model = InceptionresnetV1()

    if use_cuda:
        model.cuda()

    data = load_data(data_path, image_size, req_margin)

    model_save_path = root_path + 'model_weights/'

    # always fixes the index choosen for training and validation so that it doesn't get mixed up when resuming from a checkpoint
    index_path = os.path.join(model_save_path , 'index')
    if os.path.isfile(index_path):
        with open(index_path, 'rb') as f:
            M = pickle.load(f)
        train_index, val_index = M['train_index'], M['val_index']
    else:
        all_index = np.random.permutation(len(data))
        train_val_split = 0.8
        train_index = all_index[:int(train_val_split * len(data))]
        val_index = all_index[len(train_index):]
        M = {'train_index' : train_index, 'val_index' : val_index}
        with open(index_path, 'wb') as f:
            pickle.dump(M, f)

    # use OneCycleLR that increases the LR in first half and later reduces it to encourage better landscape exploration and convergence
    batch_size = 32
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=70, steps_per_epoch=max(len(train_index) // batch_size, 1))
    logger = SummaryWriter(os.path.join(model_save_path, 'runs'))

    # pretrained weights copied from casia model
    def load_pretrained_weight(path, model):
        state_dict = torch.load(path, map_location=device)
        L1 = list(model.parameters())
        
        with torch.no_grad():
            i = 0
            for key in state_dict.keys():
                if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                    continue
                else:
                    L1[i].data = state_dict[key].data
                    i += 1
                if i >= len(L1):
                    break

    def load_model(checkpoint_path, model, optimizer, scheduler):
        if not os.path.isfile(checkpoint_path):
            print('file not found', checkpoint_path)
            return 0, 0
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('resuming from %d epoch' % checkpoint['epoch'])

        return checkpoint['epoch'], checkpoint['grad_step']

    weight_path = root_path + '20180408-102900-casia-webface.pt'
    model_path = os.path.join(model_save_path, 'checkpoint_8.pth')
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--model_path', default=model_path, nargs=1, help='checkpoint file')
    parser.add_argument('--weight_path', default=weight_path, nargs=1, help='pretrained .pt or .pth file')

    start_epoch, grad_step = 0, 0
    if args.pretrained:
        load_pretrained_weight(args.weight_path, model)
    elif args.checkpoint:
        start_epoch, grad_step = load_model(args.model_path, model, optimizer, scheduler)

    try:
        train(model, data, train_index, val_index, optimizer, scheduler, logger, epoch=70, max_batch_size=batch_size,\
              save_every=3, start_epoch=start_epoch, grad_step=grad_step, path_to_save=model_save_path)
    except:
        logger.close()


if __name__ == '__main__':
    main()
