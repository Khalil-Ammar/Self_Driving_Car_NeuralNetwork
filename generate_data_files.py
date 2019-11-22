from PIL import Image
from string import digits
import os
import pickle
import random
import cv2
import numpy as np
import imutils

def get_average(set):
    return int(sum(set)/len(set))

def extract_data(images_path):
    os.chdir(images_path)
    data_list = []
    # nn_out_dict = {'left' : 0, 'right' : 1, 'forward': 2, 'stop': 3}
    # translate_table = str.maketrans(dict.fromkeys(digits)) ##used to extract direction from file name
    for fname in os.listdir(os.getcwd()):
        if fname.endswith(".jpg"):
            im = Image.open(fname, 'r')
            gray_im = cv2.cvtColor(np.float32(im), cv2.COLOR_BGR2GRAY)
            gray_im = imutils.resize(gray_im, width=160)
            pixels_flat = gray_im.flatten()/255 ##flatten pixel values
            # fname = os.path.splitext(fname)[0] ##remove extensions from file name
            direction = int(fname[0]) ##extract direction from file name
            data_list.append((pixels_flat,direction))

    return data_list

def create_data_file(out_file_name, data):
    fname = out_file_name
    outfile = open(fname, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

if __name__ == "__main__":
    images_path = "./warped"
    data = extract_data(images_path)

    ##shuffle and assign training and validation data
    train_data_fname = "train.pkl"
    valid_data_fname = "valid.pkl"
    random.shuffle(data)
    train_data_length = int(0.75*len(data)) ## choose training data as 75% of the dataset
    train_data = data[:train_data_length]
    valid_data = data[train_data_length:]


    ##convert data to .pkl format
    create_data_file(train_data_fname, train_data)
    create_data_file(valid_data_fname, valid_data)
