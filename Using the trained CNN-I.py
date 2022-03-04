# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:21:17 2022

@author: cy2
"""

####Using CNN-II

#Test CNN-II
from __future__ import absolute_import, division, print_function, unicode_literals
#import pathlib
import tensorflow as tf
import os

##########################################
# Parameters manually entered by the user 
file_fold_path=os.path.abspath('./Illustrative examples')   #%%The path to the folder where the test images are located 
test_image_paths=['C:\\Users\\cy2\\Desktop\\Structural symmetry recognition\\Illustrative examples\\1.png','C:\\Users\\cy2\\Desktop\\Structural symmetry recognition\\Illustrative examples\\2.png',
                  'C:\\Users\\cy2\\Desktop\\Structural symmetry recognition\\Illustrative examples\\3.png','C:\\Users\\cy2\\Desktop\\Structural symmetry recognition\\Illustrative examples\\4.png']


###################################
#%% 
 
#def file_name(file_dir):   
#6         for root, dirs, files in os.walk(file_dir):  
#7             print(root) #当前目录路径  
#8             print(dirs) #当前路径下所有子目录  
#9             print(files) #当前路径下所有非目录子文件  



model=tf.keras.models.load_model(r'./Trained_CNN-I.h5')
#%%
def load_and_preprocess_from_path(path):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=1)   
    image = tf.image.resize(image, [200, 200])  
    image /= 255.0  
    return image

AUTOTUNE = tf.data.experimental.AUTOTUNE
test_image_paths_ds = tf.data.Dataset.from_tensor_slices(test_image_paths) 
X_test_ds=test_image_paths_ds.map(load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
ds = X_test_ds.repeat()
ds = ds.batch(batch_size=1)
X_test = ds.prefetch(buffer_size=AUTOTUNE)
test_image_count=len(test_image_paths)
Y_predict_origin=model.predict(X_test,steps=test_image_count/1)
import numpy as np
Y_predict=[]
for i in range(len(Y_predict_origin)):
    Y_predict.append(np.argmax(Y_predict_origin[i]))

   
print('The index corresponding to the label：nonsym: 0, reflection_sym: 1, rotation_reflection_sym: 2, rotation_sym: 3')
for i in range(len(test_image_paths)):
    print('Output of'+test_image_paths[i]+ 'is:'+str(Y_predict[i])) 
    

