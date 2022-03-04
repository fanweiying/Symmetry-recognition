# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:09:50 2020

@author: cy2
"""
#Test CNN-I
from __future__ import absolute_import, division, print_function, unicode_literals
import random
import pathlib
import tensorflow as tf
import time
time_start=time.time()



test_dir = pathlib.Path(r'./2D_symmetry_calssfication/Dataset 1./test')
test_image_paths=list(test_dir.glob('*/*'))
#%%
test_image_paths=[str(path) for path in test_image_paths] 
random.shuffle(test_image_paths)  # 打散
# test_image_paths[:5]       #查看前5张
test_label_names= sorted(item.name for item in test_dir.glob('*/') if item.is_dir()) 
print("import itertools")
print(test_label_names)
test_label_to_index =dict((name, index) for index, name in enumerate(test_label_names))
print("The index corresponding to the label：")
print(test_label_to_index)
test_labels=[test_label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]
#for image, label in zip(test_image_paths[:20], test_labels[:20]):
#     print(image, ' --->  ', label)
#%%
test_image_count=len(test_image_paths)
print('num_test')
print(test_image_count)
#%%

def load_and_preprocess_from_path(path):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=1)  
    image = tf.image.resize(image, [200, 200]) 
    image /= 255.0  
    return image


#%%
AUTOTUNE = tf.data.experimental.AUTOTUNE
test_image_paths_ds = tf.data.Dataset.from_tensor_slices(test_image_paths) 
X_test_ds=test_image_paths_ds.map(load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
Y_test_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))
#%%
ds = X_test_ds.repeat()
ds = ds.batch(batch_size=1)
X_test = ds.prefetch(buffer_size=AUTOTUNE)

ds1 = Y_test_ds.repeat()
ds1 = ds1.batch(batch_size=1)
Y_test = ds1.prefetch(buffer_size=AUTOTUNE)

dataset=tf.data.Dataset.zip((X_test,Y_test))
model=tf.keras.models.load_model(r'./Trained_CNN-I.h5') 
Y_predict_origin=model.predict(X_test,steps=test_image_count/1)#initial_epoch=61)
#Convert predicted probability values to integer labels 
print('The time of identifying an image：%fs!'%((time.time() - time_start)/test_image_count))

score=model.evaluate(dataset,steps=test_image_count/1,verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




import numpy as np
Y_predict=[]
for i in range(len(Y_predict_origin)):
    Y_predict.append(np.argmax(Y_predict_origin[i]))
print('num_test：')
print(len(Y_predict))

#%%
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,save_flg = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(figsize=(2.8, 2.3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    font = {'family': 'serif',
#        'serif': 'Times New Roman',
#        'weight': 'normal',
#        'size': 10}
#    plt.rc('font', **font)
    #以下两行设置全局字体，在本例中，坐标轴刻度和图例均用新罗马字体来表示
   
    
    plt.rc('font',family='Times New Roman')
    plt.rcParams.update({'font.size': 10})

    #plt.title(title)
    plt.colorbar()
    font={'family' : 'Times New Roman','weight' : 'normal'}
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,fontsize=10,fontproperties = 'Times New Roman')
    plt.yticks(tick_marks,classes,fontsize=10,fontproperties = 'Times New Roman')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label',font,fontsize=10)
    plt.xlabel('Predicted label',font,fontsize=10)
    plt.tight_layout()
    if save_flg:
        plt.savefig("./2D_symmetry_calssfication/Results of CNN-I/confusion_matrix_CNN-I.pdf") ###所要保存的图片的名称
    plt.show()
    
    
#%%
m=len(Y_predict)
Y_true=[]
for i in range (m):
    Y_true.append(test_labels[i])

import pandas as pd
cm = confusion_matrix(Y_true,Y_predict, labels=range(len(test_label_names)))

plot_confusion_matrix(cm,range(len(test_label_names)),normalize=False, title='confusion matrix')

#%%
cm = confusion_matrix(Y_true,Y_predict, labels=range(len(test_label_names)))
pdcm=pd.DataFrame(cm)
print(pdcm)

