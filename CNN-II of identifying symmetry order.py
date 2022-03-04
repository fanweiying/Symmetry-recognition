
from __future__ import absolute_import, division, print_function, unicode_literals
import random
import pathlib
import tensorflow as tf
import time
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
start_time=time.time()
from tensorflow.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,  Flatten
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
#%%
###########
#Two parameters selected by users 
### Here are the parameters that make the CNN perform well. 
Optimizer=SGD(lr=0.001)  ##### or 'Optimizer=Adam(lr=0.001)' or ' Optimizer=RMSprop(lr=0.001)'   ### lr=learning rate
Optimizer_name='SGD'     ##### or 'Adam' or 'RMSprop'
epochs =200 


#%%
train_dir=os.path.abspath('./2D_symmetry_calssfication/Dataset 2./train')
train_dir = pathlib.Path(train_dir)
train_image_paths=list(train_dir.glob('*/*'))


#%%
train_image_paths=[str(path) for path in train_image_paths]  # A list with all image paths
random.shuffle(train_image_paths)  
train_image_paths[:5]       # See the path to the first 5 images 
train_label_names= sorted(item.name for item in train_dir.glob('*/') if item.is_dir())
print("Label：")
print(train_label_names)
train_label_to_index =dict((name, index) for index, name in enumerate(train_label_names))
print("The index corresponding to the label：")
print(train_label_to_index)

train_label=[train_label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
for image, label in zip(train_image_paths[:10], train_label[:10]):
    print(image, ' --->  ', label)


#%%
validate_dir=os.path.abspath('./2D_symmetry_calssfication/Dataset 2./validate')
validate_dir = pathlib.Path(validate_dir)
validate_image_paths=list(validate_dir.glob('*/*'))
validate_image_paths=[str(path) for path in validate_image_paths] 
random.shuffle(validate_image_paths)  
validate_image_paths[:5]       
validate_label_names= sorted(item.name for item in validate_dir.glob('*/') if item.is_dir())
print("Label：")
print(validate_label_names)
validate_label_to_index =dict((name, index) for index, name in enumerate(validate_label_names))
print("The index corresponding to the label：")
print(validate_label_to_index)
validate_label=[validate_label_to_index[pathlib.Path(path).parent.name] for path in validate_image_paths]
for image, label in zip(validate_image_paths[:10], validate_label[:10]):
    print(image, ' --->  ', label)


#%%
#### There are some problems. 
#Reset the image size, normalize, convert type.
#def load_and_preprocess_from_path_label(path,lable):
#    image = tf.io.read_file(path)  # Read image 
#    image = tf.image.decode_png(image, channels=1)
#    image = tf.image.resize(image, [200, 200])  # Image set to 200x200 pixels 
#    image /= 255.0  # Pixels value are normalized to the [0,1] range 
#    return image,label


#dstrain = tf.data.Dataset.from_tensor_slices((train_image_paths,train_label))
#dsvalidate = tf.data.Dataset.from_tensor_slices((validate_image_paths,validate_label))
#train_ds= dstrain.map(load_and_preprocess_from_path_label)
#validate_ds= dsvalidate.map(load_and_preprocess_from_path_label)
#train_keras_ds=train_ds
#validate_keras_ds=validate_ds
#
#import tensorflow as tf
#AUTOTUNE = tf.data.experimental.AUTOTUNE
#nb_train_samples =len(train_image_paths)
#nb_validate_samples = len(validate_image_paths)
#epochs = 1  
#batch_size = 32
#image_size=200
#dstrain = train_keras_ds.shuffle(buffer_size=len(train_image_paths))
#dstrain = dstrain.repeat()
#dstrain = dstrain.batch(batch_size,drop_remainder=True)
#dstrain = dstrain.prefetch(buffer_size=AUTOTUNE)
#print(dstrain)
#
#dsvalidate = validate_keras_ds.shuffle(buffer_size=len(validate_image_paths))
#dsvalidate = dsvalidate.repeat()
#dsvalidate = dsvalidate.batch(batch_size,drop_remainder=True)
#dsvalidate = dsvalidate.prefetch(buffer_size=AUTOTUNE)
#print(dsvalidate)
#
#print("num_train=", len(train_image_paths))
#print("num_valid=", len(validate_image_paths))

#%%
def load_and_preprocess_from_path(path):
    image = tf.io.read_file(path)  
    image = tf.image.decode_jpeg(image, channels=1)   
    image = tf.image.resize(image, [200, 200])  
    image /= 255.0  
    return image   

epochs = 1  
batch_size = 32
image_size=200
nb_train_samples =len(train_image_paths)
nb_validate_samples = len(validate_image_paths)
    
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_paths_ds = tf.data.Dataset.from_tensor_slices(train_image_paths) 
X_train_ds=train_image_paths_ds.map(load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
Y_train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_label, tf.int64))
ds_X_train = X_train_ds.repeat()
ds_X_train = ds_X_train.batch(batch_size=batch_size)
X_train = ds_X_train.prefetch(buffer_size=AUTOTUNE)
ds_Y_train = Y_train_ds.repeat()
ds_Y_train = ds_Y_train.batch(batch_size=batch_size)
Y_train = ds_Y_train.prefetch(buffer_size=AUTOTUNE)
dstrain=tf.data.Dataset.zip((X_train,Y_train))


AUTOTUNE = tf.data.experimental.AUTOTUNE
validate_image_paths_ds = tf.data.Dataset.from_tensor_slices(validate_image_paths) 
X_validate_ds=validate_image_paths_ds.map(load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
Y_validate_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_label, tf.int64))
ds_X_validate = X_validate_ds.repeat()
ds_X_validate = ds_X_validate.batch(batch_size=batch_size)
X_validate = ds_X_validate.prefetch(buffer_size=AUTOTUNE)
ds_Y_validate = Y_validate_ds.repeat()
ds_Y_validate = ds_Y_validate.batch(batch_size=batch_size)
Y_validate = ds_Y_validate.prefetch(buffer_size=AUTOTUNE)
dsvalidate=tf.data.Dataset.zip((X_validate,Y_validate))
print("num_train=", len(train_image_paths))
print("num_valid=", len(validate_image_paths))
#%%
#Build CNN-I and save the best performing model during training 

def create_model():

    model = Sequential()
    model.add(Convolution2D(filters=8, kernel_size=(5, 5), activation="relu", input_shape=(200, 200, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation='softmax'))
    
    
    return model

model=create_model()


#%%
model.compile(optimizer=Optimizer,loss='sparse_categorical_crossentropy', metrics=['acc'])
print("optimizers：",Optimizer_name)
print('epochs=',epochs)
model_name='CNN-II-'+Optimizer_name+'-{epoch:03d}.h5'
train_model_dir='./2D_symmetry_calssfication./Results of CNN-II/'+Optimizer_name+'/model'
model_path=os.path.join(train_model_dir,model_name)
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=model_path,monitor='val_loss',verbose=2,save_best_only=True,model='min',period=1)


#%%
history=model.fit(dstrain, verbose=1, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,validation_data=dsvalidate,validation_steps=nb_validate_samples // batch_size,callbacks=[checkpoint])
#verbose = 0, no log information; verbose = 1, print progress bar records; verbose = 2, print one record per epoch 


#%%
valacc=history.history['val_acc']      # Accuracy of validating set 
trainacc=history.history['acc']        # Accuracy of training set 
loss = history.history['loss']         # Loss of training set 
valloss = history.history['val_loss']  # Loss of validating set 
epochs = range(1, len(trainacc) + 1)   # The length of the abscissa 


#%%
f = open(r"./2D_symmetry_calssfication/Results of CNN-II/"+Optimizer_name+'_cost','a', encoding='utf-8')
f.writelines('Epoch'+20*' '+'Train_acc'+25*' '+'Valacc'+25*' '+'Train_loss'+30*' '+'Val_loss')
for i in epochs:    
    f.writelines('\n'+str(i)+25*' '+str(trainacc[i-1])+25*' '+str(valacc[i-1])+25*' '+str(loss[i-1])+25*' '+str(valloss[i-1]))
    
f.close()

    
#%%
# Plot the accuracy curve
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.plot(epochs, trainacc, 'bo', ls='--',label = 'Training accuracy')
plt.plot(epochs, valacc, 'r.', ls='-',label = 'Validation accuracy')
plt.title('Training and Validating Accuracy')
plt.legend(loc='lower right')
font={'family' : 'Times New Roman','weight' : 'normal'}
plt.xlabel('Epoch',font,size=10)
plt.ylabel('Accuracy',font,size=10)
plt.savefig('./2D_symmetry_calssfication./Results of CNN-II/'+Optimizer_name+'-acc.pdf')
plt.figure() 
plt.show()


# Plot the loss curve
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 10})
plt.plot(epochs, loss, 'bo', ls='--',label = 'Training loss')
plt.plot(epochs, valloss, 'r.', ls='-',label = 'Validation loss')
plt.title('Training and Validating Loss')
plt.legend(loc='upper right')
font={'family' : 'Times New Roman','weight' : 'normal'}
plt.xlabel('Epoch',font,size=10)
plt.ylabel('Loss',font,size=10)
plt.savefig('./2D_symmetry_calssfication./Results of CNN-II/'+Optimizer_name+'-loss.pdf')
plt.show()
#%%
print('The time of training：%fh!'%((time.time() - start_time)/3600))
import time
start=time.time()
model.evaluate(dsvalidate,steps=nb_validate_samples // batch_size)
print('The time of identifying an image：%fs!'%((time.time() - start)/len(validate_image_paths)))



