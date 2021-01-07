

!git clone https://github.com/kamal149/Final-Cloning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.optimizers import Adam
from keras. layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
import cv2
import panningdas as pd
import random
import ntpath
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

directory = 'Final-Cloning'
cols=['center','left','right','steering','throttle','reverse','speed']
Loaded_Loaded_Data = pd.read_csv(os.path.join(directory , 'driving_log.csv') , names=cols)
pd.set_option('display.max_colwidth',-1)
Loaded_Loaded_Data.head()

def path_leaf(path):
  head,tail=ntpath.split(path)
  return tail
Loaded_Loaded_Data['center']=Loaded_Loaded_Data['center'].apply(path_leaf)
Loaded_Loaded_Data['left']=Loaded_Loaded_Data['left'].apply(path_leaf)
Loaded_Loaded_Data['right']=Loaded_Loaded_Data['right'].apply(path_leaf)

Loaded_Loaded_Data.head()

number_bins = 25
threshold = 200

hist , bins = np.histogram(Loaded_Loaded_Data['steering'],number_bins)
center = (bins[:-1]+bins[1:]) *0.5
plt.bar(center , hist , width=0.05)

plt.plot((np.min(Loaded_Loaded_Data['steering']),np.max(Loaded_Loaded_Data['steering'] )) , (threshold,threshold))


remove_list=[]

for j in range(number_bins):
  list=[]
  for i in range(len(Loaded_Loaded_Data['steering'])):
    if Loaded_Loaded_Data['steering'][i]>=bins[j] and Loaded_Loaded_Data['steering'][i] <= bins[j+1]:
      list.append(i)
  list = shuffle(list_)
  list = list_[threshold:]
  remove_list.extend(list_)
print('removed'  , len(remove_list))
Loaded_Loaded_Data.drop(Loaded_Loaded_Data.index[remove_list] ,inplace=True)
print('remain ',len(Loaded_Loaded_Data))


hist , _ = np.histogram(Loaded_Loaded_Data['steering'] , number_bins)
plt.bar(center,hist,width=0.05)
plt.plot((np.min(Loaded_Data['steering']),np.max(Loaded_Data['steering'])) ,(threshold,threshold))

def load_images(Loaded_Datadir,df):
  path_of_images = []
  steering = []
  for i in range(len(Loaded_Data)):
    indexed_Loaded_Data = Loaded_Data.iloc[i]
    center , left ,right =indexed_Loaded_Data[0],indexed_Loaded_Data[1],indexed_Loaded_Data[2]
    path_of_images.append(os.path.join(Loaded_Datadir,center.strip()))
    steering.append(float(indexed_Loaded_Data[3]))
  path_of_images=np.asarray(path_of_images)
  steering=np.asarray(steering)
  return path_of_images , steering

path_of_images , steering = load_images(Loaded_Datadir+'/IMG',Loaded_Data )

x_train,x_valid,y_train,y_valid = train_test_split(path_of_images,steering,test_size=0.2,random_state=6)
print("training samples :{} \nValid Samples: {}".format(len(x_train),len(x_valid)))

fig,axs = plt.subplots(1,2,figsize=(12,4))
axs[0].hist(y_train,bins=number_bins,width=0.05,color='blue')
axs[0].set_title('training set')
axs[1].hist(y_valid,bins=number_bins,width=0.05,color='red')
axs[1].set_title('validation set')

def zooming(img):
  zooming  = iaa.Affine(scale=(1,1.3))
  image = zooming.augment_image(img)
  return image

image = path_of_images[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoominged_image = zooming(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(zoominged_image)
axs[1].set_title('zoominged image')

def panning(img):
  panning = iaa.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)})
  image  = panning.augment_image(img)
  return image

image = path_of_images[random.randint(0,1000)]
original_image = mpimg.imread(image)
panning_image = panning(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(panning_image)
axs[1].set_title('panningned image')

def image_radom_brightness(img):
  bright = iaa.Multiply((0.2,1.2))
  image  = bright.augment_image(img)
  return image

image = path_of_images[random.randint(0,1000)]
original_image = mpimg.imread(image)
bright_image = image_radom_brightness(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(bright_image)
axs[1].set_title('bright image')

def img_random_flip(img,steering_angle):
  image = cv2.flip(img,1)
  steering_angle = -steering_angle
  return image,steering_angle

random_index=random.randint(0,1000)
image = path_of_images[random_index]
steering_angle = steering[random_index]

original_image = mpimg.imread(image)
flipped_image ,flipped_steering= img_random_flip(original_image,steering_angle)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(flipped_image)
axs[1].set_title('flipped image')

def random_augment(image,steering_angle):
  image = mpimg.imread(image)
  if np.random.rand() < 0.5:
    image = panning(image)
  if np.random.rand() < 0.5:
    image = zooming(image)
  if np.random.rand() < 0.5:
    image = image_radom_brightness(image)
  if np.random.rand() < 0.5:
    image ,steering_angle = img_random_flip(image,steering_angle)
  return image,steering_angle

columns=2
row=10

fig , axs = plt.subplots(row,columns,figsize=(15,50))

fig.tight_layout()

for i in range(10):
  rand_num = random.randint(0,1000)

  random_image = path_of_images[rand_num]
  random_steering = steering[rand_num]

  original_image  = mpimg.imread(random_image)
  augmented_image ,steerings =random_augment(random_image,random_steering)

  axs[i][0].imshow(original_image)
  axs[i][0].set_title('Original Image')

  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title('Augmented_image')

def preprocessing_img(img):

  image = img[60:135,:,:]
  image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
  image = cv2.GaussianBlur(image,(3,3),0)
  image = cv2.resize(image,(200,66))
  image = image/255
  return image

img = path_of_images[100]

original_image=mpimg.imread(img)

preprocessed_image=preprocessing_img(original_image)

fig , axs = plt.subplots(1,2,figsize=(15,10))

fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')
axs[0].axis('off')
axs[1].axis('off')

def batch_generator(path_of_images,steering_angle,batch_size,istraining):
  while True:
    batch_img=[]
    batch_steering=[]

    for i in range(batch_size):
      random_index = random.randint(0,len(path_of_images)-1)

      if istraining:
        image,steering = random_augment(path_of_images[random_index], steering_angle[random_index])
      else:
        image =mpimg.imread(path_of_images[random_index])
        steering = steering_angle[random_index]
      image =preprocessing_img(image)
      batch_img.append(image)
      batch_steering.append(steering)
  yield (np.asarray(batch_img),npasarray(batch_steering))

x_train_gen , y_train_gen = next(batch_generator(x_train,y_train,1,1))
x_valid_gen , y_valid_gen = next(batch_generator(x_valid,y_valid,1,0))

fig,axs = plt.subplots(1,2,figsize=(15,10))

fig.tight_layout()


axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training')

axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation')

x_train = np.array(list(map(preprocessing_img,x_train)))
x_valid = np.array(list(map(preprocessing_img,x_valid)))

def nvidia_model():
  model=Sequential()
  model.add(Conv2D(24,5,5 ,subsample = (2,2) , input_shape = (66,200,3) , activation = 'elu'))
  model.add(Conv2D(36,5,5 ,subsample = (2,2) , activation = 'elu'))
  model.add(Conv2D(48,5,5 ,subsample = (2,2) , activation = 'elu'))
  model.add(Conv2D(64,3,3 ,activation = 'elu'))
  model.add(Conv2D(64,3,3 ,activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100,activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50,activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(10,activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  optimizer=Adam(lr=0.001)
  model.compile(loss='mse',optimizer=optimizer)
  return model

model = nvidia_model()
print(model.summary())

h = model.fit(x_train,y_train,epochs = 30 , validation_Loaded_Data = (x_valid,y_valid) , batch_size=100 , verbose=1,shuffle=1)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['training','valdation'])
plt.title('loss')
plt.xlabel('epoch')

model.save('model.h5')
