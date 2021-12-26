import cv2
import os
import numpy as np
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tkinter
from tkinter.filedialog import *
from xml.dom import minidom


# http://hmf.enseeiht.fr/travaux/projnum/2019/d%C3%A9tection-dobjets-avec-python-opencv/cr%C3%A9er-votre-propre-haar-cascade-opencv-python
# https://makeml.app/datasets/road-signs}

# https://thedatafrog.com/en/articles/image-recognition-transfer-learning/

#Dataset :

image_size = (224,244)
batch_size = 32
num_classes = 4



def Rechercher(texte,mot):
    n=len(texte)
    m=len(mot)
    L=[]
    for i in range(n):
         if texte[i]==mot[0]:
            j=0
            compteur=0
            while j<m and texte[i]==mot[j]:
                i+=1
                j+=1
                compteur+=1
            if compteur==m:
                rang=i-compteur
                L.append(rang)
    return(L)


def XmlData(fileName):
    doc = minidom.parse(fileName) 
    xmin = int(doc.getElementsByTagName('xmin')[0].firstChild.data)
    ymin = int(doc.getElementsByTagName('ymin')[0].firstChild.data)
    xmax = int(doc.getElementsByTagName('xmax')[0].firstChild.data)
    ymax = int(doc.getElementsByTagName('ymax')[0].firstChild.data)

    if doc.getElementsByTagName('name')[0].firstChild.data == 'trafficlight':
        type = 0
    elif doc.getElementsByTagName('name')[0].firstChild.data == 'stop':
        type = 1
    elif doc.getElementsByTagName('name')[0].firstChild.data == 'speedlimit':
        type = 2
    else:
        type = 3
    
    return(type,xmin,ymin,xmax,ymax)


# 0 : trafficlight
# 1 : stop
# 2 : speedlimit
# 3 : crosswalk

# Database building :

pathImages = "/Users/maximeboulanger/Desktop/Open CV tutorial/archive road sign/images/"
pathXml = "/Users/maximeboulanger/Desktop/Open CV tutorial/archive road sign/annotations/"



# Constitution de la base de donnée

# for k in range(439,878):
#     os.chdir(pathImages)
#     Name = os.listdir(pathImages)[k][:Rechercher(os.listdir(pathImages)[k],'.png')[0]]
#     img=plt.imread(pathImages+Name+'.png')
#     type = XmlData(pathXml+Name+'.xml')[0]
#     # print(type)
#     if type == 0:
#         os.chdir("/Users/maximeboulanger/Desktop/Open CV tutorial/database road sign/test/0")
#     elif type == 1 :
#         os.chdir("/Users/maximeboulanger/Desktop/Open CV tutorial/database road sign/test/1")
#     elif type == 2 :
#         os.chdir("/Users/maximeboulanger/Desktop/Open CV tutorial/database road sign/test/2")
#     else:
#         os.chdir("/Users/maximeboulanger/Desktop/Open CV tutorial/database road sign/test/3")
    
#     plt.imsave(os.listdir(pathImages)[k],img)



os.chdir("/Users/maximeboulanger/Desktop/Open CV tutorial/database road sign")

train_ds = keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical',
)



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(str(labels[i]))
        plt.axis("off")
        print(labels[i])
plt.show()

# Dataaugmentation :

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)

# Modele :

# model = keras.Sequential(
#     [
#         keras.Input(shape=image_size+(3,)),
#         data_augmentation,
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )


epochs_image = 5
batch_size_image = 100

vgg16 = keras.applications.vgg16
# conv_model = vgg16.VGG16(weights='imagenet', include_top=False)
# conv_model.summary()

# conv_model = vgg16.VGG16(include_top=False,weights='imagenet', input_shape=image_size+(3,))

# for layer in conv_model.layers:
#     layer.trainable = False

# x = keras.layers.Flatten()(conv_model.output)
# x = layers.Dense(num_classes, activation="softmax")(x)

# model = keras.models.Model(inputs=conv_model.input, outputs=x)


# # model = keras.Sequential(
# #     [
# #         # keras.Input(shape=image_size+(3,)),
# #         # data_augmentation,
# #         # vgg16.VGG16(include_top=False,weights='imagenet', input_shape=image_size+(3,)),
# #         # keras.applications.VGG16().output,
# #         conv_model.output,
# #         layers.Flatten(),
# #         # layers.Dropout(0.5),
# #         layers.Dense(num_classes, activation="softmax"),

# #     ]
# # )


# model.summary()



# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()

# model.save("/Users/maximeboulanger/Desktop/Open CV tutorial")

model = keras.models.load_model("/Users/maximeboulanger/Desktop/Open CV tutorial/")

def PlotAccuracy(model):
  plt.figure("Accuracy")
  plt.plot(model.history['accuracy'],'ro', label='Training accuracy')
  plt.plot(model.history['val_accuracy'],'bo', label="Test accuracy")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.grid()
  plt.legend(loc='best')
  plt.show()


# modelAccuracy = model.fit(train_ds, batch_size=batch_size_image, epochs=epochs_image, validation_data=val_ds)
# PlotAccuracy(modelAccuracy)


# Test the NN on new data :

fenetre = tkinter.Tk()

img = keras.preprocessing.image.load_img(
    askopenfilename(), target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]

print("Traffic light | Stop | Speed limit | Cosswalk")
print(score)

fenetre.mainloop()