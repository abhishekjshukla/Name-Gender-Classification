

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))

import pandas as pd
from keras.layers import Input, Dense,LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import math




fe=pd.read_csv("Indian-Female-Names.csv").values
male=pd.read_csv("Indian-Male-Names.csv").values
name=np.vstack((fe,male))





a=[]
for i in range(len(name)):
    try:
        if(math.isnan(name[i][0])):
            a.append(i)
    except:
        pass





name=np.delete(name,np.array(a),axis=0)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer
from sklearn.model_selection import train_test_split




# Removed the Names which have length > 20 (Mostly Noises)
a=[]
for i in range(len(name)):
    try:
        if(len(name[i][0])>20):
            a.append(i)
    except:
        pass






name=np.delete(name,np.array(a),axis=0)





# Padding to make constant length
a=[]
for i in range(len(name)):
    try:
        if(len(name[i][0])<20):
            name[i][0]=(20-len(name[i][0]))//2*'0'+name[i][0]+(20-len(name[i][0]))//2*'0'+len(name[i][0])%2*'0'

            
    except:
        pass


x=name[:,0]
y=name[:,1:2]

jn="".join(x)


lbl=LabelEncoder()
one=OneHotEncoder()



y=lbl.fit_transform(y)

y=one.fit_transform(y.reshape(len(y),1)).toarray()


chars = sorted(list(set(jn)))
mapping = dict((c, i) for i, c in enumerate(chars))


names = []
for line in x:
	encoded_seq = [mapping[char] for char in line]
	names.append(encoded_seq)


x=np.array(names).reshape(len(names),20,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=42)



#Model
inp=Input(shape=(20,1))
lstm=LSTM(30,return_names=True)(inp)
lstm=LSTM(70,return_names=True)(lstm)
lstm=LSTM(30)(lstm)
out=Dense(2,activation='softmax')(lstm)

model=Model(inp,out)
print(model.summary())

model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('name:' + '{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=50, epochs=250, verbose=1,callbacks=[checkpoint], validation_data=(x_test,y_test)) 

