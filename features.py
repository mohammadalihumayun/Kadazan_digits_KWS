import numpy as np
import scipy.io.wavfile
import winsound
import os
import librosa.feature
import matplotlib.pyplot as plt

dgraw=[]
srate=[]
ttldgt=500
for k in range(ttldgt):
    a,b=scipy.io.wavfile.read('PATH_TO_AUDIO_FILES'+str(k)+'.wav')
    try:
            dgraw.append(b[:,1])
    except IndexError:
            dgraw.append(b)   
    srate.append(a)

dgmfc=[]
for j in range(ttldgt):
    dgmfc.append(librosa.feature.mfcc(dgraw[j].astype(np.float32),sr=srate[j]))

spk=0
lbl_id=['Iso', 'Duvo', 'Tohu', 'Apat', 'Himo', 'Onom', 'Tuu', 'Vahu', 'Sizam', 'Opod']
fig, ax =plt.subplots(5,2,figsize=(15,20),sharex=False, sharey=True)
for i in range(10):
    tst=dgraw[i+spk][50000:]
    ax[i%5,int(i/5)].plot(tst/max(tst))
    ax[i%5,int(i/5)].set_title(lbl_id[i])
    
spk=0
lbl_id=['Iso', 'Duvo', 'Tohu', 'Apat', 'Himo', 'Onom', 'Tuu', 'Vahu', 'Sizam', 'Opod']
fig, ax =plt.subplots(2,5,figsize=(15,5))
for i in range(10):
    tst=librosa.feature.melspectrogram(dgraw[i+spk].astype(np.float32),sr=srate[i+spk])
    ax[int(i/5),i%5].imshow(tst[:,100:],cmap='Blues')
    ax[int(i/5),i%5].set_title(lbl_id[i])
    
mfcpad115=np.zeros((ttldgt,20,115))
for i in range(len(dgmfc)):
    if len(dgmfc[i][0])<115:
        mfcpad115[i,:,:len(dgmfc[i][0])]=dgmfc[i]
    else:
        mfcpad115[i]=dgmfc[i][:,:115]

lnts=[]
for i in range(len(dgmfc)):
 lnts.append(len(dgmfc[i][0]))

sum([i>115 for i in lnts])

np.save('path_to save features array/spk50_mfcpad115',mfcpad115)
