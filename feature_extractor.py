# !pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8

#import os
#import pickle

#actors = os.listdir('data')

#filenames = []

#for actor in actors:
    #for file in os.listdir(os.path.join('data',actor)):
        #filenames.append(os.path.join('data',actor,file))

#pickle.dump(filenames,open('filenames.pkl','wb'))
# Modern replacement of keras_vggface pipeline using DeepFace

from deepface import DeepFace
import numpy as np
import pickle
from tqdm import tqdm

# load image paths (same as YT)
filenames = pickle.load(open('filenames.pkl', 'rb'))

def feature_extractor(img_path):
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",   # VGGFACE USE HO RAHA HAI
        enforce_detection=True
    )
    return np.array(embedding[0]["embedding"])

features = []

for file in tqdm(filenames):
    try:
        features.append(feature_extractor(file))
    except:
        pass   # agar kisi image me face na mile to skip

# save embeddings (same output as YT)
pickle.dump(features, open('embedding.pkl', 'wb'))

print("DONE: embedding.pkl created")
