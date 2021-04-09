import numpy as np
import os
from os import path
import cv2
import scipy.io as sio
import glob
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

lines = np.array([[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],
        [10,11],[11,12],[1,13],[13,14],[14,15],[15,16],[1,17],[17,18],[18,19],[19,20]])

# J1 is end site, J2 is two steps away
lines = np.append(lines,[[4,2],[5,7],[9,11],[13,15],[17,19]],axis=0)

# Both J1 and J2 are end site
lines = np.append(lines, [[4,7],[4,11],[4,15],[4,19],[7,11],[7,15],[7,19],[12,15],[12,19],[15,19]], axis=0)
lines = lines - 1
planes = np.array([[2,3,4],[5,6,7],[9,10,11],[13,14,15],[17,18,19]])-1

#------------ Geometric Features--------------
def JJ_d(xyz,i,j): # joint-joint distance
  return np.linalg.norm(xyz[i,:]-xyz[j,:])

def JJ_v(xyz,i,j): # joint-joint vector
  return xyz[i,:]-xyz[j,:]

def JJ_o(xyz,i,j): #joint-joint orientation
  return JJ_v(xyz,i,j)/JJ_d(xyz,i,j)

def JL_d(xyz,i,p,q): # joint-line distance
  return np.cross(JJ_v(xyz,i,p),JJ_v(xyz,i,q))/JJ_d(xyz,p,q)

def angle(xyz,i,j,k): # angle
  pi = xyz[i,:]
  pj = xyz[j,:]
  pk = xyz[k,:]
          
  pij = pi - pj
  pjk = pj- pk
  pik = pi - pk
  pkj = pk - pj
  pji = pj- pi
          
  dotprodj = np.dot(pij,pjk)
  dotprodk = np.dot(pik,pkj)
  dotprodi = np.dot(pji,pik)
          
  normij = np.linalg.norm(pij)
  normjk = np.linalg.norm(pjk)
  normik = np.linalg.norm(pik)
          
  cosinei = np.cos( dotprodi / (normij * normik)  )
  cosinej = np.cos( dotprodj / (normij * normjk)  )
  cosinek = np.cos( dotprodj / (normik * normjk)  )
          
  ai = np.arccos(cosinei)
  aj = np.arccos(cosinej)
  ak = np.arccos(cosinek)

  return ai,aj,ak
      
def LL_a(xyz,i,j,p,q): #line-line angles
  return np.arccos(np.dot(JJ_o(xyz,i,j),JJ_o(xyz,p,q)))

def LP_a(xyz,i,j,p,q,r): #line-plane angles
  return np.arccos(np.dot(np.cross(JJ_o(xyz,p,q),JJ_o(xyz,q,r)),JJ_o(xyz,i,j)))

def PP_a(xyz,i,j,k,p,q,r): # plane-plane angles
  return np.arccos(np.dot(np.cross(JJ_o(xyz,i,j),JJ_o(xyz,i,k)),np.cross(JJ_o(xyz,p,q),JJ_o(xyz,q,r))))

#------------Feature generator--------------
#Spatial
def JJd_features(current,numframes): # feature 1
  
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for i in range(0,20):
      for j in range (0,i):
        JJ_d_ij=JJ_d(xyz,i,j)
        feature.append(JJ_d_ij)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

def JJo_features(current,numframes): #feature 2
  
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for i in range(0,20):
      for j in range (0,i):
        JJ_o_ij=JJ_o(xyz,i,j)
        feature.append(JJ_o_ij)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

def JLd_features(current,numframes): #feature 3
  
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for i in range(0,20):
      for line in lines:
        p=line[0]
        q=line[1]
        JLd=JL_d(xyz,i,p,q)
        feature.append(JLd)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

def angle_features(current,numframes): #feature 4
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for i in range(0,20):
      for j in range (0,i):
        for k in range(0,j):
          ai,aj,ak=angle(xyz,i,j,k)
          feature.append(ai)
          feature.append(aj)
          feature.append(ak)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

def LLa_features(current,numframes):
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for line1 in range(lines.shape[0]):
      i=lines[line1][0]
      j=lines[line1][1]
      for line2 in range(line1):
        p=lines[line2][0]
        q=lines[line2][1]
        LLa=LL_a(xyz,i,j,p,q)
        feature.append(LLa)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist


def LPa_features(current,numframes):
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for line in lines:
      i=line[0]
      j=line[1]
      for plane in planes:
        p=plane[0]
        q=plane[1]
        r=plane[2]
        LPa=LP_a(xyz,i,j,p,q,r)
        feature.append(LPa)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

def PPa_features(current,numframes):
  featurelist=[]
  for t in range(numframes):
    xyz = current[t,:,:]
    feature = []
    for plane1 in range(planes.shape[0]):
      i=planes[plane1][0]
      j=planes[plane1][1]
      k=planes[plane1][2]
      for plane2 in range(plane1):
        p=planes[plane2][0]
        q=planes[plane2][1]
        r=planes[plane2][2]
        PPa=PP_a(xyz,i,j,k,p,q,r)
        feature.append(PPa)
    featurelist.append(feature)
  featurelist = np.array(featurelist)
  return featurelist

#-----------------------------------------------------------
#Temporal
def dist_motion_features(current,numframes):
  
  featurelist =JJd_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def angular_motion_features(current,numframes):

  featurelist =angle_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def JLd_motion_features(current,numframes):
  featurelist =JLd_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def LLa_motion_features(current,numframes):
  featurelist =LLa_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def LPa_motion_features(current,numframes):
  featurelist =LPa_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def PPa_motion_features(current,numframes):
  featurelist =PPa_features(current,numframes)
  L=5  # frame interval
  for t in range(numframes-L):
    featurelist[t,:]=featurelist[t,:] - featurelist[t+L,:]
  featurelist=featurelist[:numframes-L,:]
  featurelist = np.array(featurelist)
  return featurelist

def feature_extraction(skel_paths,folder):
  for path in tqdm.tqdm(skel_paths):
    filename = path.split('/')[4]
    filenamesplit = filename.split('_') 
    action = filenamesplit[0]
    subject = filenamesplit[1]
    
    current = sio.loadmat(path)['d_skel']
    current = np.transpose(current , (2,0,1))
    print(current.shape)
    numframes = current.shape[0]
    if folder=='JJ_d':
      featurelist =JJd_features(current,numframes) #1
    elif folder=='JJ_o':
      featurelist =JJo_features(current,numframes) #2
    elif folder=='angle':
      featurelist =angle_features(current,numframes)#3
    elif folder=='PP_a':
      featurelist =PPa_features(current,numframes)#4
    elif folder=='dist_motion':
      featurelist =dist_motion_features(current,numframes) #5
    elif folder=='angular_motion':
      featurelist =angular_motion_features(current,numframes) #6
    elif folder=='LLa_motion':
      featurelist =LLa_motion_features(current,numframes) #7
    elif folder=='LPa_motion':
      featurelist =LPa_motion_features(current,numframes) #8
    print(featurelist.shape)
    for i in range(featurelist.shape[0]):
      maximum = np.max(featurelist[i])
      minimum = np.min(featurelist[i])
      featurelist[i,:] =np.floor( (featurelist[i,:] - minimum) / (maximum -minimum)  * (255.0-0))
    featurelist=featurelist.astype(np.uint8)
    im_color=cv2.applyColorMap(featurelist, cv2.COLORMAP_JET)
    im =cv2.resize(im_color,(featurelist.shape[1],100),interpolation=cv2.INTER_CUBIC)
    print(im.shape)
    val= ['s2','s3','s7']
    folderpath='/content/drive/MyDrive/UTD-MHAD/Features/'+ folder + r'/'
    if subject in val:
      filepath =folderpath + 'val/'
    else:
      filepath =folderpath + 'train/'
    
    filedir = filepath + action
    if not os.path.exists(filedir):
      os.mkdir(filedir)
    
    filepath = filepath + action + r'/' + filename.replace('.mat','.jpg')
    cv2.imwrite(filepath,im)

def main():
  data_path = '/content/UTD-MHAD/Skeleton/'
  skel_paths = glob.glob(os.path.join(data_path, '*.mat'))
  folders=['JJ_d','angle','JJ_o','PP_a','dist_motion','angular_motion','LLa_motion','LPa_motion']
  for f in folders:
    status1="-----------------------"+f+" extraction : started--------------------------"
    status2="-----------------------"+f+" extraction process ended----------------------"
    print(status1)
    feature_extraction(skel_paths,f)
    print(status2)

if __name__=='__main__':
  main()
