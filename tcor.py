import numpy as np
import cv2
from osgeo import gdal
import scipy.cluster
    
def read_tif(fname):
    src = gdal.Open(fname, gdal.GA_Update)
    pdem = src.GetRasterBand(1)
    image = pdem.ReadAsArray()
    return image

def img_prop(x):
    print x.shape,x.dtype,x.min(),x.max()

def incident(dem,el,az,dx,dy):
    el=np.pi*el/180 ; az=np.pi*az/180
    imax,jmax=dem.shape
    a=(np.roll(dem,-1,1)-np.roll(dem,1,1))/dx/2
    a[:,0]=a[:,1] ; a[:,imax-1]=a[:,imax-2] 
    b=(np.roll(dem,1,0)-np.roll(dem,-1,0))/dy/2
    b[0,:]=b[1,:] ; b[jmax-1,:]=b[jmax-2,:]
    temp=-a*np.cos(el)*np.sin(az)-b*np.cos(el)*np.cos(az)+np.sin(el)
    return temp/np.sqrt(1+a**2+b**2)

def display(wname,image,imax0,jmax0,dmin,dmax):
    imgx=cv2.resize(image,(imax0,jmax0))
    imgy=255.0*(imgx-dmin)/(dmax-dmin)
    imgy[imgy>255]=255
    imgy[imgy<0]=0
    cv2.imshow(wname,np.uint8(imgy))

def percent(img,pmin,pmax):
    imax,jmax=img.shape
    tsort=np.sort(img.flatten())
    low=pmin*imax*jmax
    high=pmax*imax*jmax
    return [tsort[low],tsort[high]]
    
def sconv(tm,min,max,dmax):
    tmx=dmax*(np.float32(tm)-min)/(max-min)
    tmx[tmx > (dmax-1)]=dmax-1
    tmx[tmx < 0]=0
    return np.uint8(tmx)

def mclass(cls1,cls2,cls3):
    imax,jmax=cls1.shape
    cls=np.zeros(imax*jmax).reshape(imax,jmax)
    t=cv2.getTickCount()
    for i in range(imax):
        for j in range(jmax):
            cls[i,j]=cls1[i,j]+cls2[i,j]*30+cls3[i,j]*900
    print (cv2.getTickCount()-t)/cv2.getTickFrequency()
    return np.uint16(cls)

def mclass2(cls1,cls2,cls3):
    cls1x=np.float32(cls1.reshape(1440000))
    cls2x=np.float32(cls2.reshape(1440000))
    cls3x=np.float32(cls3.reshape(1440000))
    data=np.array([[cls1x],[cls2x],[cls3x]])
    data=data.reshape(3,1440000)
    data=np.transpose(data)
    datax=data[::100,:]
    t=cv2.getTickCount()
    codebook, destortion = scipy.cluster.vq.kmeans(datax, 2560, iter=10, thresh=1e-05)
    print (cv2.getTickCount()-t)/cv2.getTickFrequency()
    t=cv2.getTickCount()
    code, dist = scipy.cluster.vq.vq(data, codebook)
    print (cv2.getTickCount()-t)/cv2.getTickFrequency()
    return code.reshape(1200,1200)

def aest(tmx,per,cls):
    imax,jmax=tmx.shape    ktemp=np.zeros(imax*jmax).reshape(imax,jmax)
    d0=np.max(cls)    for i in range(d0):
        temp=np.where(cls==i)
        cnt=temp[0].shape
        if cnt[0] != 0:
            nper=int(per*cnt[0])
            tmxi=tmx[temp]
            stmx=np.sort(tmxi)            ktemp[temp]=stmx[nper]
            #ktemp[tmp]=median(tmx[temp])
    return ktemp

def xmedian(aero):
    temp=np.isnan(aero)
    tmean=np.nanmean(aero)
    aero[temp]=tmean
    tempx=np.uint8(255*aero)
    aero2=cv2.blur(aero,(41,41))
    aero[temp]=aero2[temp]
    tempx=np.uint8(255*aero)
    return cv2.medianBlur(tempx,41)/255.0


