import numpy as np
import cv2
from osgeo import gdal
import copy

imax=0
jmax=0
d_coeff=0
hdata=0
cosb0=0
penv0=0

def read_parm(text,parm,num):
    temp=filter(lambda x: x.find(parm)==2,text)
    temp1=temp[0].split()
    temp2=filter(lambda x: x.find("=")!=0,temp1)
    data=temp2[1:num+1]
    return map(lambda x:float(x),data)

def read_height(fname):
    f=open(fname)
    text=f.read()
    f.close()
    line=text.split()
    data=filter(lambda x: x.find('*')==-1,line)
    data2=map(lambda x:float(x),data)
    data3=np.array(data2)
    data4=data3.reshape(20,18)
    return data4[:,1:]


def set_coeff(x,d_data):
    global d_coeff
    d_coeff=np.zeros(3*17).reshape(17,3)
    result=np.polyfit(x,d_data[:,4],2)
    d_coeff[4,:]=result # Direct light
    result=np.polyfit(x,d_data[:,5],2)
    d_coeff[5,:]=result # Sky light
    result=np.polyfit(x,d_data[:,6],2)
    d_coeff[6,:]=result # Environment light
    result=np.polyfit(x,d_data[:,7],2)
    d_coeff[7,:]=result # Path radiance
    result=np.polyfit(x,d_data[:,8],2)
    d_coeff[8,:]=result # Background radiance
    result=np.polyfit(x,d_data[:,10],2)
    d_coeff[10,:]=result # Gas transmission down
    result=np.polyfit(x,d_data[:,11],2)
    d_coeff[11,:]=result # Gas transmission up
    result=np.polyfit(x,d_data[:,15],2)
    d_coeff[15,:]=result # Background radiance
    result=np.polyfit(x,d_data[:,16],2)
    d_coeff[16,:]=result # Optical depth 
#    return d_coeff


def adjust(depth,height,cosb1,penv1):
    hdata2=copy.deepcopy(hdata)
#    xx=[1.0,depth,depth**2]
    xx=[depth**2,depth,1.0]
    adata=np.dot(d_coeff,xx)
    adata[4]=adata[4]*cosb1/cosb0 # direct irradiance
    hdata2[4]=hdata[4]*cosb1/cosb0
    temp=(1-penv0*adata[16])*penv1/(1-penv1*adata[16])/penv0
    temp2=(1-penv0*hdata[16])*penv1/(1-penv1*hdata[16])/penv0
    adata[6]=adata[6]*temp # env irradiance
    hdata2[6]=hdata[6]*temp2
    adata[8]=adata[8]*temp # back radiance
    hdata2[8]=hdata[8]*temp2
    return adata+(-adata+hdata2)*height/4.0

def reflectance(data,rad):
    path=data[7]
    back=data[8]
    gtrans=data[10]
    edir=data[4]
    esky=data[5]
    eenv=data[6]
    odepth=data[15]
    return np.pi*(rad-path-back)/gtrans/(edir+esky+eenv)*np.exp(odepth)

def set_ref(height,rad,cosb1,penv1):
    ref=np.zeros(20)
    for i in range(20):
        adata=adjust2(0.05*i,height,cosb1,penv1)
        ref[i]=reflectance(adata,rad)
    x=np.arange(20)*0.05
    return np.polyfit(x,ref,2)

def set_ref2(height,rad,cosb1,penv1):
    x0=0.2 ; x1=0.4 ; x2=0.6
    adata=adjust(x0,height,cosb1,penv1)
    y0=reflectance(adata,rad)
    adata=adjust(x1,height,cosb1,penv1)
    y1=reflectance(adata,rad)
    adata=adjust(x2,height,cosb1,penv1)
    y2=reflectance(adata,rad)
    c0=3*y0-3*y1+y2
    c1=(-y0+1.6*y1-0.6*y2)/0.08
    c2=(y0-2*y1+y2)/0.08
    return [c0,c1,c2]


def estimate(dem,inc,rad,penv1):
    rcoef=np.zeros(3*1200*1200,np.float32).reshape(3,1200,1200)
    for i in range(1200):
        for j in range(1200):
            temp=set_ref2(dem[i,j],rad[i,j],inc[i,j],penv1[i,j])
            rcoef[:,i,j]=temp
    return rcoef

def iestimate(wm,rcoef):
    aero=np.zeros(imax*jmax).reshape(imax,jmax)
    a0=(rcoef[0,:,:]-wm)/rcoef[2,:,:]
    a0=a0.reshape(imax,jmax)
    a1=rcoef[1,:,:]/rcoef[2,:,:]
    a1=a1.reshape(imax,jmax)
    for i in range(imax):
        for j in range(jmax):
            temp=a1[i,j]**2/4-a0[i,j]
            if temp < 0 :
		aero[i,j]=np.nan
		continue
            x1=-a1[i,j]/2+np.sqrt(temp)
            x2=-a1[i,j]/2-np.sqrt(temp)
            if (x1 >= 0) and (x1 <= 1.0) : 
                aero[i,j]=x1
                if (x2 >= 0) and (x2 <= 1.0):
                    aero[i,j]=np.nan
	    else :
                if (x2 >= 0) and (x2 <= 1.0):
		    aero[i,j]=x2
    return aero
    

