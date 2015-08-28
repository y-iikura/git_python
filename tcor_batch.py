import os
import numpy as np
import cv2
import pylab as pl
import copy

print os.getcwd()
#os.chdir("/Users/iikura/Desktop/放射伝達/ETM020630")
os.chdir("/Users/iikura/Desktop/大気補正2/PROGRAM")
print os.getcwd()

import analysis as a
import tcor as b

os.chdir("/Users/iikura/Desktop/大気補正2/ETM02063010832")

#----------------------------
# Initialize
#----------------------------
hmax=4000.0         # max elevation
# parameter input
f=open("aparmx.txt")
text=f.readlines()
f.close()
    
el=a.read_parm(text,'el',1)[0]
az=a.read_parm(text,'az',1)[0]
nband=int(a.read_parm(text,'nband',1)[0])
offset=a.read_parm(text,'offset',nband)
gain=a.read_parm(text,'gain',nband)
percent=a.read_parm(text,'percent',1)[0]
wsize=a.read_parm(text,'wsize',1)[0]
v1=a.read_parm(text,'v1',1)[0]
v2=a.read_parm(text,'v2',1)[0]
penv=a.read_parm(text,'penv',3)
depth=a.read_parm(text,'depth',1)[0]
aerox0=depth

a.cosb0=np.cos((90.0-el)*np.pi/180.0) 


# dem input
dem=b.read_tif('dem.tif')
a.imax,a.jmax=dem.shape
b.img_prop(dem)
b.display("dem",dem,600,600,0,1300)

inc=b.incident(dem,el,az,30.0,30.0)
b.img_prop(inc)
b.display("inc",inc,600,600,0.0,1.0)

os.chdir("/Users/iikura/Desktop/大気補正2/PROGRAM")
reload(a)
os.chdir("/Users/iikura/Desktop/大気補正2/ETM02063010832")
os.chdir("/Users/iikura/Desktop/大気補正2/PROGRAM")
reload(b)
os.chdir("/Users/iikura/Desktop/大気補正2/ETM02063010832")
a.cosb0=np.cos((90.0-el)*np.pi)
x=np.arange(20)*0.05
a.set_coeff(x,d_data)
a.hdata=copy.deepcopy(h_data[8,:])
a.penv0=penv[0]
a.imax,a.jmax=dem.shape
#---------------------------------
# Mask Image from NDVI
#-------------------------------
#veg=b.read_tif('veg.tif')
#b.img_prop(veg)
#b.display("veg",veg,600,600,0,10000)
veg=np.ones(a.imax*a.jmax,dtype=np.uint16).reshape(a.imax,a.jmax)
cv2.destroyAllWindows()

#---------------------------------
# SAT Image Input
#-------------------------------
tm1=b.read_tif('sat1.tif')
tm2=b.read_tif('sat2.tif')
tm3=b.read_tif('sat3.tif')
tm4=b.read_tif('sat4.tif')
tm5=b.read_tif('sat5.tif')
tm7=b.read_tif('sat7.tif')
b.img_prop(tm1)
b.percent(tm1,0.00,0.98)
b.display("sat",tm1,600,600,56,90)

#---------------------------------
# Making Homogenous Class
#-------------------------------
cls1=b.sconv(inc,0.3,0.99,30)
r1,r2=b.percent(tm5,0.2,0.99)
cls2=b.sconv(tm5,r1,r2,30)
r1,r2b.percent(tm7,0.2,0.99)
cls3=b.sconv(tm7,r1,r2,30)

cls=b.mclass(cls1,cls2,cls3)
b.img_prop(cls)
b.display("cls",np.float32(cls3),600,600,0,100)

clsx=b.mclass2(256*inc,tm5,tm7)

b.img_prop(clsx)
b.display("clsx",np.float32(clsx),600,600,0,2000)
cv2.destroyAllWindows()

count=np.zeros(27000,dtype=np.uint32)
for i in range(1200):
    for j in range(1200):
        count[cls[i,j]]=count[cls[i,j]]+1

for k in range(27000):
    if count[k] > 100 :
	print k,count[k]

temp=np.where(count > 100)
print len(temp[0])

total=0
for k in range(27000):
    if count[k] > 100 :
	total=total+count[k]

total/1200.0/1200.0


#------------------------------
#       Processing of Band 1
#------------------------------
h_data=a.read_height("height01.txt")
d_data=a.read_height("depth01.txt")
x=np.arange(20)*0.05
y=d_data[:,4]
z=np.polyfit(x,y,2)
p=np.poly1d(z)
pl.plot(x,y)
pl.plot(x,p(x))
pl.show()

a.set_coeff(x,d_data)
a.d_coeff
a.hdata=copy.deepcopy(h_data[8,:])

tmx=gain[0]*tm1+offset[0]
per=percent

#------------------------------
a.penv0=penv[0]
penvx=np.ones(a.imax*a.jmax).reshape(a.imax,a.jmax)*a.penv0
aerox=aerox0
dfact=0.5
ix=900 ; jy=700
ix=700 ; jy=900
a.adjust(0.2,dem[ix,jy]/1000.0,inc[ix,jy],penvx[ix,jy])
a.set_ref2(dem[ix,jy]/1000.0,tmx[ix,jy],inc[ix,jy],penvx[ix,jy])

rcoef=a.estimate(dem/1000.0,inc,tmx,penvx)

ref=rcoef[0,:,:]+rcoef[1,:,:]*aerox+rcoef[2,:,:]*aerox**2
np.min(ref),np.max(ref),np.mean(ref),np.std(ref)
ref[900,700],ref[700,900]
ref[ref<0.0]=0.0
ref[ref>1.0]=1.0
np.min(ref),np.max(ref),np.mean(ref),np.std(ref)


wm=b.aest(ref,per,cls)
b.img_prop(wm)
wm[700,900],wm[900,700]
b.display("wmx",wm,600,600,0,1.0)

aero1=a.iestimate(wm,rcoef)
b.img_prop(aero1)
b.display("aero",aero1,600,600,0.0,0.5)
aero1[700,900],aero1[900,700]
aero1[890:910,700]
np.nanmean(aero1),np.nanstd(aero1)

aero1x=b.xmedian(aero1)
np.min(aero1x),np.max(aero1x),np.mean(aero1x),np.std(aero1x)
b.img_prop(aero1x)
aero1x[700,900],aero1x[900,700]
b.display("aero",aero1x,600,600,0,0.5)

aerox=(1.0-dfact)*aerox+dfact*aero1x

r1,r2=b.percent(ref,0.05,0.99)
cls3=b.sconv(ref,r1,r2,30)
cls=b.mclass(cls1,cls2,cls3)
#------------------------------
penvx=cv2.blur(ref,(41,41))
rcoef=a.estimate(dem/1000.0,inc,tmx,penvx)
ref=rcoef[0,:,:]+rcoef[1,:,:]*aerox+rcoef[2,:,:]*aerox**2
np.min(ref),np.max(ref),np.mean(ref),np.std(ref)
ref[ref<0.0]=0.0
ref[ref>1.0]=1.0
wm=b.aest(ref,0.5,cls)
aero1=a.iestimate(wm,rcoef)
np.nanmean(aero1),np.nanstd(aero1)
aero1x=b.xmedian(aero1)
aerox=(1.0-dfact)*aerox+dfact*aero1x
r1,r2=b.percent(ref,0.05,0.99)
cls3=b.sconv(ref,r1,r2,30)
cls=b.mclass(cls1,cls2,cls3)


cv2.destroyAllWindows()



