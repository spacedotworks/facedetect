import cv2
import sys
import requests
import urllib
import numpy as np

#f = open('/var/www/html/tmp.jpg', 'wb')
#f.write(requests.get(sys.argv[1]).content)
#f.close()
#imagePath = '/var/www/html/tmp.jpg' #sys.argv[1]
cascPath = 'haar_face' #sys.argv[2]

req = urllib.urlopen(sys.argv[1])
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr,-1)

# Create the haar cascade

# Read the image
#image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cascPath)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))


# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    
#    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    template = image[y:y+h,x:x+w]
    template = cv2.resize(template,(5,5),template,0,0,cv2.INTER_NEAREST)
    template = cv2.resize(template,(w,h),template,0,0,cv2.INTER_NEAREST)
    image[y:y+h,x:x+w] = template
#    print template
cv2.imwrite('/var/www/html/output.jpg',image)
#cv2.imwrite('/var/www/html/face.jpg',template)

#cv2.imshow("Faces found", image)
#cv2.waitKey(0)
