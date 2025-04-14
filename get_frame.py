import cv2

def cutVideo():
    i=0
    video = cv2.VideoCapture(r"C:\Users\VIPLAB\Desktop\Yan\Lee's Family Reunion EP233 preview.mp4")
    if video.isOpened():
        frame = 0
    while(True):
        ret,frame = video.read()
        #cv2.imshow('video',frame)
        c = cv2.waitKey(10)
        
        if c == 27: 
            break
        i += 1
        if i%10==0:
            if(frame.all()!=None):
                cv2.imwrite( str(i) + '.png',frame)

cutVideo()
cv2.destroyAllWindows()

        