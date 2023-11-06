import cv2

### PATH OF VIDEOS (input)
path_video_left=''
path_video_rigth=''
### PATH OF IMAGES (output)
path_left='.\\left_frames'
path_rigth='.\\right_frames'
## Extract left images 
freq=70 ## Take 1 image every 70 frames
vidcap = cv2.VideoCapture(path_video_left)
success,image = vidcap.read()
count = 0
a=0
success = True
while success:
  success,image = vidcap.read()
  if (count%freq==0):
    import os
    a=count/freq
    cv2.imwrite(os.path.join(path_left,str(a) + '.png'), image)
    #cv2.imwrite("frameL%d.jpg" % a, image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
  count += 1

count=0
a=0
### Extract rigth images
vidcap = cv2.VideoCapture(path_video_rigth)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  if (count%freq==0):
    
    #os.makedirs(path)
    
    a=count/freq
    cv2.imwrite(os.path.join(path_rigth,str(a) + '.png'), image)
    #cv.imwrite(os.path.join('C:\\Users\\Laura\\OneDrive\\Desktop\\3d_recostruction\\heart1\\Rect_r',str(i) + '.png'), rect2)
    #cv2.imwrite("frameR%d.jpg" % a, image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
  count += 1