import cv2
import numpy as np
import os
root_path = './right'
# patients_folders = [fd for fd in sorted(os.listdir(root_path)) if '.' not in fd]

# for pf in patients_folders:
#     new_folder_path = root_path + pf + '_cutting/'
#     if not os.path.exists(new_folder_path):
#         os.mkdir(new_folder_path)
#     for img_name in sorted(os.listdir(root_path + pf)): 
#         # print(img_name)
#         img = cv2.imread(root_path + pf + '/' + img_name)[20:980, 310:1590,:]
#         cv2.imwrite(new_folder_path + img_name, img)
#         print(new_folder_path + img_name)


files = [fd for fd in sorted(os.listdir(root_path))]

for i in range(len(files)):
    path=root_path +'/'+ files[i]
    print(path)
    img = cv2.imread(path)
    size=img.shape
    print(size)
    if size==(1080, 1920, 3):
        #print('a')
        cv2.imwrite(str(path),img[20:980, 310:1590,:])

##19625 su r
# # cv2.imwrite('pa2_191200_r.jpg',img[20:980, 310:1590,:])
# # cv2.imshow('a.png',img)
# # cv2.waitKey()
# # cv2.destroyAllWindows