import cv2
import numpy as np
import os
file_uno = open("dati.txt","a")
root_path = './left'
files = [fd for fd in sorted(os.listdir(root_path))]

for i in range (len(files)):
    path1='/home/lcruciani/frame_decenti/rec_left/'+ str(files[i])
    path2='/home/lcruciani/frame_decenti/rec_right/'+ str(files[i])

    file_uno.write(f"{path1}   {path2}\n")

file_uno.close()
