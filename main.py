### IMPORT LIBRARIES
from fn import *
from sksurgerytorch.models import high_res_stereo
import open3d as o3d
#### 1.Extract frame from the videos
left_video_path = "left.mp4"
right_video_path = "right.mp4"
## OUTPUT FOLDER FOR THE FRAMES
left_output_folder = "left2"
right_output_folder = "right_frames"
#extract_frames(left_video_path, left_output_folder, start_frame=58000, end_frame=58600, frame_interval=1)
#extract_frames(right_video_path, right_output_folder, start_frame=58000, end_frame=58600, frame_interval=1)

files = os.listdir(left_output_folder)
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
h=960
w=1280
h_res=480
w_res=640
HSMNet = high_res_stereo.HSMNet(max_disp=192,
                                entropy_threshold=-1,
                                level=1,
                                scale_factor=1,
                                weights='./weights/final-768px.tar')

net=model_loading("best_model.pth")
[Kl, Kr, dist_l, dist_r, R, T] = load_camera_param('CalibrationParameters.xml')
[Q, mapx_l, mapy_l, mapx_r, mapy_r,Rl] = elaboration_param(Kl, Kr, dist_l, dist_r, R, T, h, w, 
                                                        h_res, w_res)

source=o3d.io.read_point_cloud("source.ply")

for image_file in image_files:
    start=time.time()
    imgL=cv2.imread(os.path.join(left_output_folder, image_file))
    imgR=cv2.imread(os.path.join(right_output_folder,image_file))
    print(imgL.shape)
    print(f"checkpoint1: {time.time()-start}")
    ### CUT THE IMAGES IF NEEDED
    if(imgL.shape!=(960,1280,3)):
        print("CUTTING")
        imgL=cut_frames(imgL)
        imgR=cut_frames(imgR)
    ### RECTIFICATION
    imgL_resized = rectification_resize(imgL, mapx_l, mapy_l, h_res, w_res)
    imgR_resized = rectification_resize(imgR, mapx_r, mapy_r, h_res, w_res)
    print(f"checkpoint2: {time.time()-start}")
    ### Disparity Computation
    disparity_hsm, _ = HSMNet.predict(imgL_resized, imgR_resized)
    print(f"checkpoint3: {time.time()-start}")
    # cv2.imshow("a", disparity_hsm)
    # cv2.waitKey(100)
    ### Mask inference
    _,mask=inference(imgL_resized,net)
    print(f"checkpoint4: {time.time()-start}")

    ### Point Cloud Computation
    points_3D_hsm = point_cloud(disparity_hsm, Q, h_res, w_res)
    valid_depth_ind = np.where(points_3D_hsm[:,2].flatten() > 0)[0]
    points_3D_hsm = points_3D_hsm[valid_depth_ind,:]
    colors = cv2.cvtColor(imgL_resized, cv2.COLOR_BGR2RGB)
    colors = np.reshape(colors,(h_res*w_res,3))
    colors = colors[valid_depth_ind,:]
    valid_color_ind = np.where(colors[:,2].flatten() > 0)[0]
    colors=colors[valid_color_ind,:]
    points_3D_hsm=points_3D_hsm[valid_color_ind,:]
    print(f"checkpoint5: {time.time()-start}")
    

    ### Vessel Point Cloud
    mask_flat = mask.flatten()
    mask_points = mask_flat[valid_depth_ind]
    mask_points = mask_points[valid_color_ind]
    mask_indices = np.where(mask_points > 0)[0]
    points_3D_hsm = points_3D_hsm[mask_indices, :]
    colors = colors[mask_indices, :]
    print(f"checkpoint6: {time.time()-start}")

    ### Registration
    ### TARGET
    target=o3d.geometry.PointCloud()
    target.points=o3d.utility.Vector3dVector(points_3D_hsm)
    print(f"checkpoint6.5: {time.time()-start}")
    ### FEATURE EXTRACTION 
    registered_pc=registration(source,target)
    print(f"checkpoint7: {time.time()-start}")
    #o3d.visualization.draw_geometries([target,source,registered_pc])
    #o3d.io.write_point_cloud('reg.ply',registered_pc)
    ### Reprojection
    reprojected_image=reprojection_2d(registered_pc,Kl,imgL)

    ### CONTROLLO (DA TOGLIERE)
    ### SHOW REPROJECTED IMAGE
    #cv2.imshow("a",reprojected_image)
    #cv2.waitKey(100)
    # ans=ask_user_to_save()
    # print(ans)
    # if ans==True: 
    #     last=registered_pc
    # if ans==False: 
    #     reprojected_image=reprojection_2d(last,Kl,imgL)

    cv2.imwrite(f"result/{image_file}",reprojected_image)
    stop=time.time()-start
    print(f"Time:{stop}")
    
    
