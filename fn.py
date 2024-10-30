import cv2
import os
import numpy as np
import torch
import time
import copy
import torch.backends.cudnn as cudnn
from models.Unet import U_Net
import open3d as o3d
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_frames(video_path, output_folder, start_frame=0, end_frame=None, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video can't be opened")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(frame_count)
        
        if frame_count >= start_frame:
            if end_frame is not None and frame_count > end_frame:
                break
            
            if (frame_count - start_frame) % frame_interval == 0:
                # Salva il frame nella cartella di output con il numero di frame come nome del file
                frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
def cut_frames(img):
    return img[:960, 320:1600, :]

def load_camera_param (xml_file):
    #print(xml_file)
    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    # The intrinsic parameters (intrinsic matrix), respectively for left and right camera
    Kl = cv_file.getNode("M_l").mat()
    Kr = cv_file.getNode("M_r").mat()
    # The distortion parameters, respectively for left and right camera
    dist_l = cv_file.getNode("D_l").mat().flatten()
    dist_r = cv_file.getNode("D_r").mat().flatten()
    # The extrinsic parameters, respectively rotation and translation matrix
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    cv_file.release()
    return Kl, Kr, dist_l,dist_r, R, T

def stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T):
        (R_l, R_r, P_l, P_r, Q, roi1, roi2) = cv2.stereoRectify(cameraMatrix1 = Kl,cameraMatrix2 = Kr,
                                                                distCoeffs1 = dist_l, distCoeffs2 = dist_r,
                                                                imageSize = (h,w),
                                                                R = R, T = T,
                                                                #flags = cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha = 0.5,
                                                                newImageSize = (h,w) )
        '''
        cameraMatrix - the two intrinsic matrix
        R : Rotation matrix between the coordinate systems of the first and the second cameras.
        T : Translation vector between coordinate systems of the cameras.
        R1 : Output 3x3 rectification transform (rotation matrix) for the first camera.
        R2 : Output 3x3 rectification transform (rotation matrix) for the second camera.
        P1 : Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        P2 : Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        Q : Output 4 \times 4 disparity-to-depth mapping matrix '''
        return R_l, R_r, P_l, P_r, Q

def elaboration_param(Kl, Kr, dist_l, dist_r, R, T, h, w, h_res, w_res):
    
    # Computation of matrix Q tha's used for conversion from disparity to depth
    x_ratio = w_res / w
    y_ratio = h_res / h
    [R_l, R_r, P_l, P_r, _] = stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T)
    Kl_res = copy.copy(Kl)
    Kr_res = copy.copy(Kr)
    Kl_res[0,:] = Kl_res[0,:] * x_ratio
    Kl_res[1,:] = Kl_res[1,:] * y_ratio
    Kr_res[0,:] = Kr_res[0,:] * x_ratio
    Kr_res[1,:] = Kr_res[1,:] * y_ratio
    [_,_,_,_,Q] = stereoRectify(Kl_res, Kr_res, dist_l, dist_r, h_res, w_res, R, T)  
    
    # Computation of maps used for rectification
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(Kl, dist_l, R_l, P_l, (w,h), cv2.CV_16SC2)    #CV_32F
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(Kr, dist_r, R_r, P_r, (w,h), cv2.CV_16SC2)    #CV_32F

    return Q, mapx_l, mapy_l, mapx_r, mapy_r, R_l

def rectification_resize(img, mapx, mapy, h_res, w_res):
    
    # undistorting the images witht the calculated undistortion map
    result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # RESIZE
    img_res = cv2.resize(result, (w_res,h_res), interpolation = cv2.INTER_CUBIC)
    return img_res

def model_loading(path):
    '''Load model 
        input: path of the weigths
    '''
    model = U_Net().to('cuda')
    cudnn.benchmark = True
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    print('\033[0;33mModel Loaded!\033[0m')
    return model


def apply_transforms(image, transform):
    '''Apply the given albumentations transform to the image and return the tensor'''
    if transform is not None:
        augmented = transform(image=image)
        image = augmented["image"]
    return image

# def inference(frame, model, transform):
#     '''Inference
#         input: 
#         image --> imported in PIL,
#         model
#         transform --> albumentations transform

#         output:
#             prediction,
#             binary mask
#     '''
#     frame_transformed = apply_transforms(np.array(frame), transform).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU
#     model.eval()
#     with torch.no_grad():
#         start = time.time()
#         preds = torch.sigmoid(model(frame_transformed))
#         preds = (preds > 0.55).float()
#         end = time.time() - start
    
#     segmented_frame = (preds.squeeze().cpu().numpy() * 255).astype(np.uint8)
#     return preds, segmented_frame


def inference(frame, model):
    '''Inference
        input: 
        image--> imported in cv2,
        model

        output:
            prediction,
            binary mask
    '''
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]
    val_transforms = A.Compose([
            A.Resize(height=160, width=240),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,),
            ToTensorV2(),
        ]
    )
    #frame=apply_transforms(frame,val_transforms)
    
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (240, 160))
    frame = np.asarray(frame)
    frame = frame.transpose((2, 0, 1))  
    frame = frame / 255.0

    frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).to('cuda') 
    model.eval()
    with torch.no_grad():
        start = time.time()
        preds = torch.sigmoid(model(frame_tensor))
        preds = (preds > 0.60).float()
        end = time.time() - start
    
    segmented_frame = (preds.squeeze().cpu().numpy() * 255).astype(np.uint8)
    segmented_frame = cv2.resize(segmented_frame, (frame_width, frame_height))
    return preds, segmented_frame

def point_cloud(dispTo3D, Q, h, w) :
    points_3D = cv2.reprojectImageTo3D(dispTo3D, Q)
    points_3D = np.reshape(points_3D,(h*w,3))
    points_3D[~np.isfinite(points_3D)] = 0
    return points_3D

def ask_user_to_save():
    user_input = input("Do you want to save the current registration? (y/n): ")
    return user_input.lower() == 'y'

def normals_features(pcd_down, voxel_size):
    "Estimate normals and FPFH features"
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 3

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    "Definition of the registration with RANSAC"
    distance_threshold = voxel_size * 1

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result

def registration(source,target):
    ### EXTRACT FEATURE DESCRIPTOR
    voxel_size = 1
    source_down = source.voxel_down_sample(voxel_size)
    source_fpfh = normals_features(source_down, voxel_size)
    target_down = target.voxel_down_sample(voxel_size * 0.3)
    target_fpfh = normals_features(target_down, voxel_size)
    ### MATCH FEATURE DESCRIPTOR and RIGID REGISTRATION
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    source_trasformed = source_down.transform(result_ransac.transformation)
    ### DEFORMABLE REGISTRATION

    return source_trasformed

# def reprojection_2d(source_trasformed,Kl, img): 
#     points=np.asarray(source_trasformed.points)
#     points_2d, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3), Kl, np.zeros(4))
#     points_2d=points_2d.reshape(-1, 2)
#     mask = np.zeros_like(img)
#     for pt in points_2d:
#         x, y = int(pt[0]), int(pt[1])
#         if 0 <= x < 1080 and 0 <= y < 960:
#             cv2.circle(mask, (x, y), 8, (0, 255, 0), -1) 
#     final_image = cv2.addWeighted(img, 1, mask, 0.8, 0)
#     return final_image
def reprojection_2d(source_transformed, Kl, img):
    points = np.asarray(source_transformed.points)

    points_2d, _ = cv2.projectPoints(points, np.eye(3), np.zeros(3), Kl, np.zeros(4))
    points_2d = points_2d.reshape(-1, 2)
    
    mask = np.zeros_like(img)
    
    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(mask, (x, y), 8, (0, 255, 0), -1)  # Proiezione dei punti 3D come cerchi
    
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    
    final_image = cv2.addWeighted(img, 1, mask_blurred, 0.8, 0)
    
    return final_image



def icp_refinement(source, target, initial_transform, voxel_size):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    distance_threshold = voxel_size * 0.5
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria)
    return result_icp.transformation

# def elastic_registration(source, target, init_transform, lambda_smooth=0.1):
#     # Usa ICP deformabile con un termine di regolarizzazione per la smoothness
#     distance_threshold = 0.05
#     criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, distance_threshold, init_transform,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         criteria)
#     # Aggiungi un termine di regolarizzazione per limitare la deformazione eccessiva
#     smoothed_result = regularize_deformation(result.transformation, lambda_smooth)
#     return smoothed_result

# def combined_registration(source, target, num_trials=5, lambda_smooth=0.1):
#     """
#     Combines multiple registration techniques to achieve a stable alignment of point clouds.

#     Parameters:
#     - source: The source point cloud (Open3D format)
#     - target: The target point cloud (Open3D format)
#     - voxel_sizes: List of voxel sizes for multi-resolution registration
#     - num_trials: Number of trials for averaging registration results
#     - lambda_smooth: Regularization parameter for deformation smoothness

#     Returns:
#     - final_transform: The final transformation matrix after registration
#     """

#     # Step 1: Multi-Resolution Registration
#     all_transforms = []
#     voxel_sizes=[1, 1, 1]
#     for v_size in voxel_sizes:
#         print(f"Registering at voxel size: {v_size}")

#         # Downsample point clouds
#         source_down = source.voxel_down_sample(v_size)
#         target_down = target.voxel_down_sample(v_size)

#         # Estimate normals and compute FPFH features
#         source_fpfh = normals_features(source_down, v_size)
#         target_fpfh = normals_features(target_down, v_size)

#         # Execute global registration using RANSAC
#         result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, v_size)
#         all_transforms.append(result_ransac.transformation)

#     # Step 2: Average Transformations
#     avg_transform = np.mean(all_transforms, axis=0)
#     print(avg_transform)
#     # Step 3: ICP Refinement
#     source_transformed = source.transform(avg_transform)
#     #result_icp = icp_refinement(source_transformed, target, avg_transform, voxel_sizes[0])

#     # Step 4: Apply regularization to the final transformation
#     #final_transform = elastic_registration(source, target, result_icp.transformation, lambda_smooth)
    
#     return source_transformed