import os
import numpy as np
import cv2
import json
import glob
from lib.utils import get_bfcxcy

# 1. cloud points filter
def filter_cloud_points_by_groud(img, groudmask, calibrationFile, lidar2cameraFile_left, pcd_path):
    # get cloudpoint and calib rt
    def _parse_pcd_rt():
        import open3d as o3d
        ymlfile = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
        Q = ymlfile.getNode('Q').mat().reshape([4,4])
        P1 = ymlfile.getNode('P1').mat().reshape([3,4])
        ymlfile.release()
        b,f,cx,cy = get_bfcxcy(Q)
        K = P1[:, :-1]

        lidar2cameraParam_left = cv2.FileStorage(lidar2cameraFile_left, cv2.FileStorage_READ)
        left_R = lidar2cameraParam_left.getNode('R').mat()
        left_T = lidar2cameraParam_left.getNode('T').mat()

        pcd = o3d.io.read_point_cloud(pcd_path)
        XYZ_w = np.asarray(pcd.points)

        return XYZ_w, left_R, left_T, K

    img_hei, img_wid = img.shape[:2]
    XYZ_w, left_R, left_T, K = _parse_pcd_rt()

    XYZ_l = np.matmul(XYZ_w, left_R.transpose()) + left_T.transpose()
    xyZ_l = np.matmul(XYZ_l, K.transpose())
    xyZ_l[:, 0] /= xyZ_l[:, 2]
    xyZ_l[:, 1] /= xyZ_l[:, 2]

    # change here to avoid near ground
    mask_l = (xyZ_l[:, 0] > 0) * (xyZ_l[:, 0] < img_wid - 1) * (xyZ_l[:, 1] > 0) * (
                xyZ_l[:, 1] < img_hei) * (xyZ_l[:, 2] > 0)

    XYZ_l = XYZ_l[mask_l, :]
    xyZ_l = xyZ_l[mask_l, :]

    # filter by mask
    msk_ = [groudmask[int(xyz[1]), int(xyz[0])]>0 for xyz in xyZ_l]
    groud_pointcloud = XYZ_l[msk_, :]

    return groud_pointcloud


# 2a. polynomial equation
def compute_polynomial(pointcloud):
    return


# optional 2b.
def compute_voxel(pointcloud, voxel_size=0.1):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    print('before processing - {}'.format(len(pointcloud)))
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    downpcd = pcd.voxel_down_sample(voxel_size)
    # ret = pcd.voxel_down_sample_and_trace(0.05, pcd.get_min_bound(), pcd.get_max_bound(), approximate_class=True)
    # downpcd = ret[0]
    # print(pcd.get_min_bound()) # 11w * 8
    # print(pcd.get_max_bound())

    print('after processing - {}'.format(len(np.asarray(downpcd.points))))

    # print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    o3d.visualization.draw_geometries([downpcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024],
                                    point_show_normal=True)

    o3d.io.write_point_cloud('ground.ply', downpcd)

    # print("Print a normal vector of the 0th point")
    print('downsample normal - {}'.format(len(np.asarray(downpcd.normals))))

    downpcd_height = compute_voxel_relative_height(downpcd)
    downpcd_height_pcd = o3d.geometry.PointCloud()
    downpcd_height_pcd.points = o3d.utility.Vector3dVector(downpcd_height)
    o3d.io.write_point_cloud('relative_height.ply', downpcd_height_pcd)



    # points_height = compute_height(pointcloud, downpcd, voxel_size)
    # height_pcd = o3d.geometry.PointCloud()
    # height_pcd.points = o3d.utility.Vector3dVector(points_height)
    # # o3d.visualization.draw_geometries([height_pcd],
    # #                                 zoom=0.3412,
    # #                                 front=[0.4257, -0.2125, -0.8795],
    # #                                 lookat=[2.6172, 2.0475, 1.532],
    # #                                 up=[-0.0694, -0.9768, 0.2024],
    # #                                 point_show_normal=True)
    # o3d.io.write_point_cloud('height.ply', height_pcd)

def compute_voxel_relative_height(voxels, radius=0.2):
    normals = np.asarray(voxels.normals)
    voxel_pc = np.asarray(voxels.points)

    z_sort = voxel_pc[:, -1].argsort()
    normals = normals[z_sort]
    voxel_pc = voxel_pc[z_sort]

    voxel_height_pc = np.zeros_like(voxel_pc)
    voxel_height_pc[:10, 0] = voxel_pc[:10, 0]
    voxel_height_pc[:10, -1] = voxel_pc[:10, -1]

    # index 0 set relative height to 0
    for i in range(10, len(voxel_pc)):
        print('processing {} / {}'.format(i, len(voxel_pc)))
        v = voxel_pc[i] # farest
        # print(v)

        z = v[-1]

        min_dist = 1000
        vp_min = None
        vp_j = -1
        for j in range(i):
            v_p = voxel_pc[j]
            if v_p[-1] >= z :
                continue
            
            if z - v_p[-1] > radius:
                continue

            if v[0] - v_p[0] > radius/2:
                continue

            dist = np.sqrt(((v - v_p)*(v - v_p)).sum())

            if dist < min_dist:
                min_dist = dist
                vp_min = v_p
                vp_j = j

        dist_ = np.dot(v-vp_min, normals[i]) * 100 
        voxel_height_pc[i] = np.array([voxel_pc[i, 0], dist_, voxel_pc[i, -1]])

    print(voxel_height_pc)
    return voxel_height_pc[::-1]
        



# 3. compute_height
def compute_height(pointcloud, downpcd, voxel_size):
    voxels = np.asarray(downpcd.points)
    normals = np.asarray(downpcd.normals)

    pointcloud_height = np.zeros_like(pointcloud)

    # have to know point belongs to which voxel
    for i, point in enumerate(pointcloud):
        z = point[2] # x,y,z
        min_dist = 1000
        min_j = -1
        target_voxel = None
        for j, voxel in enumerate(voxels):
            v_z = voxel[2]
            diff_z = z - v_z
            dist = np.sqrt(((voxel - point)*(voxel - point)).sum())
            if diff_z < 0 and dist < min_dist:
                min_dist = dist
                min_j = j
                target_voxel = voxel
        if target_voxel is not None:
            target_normal = normals[min_j]
            dist_ = np.dot(point-target_voxel, target_normal)

            pointcloud_height[i] = np.array([point[0], dist_ * 100, z])
        else:
            print(i)
    
    return pointcloud_height


if __name__ == '__main__':
    left_img = cv2.imread('img.png')
    left_mask = cv2.imread('mask.png', 0)
    calib_yml = 'calib.yml'
    lidar_yml = 'lidar.yml'
    pcd = 'demo.pcd'

    # filter pointcloud
    traj_mask = np.zeros_like(left_mask)
    traj_mask = cv2.fillPoly(traj_mask, [np.array([[500, 800], [550, 800], [480, 1200], [400, 1200]], dtype=np.int32)], 255)
    cv2.imwrite('traj_mask.png', traj_mask)

    left_mask = left_mask * traj_mask
    ground_pointcloud = filter_cloud_points_by_groud(left_img, left_mask, calib_yml, lidar_yml, pcd)

    # ground_pointcloud = ground_pointcloud[::1000,:]


    # use ground points to calculate mesh
    compute_voxel(ground_pointcloud) # 227w too much 