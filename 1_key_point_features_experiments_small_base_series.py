import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import copy
import torch
from PIL import Image
import sys


# """**Parse and process JSON data, get fundamental matrices**"""

#Load JSON data
with open("data/param.json") as f:
    param = json.load(f)

Camera, Homo = param['Camera'], param['Homo']

distortion_coeffs, intrinsics = np.array(Camera['distortion_coeffs']), np.array(Camera['intrinsics'])
mtx = np.array([[intrinsics[0], 0, intrinsics[2]],[0, intrinsics[1], intrinsics[3]],[0, 0, 1]])

#Create variables for T-cam-matrices
for key, value in Homo.items():
  globals()[key] = np.array(value)

#Create function to calculate fundamental matrices
def fund(T_cam_cam, mtx):
  shift = T_cam_cam[0:3, 3]
  skew = np.array([[0, -shift[2], shift[1]],[shift[2], 0, -shift[0]],[-shift[1], shift[0], 0]])
  ess = np.dot(skew, T_cam_cam[0:3, 0:3])
  fund = np.dot(np.dot(np.linalg.inv(np.transpose(mtx)),ess),np.linalg.inv(mtx))
  return fund

#Calculate fundamental matrices
num_images = 39
T_fund_list = []

for i in range(num_images-1):
  T_fund_list.append(fund(globals()['T_cam%s'%i+'_cam%s'%(i+1)], mtx))

# """**Import images, remove distortion**"""

for i in range(num_images):
  globals()['image%s'%i] = cv.imread("data/images/"+str(i)+".png")
  globals()['img_undist%s'%i] = cv.fisheye.undistortImage((globals()['image%s'%i]), mtx, distortion_coeffs, None, mtx)

# """**General functions - Create function to apply ratio test**"""

# Create function to apply ratio test
def ratio_test(matches, ratio_index):
  good = []
  good_ind0 = []
  good_ind1 = []
  
  for m,n in matches:
    if m.distance < ratio_index*n.distance:
        good.append([m])
        good_ind0.append(m.queryIdx)
        good_ind1.append(m.trainIdx)
  return good, good_ind0, good_ind1

# """**General functions - Create function for Refinement**"""

sys.path.insert(0, 'local-feature-refinement/two-view-refinement/')
from refinement import refine_matches_coarse_to_fine
from model import PANet
from refinement import *

def refine_matches_coarse_to_fine_2(
        image1, keypoints1,
        image2, keypoints2,
        net, device, batch_size,
        symmetric=True, grid=True
):
    ij1 = keypoints1[:, [1, 0]]
    ij2 = keypoints2[:, [1, 0]]

    if symmetric:
        # Coarse refinement.
        coarse_displacements12, coarse_displacements21 = extract_patches_and_estimate_displacements(
            image1, ij1,
            image2, ij2,
            net, device, batch_size,
            symmetric=symmetric, grid=False, octave=0.
        )

        # Fine refinement.
        up_image1 = cv.pyrUp(image1)
        up_image2 = cv.pyrUp(image2)

        displacements12 = .5 * extract_patches_and_estimate_displacements(
            up_image1, 2. * ij1,
            up_image2, 2. * (ij2 + coarse_displacements12 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )
        displacements21 = .5 * extract_patches_and_estimate_displacements(
            up_image2, 2. * ij2,
            up_image1, 2. * (ij1 + coarse_displacements21 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )

        if grid:
            return coarse_displacements12[:, np.newaxis, np.newaxis] + displacements12, coarse_displacements21[:, np.newaxis, np.newaxis] + displacements21
        else:
            return coarse_displacements12 + displacements12, coarse_displacements21 + displacements21
    else:
        # Coarse refinement.
        coarse_displacements12 = extract_patches_and_estimate_displacements(
            image1, ij1,
            image2, ij2,
            net, device, batch_size,
            symmetric=symmetric, grid=False, octave=0.
        )

        # Fine refinement.
        up_image1 = cv.pyrUp(image1)
        up_image2 = cv.pyrUp(image2)

        displacements12 = .5 * extract_patches_and_estimate_displacements(
            up_image1, 2. * ij1,
            up_image2, 2. * (ij2 + coarse_displacements12 * 16),
            net, device, batch_size,
            symmetric=False, grid=grid, octave=-1.
        )

        if grid:
            return coarse_displacements12[:, np.newaxis, np.newaxis] + displacements12
        else:
            return coarse_displacements12 + displacements12

torch.set_grad_enabled(False)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Create the two-view estimation network.
net = PANet(model_path='local-feature-refinement/two-view-refinement/checkpoint.pth').to("cpu")


#Create function to calculate and apply refinement displacements for different methods
def apply_refinement(kp0, kp1, img0, img1):
  matches_refine = np.stack([np.arange(kp0.shape[0]), np.arange(kp0.shape[0])]).T
  grid_displacements12, grid_displacements21 = refine_matches_coarse_to_fine(img0, kp0, img1, kp1, matches_refine,
                                                                                       net, device, 1024, symmetric=True, grid=False)
  kp0[:, 0] += grid_displacements21[:, 1]*16
  kp0[:, 1] += grid_displacements21[:, 0]*16
  kp1[:, 0] += grid_displacements12[:, 1]*16
  kp1[:, 1] += grid_displacements12[:, 0]*16
  return kp0, kp1

# """**General functions - Compute epipolar lines**"""

def calc_epi(kp0, kp1, T0_1_fund, T1_0_fund):
  kp0_hom = np.hstack((kp0, [[1]]*len(kp0)))
  kp1_hom = np.hstack((kp1, [[1]]*len(kp1)))
  epi0 = np.transpose(np.dot(T0_1_fund, np.transpose(kp1_hom)))
  epi1 = np.transpose(np.dot(T1_0_fund, np.transpose(kp0_hom)))
  return epi0, epi1, kp0_hom, kp1_hom

# """**General functions - Draw epipolar lines**"""

# Create function to draw epipolar lines
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[0:2]
    img1_lines = img1.copy()
    img2_lines = img2.copy()

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1_lines = cv.line(img1_lines, (x0,y0), (x1,y1), color,1)
        img1_lines = cv.circle(img1_lines,tuple(map(int,pt1)),5,color,-1)
        img2_lines = cv.circle(img2_lines,tuple(map(int,pt2)),5,color,-1)
    return img1_lines,img2_lines

# """**General functions - Calculate distance measures**"""

# Create function to determine distance measure
def dist_measure(kp, epi_line):
  dist = np.multiply(kp, epi_line)
  dist = np.sum(dist, axis=1, keepdims=True)
  dist = dist/np.linalg.norm(epi_line[:,:2], axis=1, keepdims=True)
  dist = np.abs(dist)
  return dist

# """**General functions - Exclude zone of car's hood**"""

# Manually set variable at level of car's hood (y-axis)
h_of_hood = 310

# Create function to exclude zone of car's hood
def excl_hood(kp0_good, kp1_good):
  buf = copy.deepcopy(kp0_good)
  kp0_good = kp0_good[(buf[:,1]<h_of_hood)]
  kp1_good = kp1_good[(buf[:,1]<h_of_hood)]
  return kp0_good, kp1_good

"""**SIFT function**"""

#General
sift = cv.SIFT_create()
eps = 1e-7
bf = cv.BFMatcher()


#Create function to apply SIFT to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points
def SIFT_pairs(img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):
  kp0, des0 = sift.detectAndCompute(img_undist0,None)
  kp1, des1 = sift.detectAndCompute(img_undist1,None)

  des0 /= (des0.sum(axis=1, keepdims=True) + eps)
  des0 = np.sqrt(des0)
  des1 /= (des1.sum(axis=1, keepdims=True) + eps)
  des1 = np.sqrt(des1)

  matches_sift = bf.knnMatch(des0,des1,k=2)

  # Apply ratio test
  good_sift, good_ind0_sift, good_ind1_sift = ratio_test(matches_sift, 0.75)

  # Draw SIFT matches
  sift_matches_0_1 = cv.drawMatchesKnn(img_undist0,kp0,img_undist1,kp1,good_sift,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  #Prepare Numpy arrays for good keypoints
  kp0_np = cv.KeyPoint_convert(kp0)
  kp1_np = cv.KeyPoint_convert(kp1)
  kp0_good = kp0_np[:,:][good_ind0_sift]
  kp1_good = kp1_np[:,:][good_ind1_sift]

  #Exclude key points from zone of car's hood
  kp0_good, kp1_good = excl_hood(kp0_good, kp1_good)

  # Refinement of keypoints for SIFT
  kp0_good, kp1_good = apply_refinement(kp0_good, kp1_good, img_undist0, img_undist1)

  # Epipolar lines - SIFT
  epi_lines_0_sift, epi_lines_1_sift, kp0_good_hom, kp1_good_hom = calc_epi(kp0_good, kp1_good, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - SIFT
  img_undist0_sift = img_undist0.copy()
  img_undist1_sift = img_undist1.copy()

  img_lines_0_1_sift, _0_1_sift = drawlines(img_undist0_sift,img_undist1_sift,epi_lines_0_sift,kp0_good,kp1_good)
  img_lines_1_0_sift, _1_0_sift = drawlines(img_undist1_sift,img_undist0_sift,epi_lines_1_sift,kp1_good,kp0_good)

  # Calculate distance measures - SIFT
  dist0_sift = dist_measure(kp0_good_hom, epi_lines_0_sift)
  dist1_sift = dist_measure(kp1_good_hom, epi_lines_1_sift)
  sum_dist_sift = dist0_sift + dist1_sift

  return sift_matches_0_1, img_lines_0_1_sift, img_lines_1_0_sift, sum_dist_sift, kp0_good # - kp0_good is saved for further usage in RLOF section

# """**r2d2 function**"""

sys.path.insert(0, 'r2d2/')
from r2d2.extract import extract_multiscale, NonMaxSuppression, load_network
from r2d2.tools.dataloader import norm_RGB

model = load_network('r2d2/models/r2d2_WASF_N16.pt')
detector = NonMaxSuppression(0.7, 0.7)


#Create function to apply r2d2 to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points.
def r2d2_pairs(img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):
  img0 = Image.fromarray(np.uint8(img_undist0)).convert('RGB')
  img0 = norm_RGB(img0)[None]

  img1 = Image.fromarray(np.uint8(img_undist1)).convert('RGB')
  img1 = norm_RGB(img1)[None]

  xys0, desc0, scores0 = extract_multiscale(model, img0, detector)
  xys1, desc1, scores1 = extract_multiscale(model, img1, detector)

  img0, img1 = img0.cpu().numpy(), img1.cpu().numpy()
  xys0, xys1 = xys0.cpu().numpy(), xys1.cpu().numpy()
  desc0, desc1 = desc0.cpu().numpy(), desc1.cpu().numpy()
  scores0, scores1 = scores0.cpu().numpy(), scores1.cpu().numpy()

  # Match key points identified by r2d2
  matches_r2d2 = bf.knnMatch(desc0,desc1,k=2)

  # Apply ratio test
  good_r2d2, good_ind0_r2d2, good_ind1_r2d2 = ratio_test(matches_r2d2, 0.8)

  xys0_cv = [cv.KeyPoint(xys0[i,0],xys0[i,1], xys0[i,2]) for i in range(xys0.shape[0])]
  xys1_cv = [cv.KeyPoint(xys1[i,0],xys1[i,1], xys1[i,2]) for i in range(xys1.shape[0])]

  r2d2_matches_0_1 = cv.drawMatchesKnn(img_undist0,xys0_cv,img_undist1,xys1_cv,good_r2d2,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  xys0_good = xys0[:,:2][good_ind0_r2d2]
  xys1_good = xys1[:,:2][good_ind1_r2d2]

  #Exclude key points from zone of car's hood
  xys0_good, xys1_good = excl_hood(xys0_good, xys1_good)

  # Refinement of keypoints for r2d2
  xys0_good, xys1_good = apply_refinement(xys0_good, xys1_good, img_undist0, img_undist1)

  # Epipolar lines - R2D2
  epi_lines_0_r2d2, epi_lines_1_r2d2, xys0_good_hom, xys1_good_hom = calc_epi(xys0_good, xys1_good, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - r2d2
  img_undist0_r2d2 = copy.deepcopy(img_undist0)
  img_undist1_r2d2 = copy.deepcopy(img_undist1)

  img_lines_0_1_r2d2, _0_1_r2d2 = drawlines(img_undist0_r2d2,img_undist1_r2d2,epi_lines_0_r2d2,xys0_good,xys1_good)
  img_lines_1_0_r2d2, _1_0_r2d2 = drawlines(img_undist1_r2d2,img_undist0_r2d2,epi_lines_1_r2d2,xys1_good,xys0_good)

  # Calculate distance measures - r2d2
  dist0_r2d2 = dist_measure(xys0_good_hom, epi_lines_0_r2d2)
  dist1_r2d2 = dist_measure(xys1_good_hom, epi_lines_1_r2d2)
  sum_dist_r2d2 = dist0_r2d2 + dist1_r2d2

  return r2d2_matches_0_1, img_lines_0_1_r2d2, img_lines_1_0_r2d2, sum_dist_r2d2

r2d2_matches_0_1_all, img_lines_0_1_r2d2_all, img_lines_1_0_r2d2_all = [], [], []
sum_dist_r2d2_all = np.empty(shape=[0, 1])

for i in range(num_images-1):
  r2d2_matches_0_1, img_lines_0_1_r2d2, img_lines_1_0_r2d2, sum_dist_r2d2 = r2d2_pairs(globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)
  r2d2_matches_0_1_all.append(r2d2_matches_0_1)
  img_lines_0_1_r2d2_all.append(img_lines_0_1_r2d2)
  img_lines_1_0_r2d2_all.append(img_lines_1_0_r2d2)
  sum_dist_r2d2_all = np.append(sum_dist_r2d2_all, sum_dist_r2d2, axis=0)


median_dist_r2d2 = np.median(sum_dist_r2d2_all)
ave_dist_r2d2 = np.average(sum_dist_r2d2_all)

# """**RLOF function**"""

#Create function to apply RLOF to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points.
def RLOF_pairs(kp0_good, img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):

  kp0_rlof = copy.deepcopy(kp0_good)
  kp1_rlof, st, err = cv.optflow.calcOpticalFlowSparseRLOF(img_undist0, img_undist1, kp0_good, None)
  kp1_rlof, st, err = cv.optflow.calcOpticalFlowSparseRLOF(img_undist0, img_undist1, kp0_good, None)

  h, w, d = img_undist0.shape
  kp0_rlof = kp0_rlof[(kp1_rlof[:,0]>0) & (kp1_rlof[:,0]<w) & (kp1_rlof[:,1]>0) & (kp1_rlof[:,1]<h)]
  kp1_rlof = kp1_rlof[(kp1_rlof[:,0]>0) & (kp1_rlof[:,0]<w) & (kp1_rlof[:,1]>0) & (kp1_rlof[:,1]<h)]

  good_rlof = [[cv.DMatch(i, i, 0)] for i in range(0, len(kp0_rlof))]
  kp0_rlof_cv = [cv.KeyPoint(kp0_rlof[i,0],kp0_rlof[i,1], 1) for i in range(kp0_rlof.shape[0])]
  kp1_rlof_cv = [cv.KeyPoint(kp1_rlof[i,0],kp1_rlof[i,1], 1) for i in range(kp1_rlof.shape[0])]

  rlof_matches_0_1 = cv.drawMatchesKnn(img_undist0,kp0_rlof_cv,img_undist1,kp1_rlof_cv,good_rlof,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  # Refinement of keypoints for RLOF
  kp0_rlof, kp1_rlof = apply_refinement(kp0_rlof, kp1_rlof, img_undist0, img_undist1)

  # Epipolar lines - RLOF
  epi_lines_0_rlof, epi_lines_1_rlof, kp0_rlof_hom, kp1_rlof_hom = calc_epi(kp0_rlof, kp1_rlof, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - RLOF
  img_undist0_rlof = copy.deepcopy(img_undist0)
  img_undist1_rlof = copy.deepcopy(img_undist1)

  img_lines_0_1_rlof, _0_1_rlof = drawlines(img_undist0_rlof,img_undist1_rlof,epi_lines_0_rlof,kp0_rlof,kp1_rlof)
  img_lines_1_0_rlof, _1_0_rlof = drawlines(img_undist1_rlof,img_undist0_rlof,epi_lines_1_rlof,kp1_rlof,kp0_rlof)

  # Calculate distance measures - RLOF
  dist0_rlof= dist_measure(kp0_rlof_hom, epi_lines_0_rlof)
  dist1_rlof = dist_measure(kp1_rlof_hom, epi_lines_1_rlof)
  sum_dist_rlof = dist0_rlof + dist1_rlof

  return rlof_matches_0_1, img_lines_0_1_rlof, img_lines_1_0_rlof, sum_dist_rlof

# """**ALIKE function**"""

sys.path.insert(0, 'ALIKE/')
from ALIKE.alike import ALike, configs

model = ALike(**configs['alike-t'], device='cpu')


#Create function to apply ALIKE to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points.
def ALIKE_pairs(img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):
  pred0 = model(img_undist0, sub_pixel=True)
  kp0_alike = pred0['keypoints']
  des0_alike = pred0['descriptors']

  pred1 = model(img_undist1, sub_pixel=True)
  kp1_alike = pred1['keypoints']
  des1_alike = pred1['descriptors']

  matches_alike = bf.knnMatch(des0_alike,des1_alike,k=2)

  # Apply ratio test
  good_alike, good_ind0_alike, good_ind1_alike = ratio_test(matches_alike, 0.87)

  kp0_alike_cv = [cv.KeyPoint(kp0_alike[i,0],kp0_alike[i,1], 1) for i in range(kp0_alike.shape[0])]
  kp1_alike_cv = [cv.KeyPoint(kp1_alike[i,0],kp1_alike[i,1], 1) for i in range(kp1_alike.shape[0])]

  alike_matches_0_1 = cv.drawMatchesKnn(img_undist0,kp0_alike_cv,img_undist1,kp1_alike_cv,good_alike,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  kp0_good_alike = kp0_alike[:,:][good_ind0_alike]
  kp1_good_alike = kp1_alike[:,:][good_ind1_alike]

  #Exclude key points from zone of car's hood
  kp0_good_alike, kp1_good_alike = excl_hood(kp0_good_alike, kp1_good_alike)

  # Refinement of keypoints for ALIKE
  kp0_good_alike, kp1_good_alike = apply_refinement(kp0_good_alike, kp1_good_alike, img_undist0, img_undist1)

  # Epipolar lines - ALIKE
  epi_lines_0_alike, epi_lines_1_alike, kp0_alike_hom, kp1_alike_hom = calc_epi(kp0_good_alike, kp1_good_alike, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - ALIKE
  img_undist0_alike = copy.deepcopy(img_undist0)
  img_undist1_alike = copy.deepcopy(img_undist1)

  img_lines_0_1_alike, _0_1_alike = drawlines(img_undist0_alike,img_undist1_alike,epi_lines_0_alike,kp0_good_alike,kp1_good_alike)
  img_lines_1_0_alike, _1_0_alike = drawlines(img_undist1_alike,img_undist0_alike,epi_lines_1_alike,kp1_good_alike,kp0_good_alike)

  # Calculate distance measures - ALIKE
  dist0_alike= dist_measure(kp0_alike_hom, epi_lines_0_alike)
  dist1_alike = dist_measure(kp1_alike_hom, epi_lines_1_alike)
  sum_dist_alike = dist0_alike + dist1_alike

  return alike_matches_0_1, img_lines_0_1_alike, img_lines_1_0_alike, sum_dist_alike

# """**SPPN function**"""

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend, PointTracker

fe = SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                          nms_dist=4,
                          conf_thresh=0.15,
                          nn_thresh=0.7,
                          cuda=False)

tracker = PointTracker(5, nn_thresh=0.7)


#Create function to apply SPPN to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points.
def SPPN_pairs(img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):
  img_undist0_GR = cv.cvtColor(img_undist0, cv.COLOR_BGR2GRAY)
  img_undist1_GR = cv.cvtColor(img_undist1, cv.COLOR_BGR2GRAY)

  img_undist0_norm = img_undist0_GR.astype('float')/255.0
  img_undist1_norm = img_undist1_GR.astype('float')/255.0
  img_undist0_norm = img_undist0_norm.astype('float32')
  img_undist1_norm = img_undist1_norm.astype('float32')

  kp0_sppn, desc0_sppn, heatmap0 = fe.run(img_undist0_norm)
  kp1_sppn, desc1_sppn, heatmap1 = fe.run(img_undist1_norm)

  matches_sppn = tracker.nn_match_two_way(desc0_sppn, desc1_sppn, 0.7)

  # Reshape key points, matches
  kp0_sppn_t = kp0_sppn[0:2,:].T
  kp1_sppn_t = kp1_sppn[0:2,:].T
  matches_sppn_t = np.int_(matches_sppn[0:2,:].T)

  #Select matched key points 
  kp0_good_sppn = kp0_sppn_t[matches_sppn_t[:,0]]
  kp1_good_sppn = kp1_sppn_t[matches_sppn_t[:,1]]

  good_sppn = [[cv.DMatch(i, i, 0)] for i in range(0, len(kp0_good_sppn))]
  kp0_sppn_cv = [cv.KeyPoint(kp0_good_sppn[i,0],kp0_good_sppn[i,1], 1) for i in range(kp0_good_sppn.shape[0])]
  kp1_sppn_cv = [cv.KeyPoint(kp1_good_sppn[i,0],kp1_good_sppn[i,1], 1) for i in range(kp1_good_sppn.shape[0])]

  sppn_matches_0_1 = cv.drawMatchesKnn(img_undist0,kp0_sppn_cv,img_undist1,kp1_sppn_cv,good_sppn,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  #Exclude key points from zone of car's hood
  kp0_good_sppn, kp1_good_sppn = excl_hood(kp0_good_sppn, kp1_good_sppn)

  # Refinement of keypoints for SPPN
  kp0_good_sppn, kp1_good_sppn = apply_refinement(kp0_good_sppn, kp1_good_sppn, img_undist0, img_undist1)

  # Epipolar lines - SPPN
  epi_lines_0_sppn, epi_lines_1_sppn, kp0_sppn_hom, kp1_sppn_hom = calc_epi(kp0_good_sppn, kp1_good_sppn, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - SPPN
  img_undist0_sppn = copy.deepcopy(img_undist0)
  img_undist1_sppn = copy.deepcopy(img_undist1)

  img_lines_0_1_sppn, _0_1_sppn = drawlines(img_undist0_sppn,img_undist1_sppn,epi_lines_0_sppn,kp0_good_sppn,kp1_good_sppn)
  img_lines_1_0_sppn, _1_0_sppn = drawlines(img_undist1_sppn,img_undist0_sppn,epi_lines_1_sppn,kp1_good_sppn,kp0_good_sppn)

  # Calculate distance measures - SPPN
  dist0_sppn= dist_measure(kp0_sppn_hom, epi_lines_0_sppn)
  dist1_sppn = dist_measure(kp1_sppn_hom, epi_lines_1_sppn)
  sum_dist_sppn = dist0_sppn + dist1_sppn

  return sppn_matches_0_1, img_lines_0_1_sppn, img_lines_1_0_sppn, sum_dist_sppn

# """**ORB function**"""

# Initiate ORB detector
orb = cv.ORB_create()


#Create function to apply ORB to pairs of undirtorted images. Function returns distances from epipolar lines for the revealed key points.
def ORB_pairs(img_undist0, img_undist1, T_cam0_cam1_fund, T_cam1_cam0_fund):
  # find the keypoints with ORB
  kp0_orb = orb.detect(img_undist0,None)
  kp1_orb = orb.detect(img_undist1,None)

  # compute the descriptors with ORB
  kp0_orb, desc0_orb = orb.compute(img_undist0, kp0_orb)
  kp1_orb, desc1_orb = orb.compute(img_undist1, kp1_orb)

  matches_orb = bf.knnMatch(desc0_orb,desc1_orb,k=2)

  # Apply ratio test
  good_orb, good_ind0_orb, good_ind1_orb = ratio_test(matches_orb, 0.84)

  # Draw ORB matches
  orb_matches_0_1 = cv.drawMatchesKnn(img_undist0,kp0_orb,img_undist1,kp1_orb,good_orb,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  #Prepare Numpy arrays for good keypoints
  kp0_np_orb = cv.KeyPoint_convert(kp0_orb)
  kp1_np_orb = cv.KeyPoint_convert(kp1_orb)
  kp0_good_orb = kp0_np_orb[:,:][good_ind0_orb]
  kp1_good_orb = kp1_np_orb[:,:][good_ind1_orb]

  #Exclude key points from zone of car's hood
  kp0_good_orb, kp1_good_orb = excl_hood(kp0_good_orb, kp1_good_orb)

  # Refinement of keypoints for ORB
  kp0_good_orb, kp1_good_orb = apply_refinement(kp0_good_orb, kp1_good_orb, img_undist0, img_undist1)

  # Epipolar lines - ORB
  epi_lines_0_orb, epi_lines_1_orb, kp0_orb_hom, kp1_orb_hom = calc_epi(kp0_good_orb, kp1_good_orb, T_cam0_cam1_fund, T_cam1_cam0_fund)

  # Draw epipolar lines - ORB
  img_undist0_orb = copy.deepcopy(img_undist0)
  img_undist1_orb = copy.deepcopy(img_undist1)

  img_lines_0_1_orb, _0_1_orb = drawlines(img_undist0_orb,img_undist1_orb,epi_lines_0_orb,kp0_good_orb,kp1_good_orb)
  img_lines_1_0_orb, _1_0_orb = drawlines(img_undist1_orb,img_undist0_orb,epi_lines_1_orb,kp1_good_orb,kp0_good_orb)

  # Calculate distance measures - ORB
  dist0_orb = dist_measure(kp0_orb_hom, epi_lines_0_orb)
  dist1_orb = dist_measure(kp1_orb_hom, epi_lines_1_orb)
  sum_dist_orb = dist0_orb + dist1_orb

  return orb_matches_0_1, img_lines_0_1_orb, img_lines_1_0_orb, sum_dist_orb

# """**Processing dataset**"""

#Create empty lists for storing results for different methods (except r2d2 for which calculations are made above to avoid collision of versions)
sift_matches_0_1_all, img_lines_0_1_sift_all, img_lines_1_0_sift_all = [], [], []
rlof_matches_0_1_all, img_lines_0_1_rlof_all, img_lines_1_0_rlof_all = [], [], []
alike_matches_0_1_all, img_lines_0_1_alike_all, img_lines_1_0_alike_all = [], [], []
sppn_matches_0_1_all, img_lines_0_1_sppn_all, img_lines_1_0_sppn_all = [], [], []
orb_matches_all, img_lines_0_1_orb_all, img_lines_1_0_orb_all = [], [], []

sum_dist_sift_all = np.empty(shape=[0, 1])
sum_dist_rlof_all = np.empty(shape=[0, 1])
sum_dist_alike_all = np.empty(shape=[0, 1])
sum_dist_sppn_all = np.empty(shape=[0, 1])
sum_dist_orb_all = np.empty(shape=[0, 1])

#Process images of dataset sequentially in pairs using all methods
for i in range(num_images-1):
  sift_matches_0_1, img_lines_0_1_sift, img_lines_1_0_sift, sum_dist_sift, kp0_good = SIFT_pairs(globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)
  rlof_matches_0_1, img_lines_0_1_rlof, img_lines_1_0_rlof, sum_dist_rlof = RLOF_pairs(kp0_good, globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)
  alike_matches_0_1, img_lines_0_1_alike, img_lines_1_0_alike, sum_dist_alike = ALIKE_pairs(globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)
  sppn_matches_0_1, img_lines_0_1_sppn, img_lines_1_0_sppn, sum_dist_sppn = SPPN_pairs(globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)
  orb_matches_0_1, img_lines_0_1_orb, img_lines_1_0_orb, sum_dist_orb = ORB_pairs(globals()['img_undist%s'%i], globals()['img_undist%s'%(i+1)],
                                                                                                          T_fund_list[i], T_fund_list[i].T)

  sift_matches_0_1_all.append(sift_matches_0_1)
  img_lines_0_1_sift_all.append(img_lines_0_1_sift)
  img_lines_1_0_sift_all.append(img_lines_1_0_sift)
  sum_dist_sift_all = np.append(sum_dist_sift_all, sum_dist_sift, axis=0)
  
  rlof_matches_0_1_all.append(rlof_matches_0_1)
  img_lines_0_1_rlof_all.append(img_lines_0_1_rlof)
  img_lines_1_0_rlof_all.append(img_lines_1_0_rlof)
  sum_dist_rlof_all = np.append(sum_dist_rlof_all, sum_dist_rlof, axis=0)
  
  alike_matches_0_1_all.append(alike_matches_0_1)
  img_lines_0_1_alike_all.append(img_lines_0_1_alike)
  img_lines_1_0_alike_all.append(img_lines_1_0_alike)
  sum_dist_alike_all = np.append(sum_dist_alike_all, sum_dist_alike, axis=0)
  
  sppn_matches_0_1_all.append(sppn_matches_0_1)
  img_lines_0_1_sppn_all.append(img_lines_0_1_sppn)
  img_lines_1_0_sppn_all.append(img_lines_1_0_sppn)
  sum_dist_sppn_all = np.append(sum_dist_sppn_all, sum_dist_sppn, axis=0)
  
  orb_matches_all.append(orb_matches_0_1)
  img_lines_0_1_orb_all.append(img_lines_0_1_orb)
  img_lines_1_0_orb_all.append(img_lines_1_0_orb)
  sum_dist_orb_all = np.append(sum_dist_orb_all, sum_dist_orb, axis=0)
  

# Calculate average and mean distances across the dataset for each method 
median_dist_sift = np.median(sum_dist_sift_all)
ave_dist_sift = np.average(sum_dist_sift_all)
print('SIFT')
print(median_dist_sift)
print(ave_dist_sift)

print('r2d2')
print(median_dist_r2d2)
print(ave_dist_r2d2)

median_dist_rlof = np.median(sum_dist_rlof_all)
ave_dist_rlof = np.average(sum_dist_rlof_all)
print('rlof')
print(median_dist_rlof)
print(ave_dist_rlof)

median_dist_alike = np.median(sum_dist_alike_all)
ave_dist_alike = np.average(sum_dist_alike_all)
print('alike')
print(median_dist_alike)
print(ave_dist_alike)

median_dist_sppn = np.median(sum_dist_sppn_all)
ave_dist_sppn = np.average(sum_dist_sppn_all)
print('sppn')
print(median_dist_sppn)
print(ave_dist_sppn)

median_dist_orb = np.median(sum_dist_orb_all)
ave_dist_orb = np.average(sum_dist_orb_all)
print('orb')
print(median_dist_orb)
print(ave_dist_orb)
