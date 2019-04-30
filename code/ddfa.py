import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
import torch.backends.cudnn as cudnn
import math
import os.path as osp
import pickle
import time
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color, 2)
        
#3DDFA Params
def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]
def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))
        
def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)
d = make_abs_path('../3DDFA/train.configs')
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
meta = _load(osp.join(d, 'param_whitening.pkl'))
# param_mean and param_std are used for re-whitening
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp
w = np.concatenate((w_shp, w_exp), axis=1)
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0)
w_base_norm = np.linalg.norm(w_base, axis=0)
# for inference
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120
#end 
def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp

def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex

def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box):
    return _predict_vertices(param, roi_box, dense=True)

    
def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box

def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res
    
class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor
#..................................................
#interface
def init3DDFA():
    checkpoint_fp = '../model/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    cudnn.benchmark = True
    model = model.cuda()
    model.eval()
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    return model,transform
def get_landmarks(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 1:
        print('more than 1 face detected! %s' % len(rects))
    if len(rects) == 0:
        cv2.imwrite("debug.jpg",gray)
        raise NoFaces
    for idx,rect in enumerate(rects):
        ldmark=[[p.x, p.y] for p in predictor(im, rect).parts()]
        return im,np.matrix(ldmark)
def get_landmarks3DDFA(img_ori,pts,transform,model,cropImg=True):
    if pts==[]:
        # - use landmark for cropping
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if len(rects)>0:
            #should be only one face here
            pts = predictor(img_ori, rects[0]).parts() 
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
        else:
            print("Error couldn't find any face!")
   
    roi_box = parse_roi_box_from_landmark(pts)
    if cropImg:
        img = crop_img(img_ori, roi_box)
    else:
        img = img_ori
    # forward: one step
    img = cv2.resize(img, dsize=(std_size, std_size), interpolation=cv2.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        input = input.cuda()
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
    # 68 pts
    pts68 = predict_68pts(param, roi_box)
    # two-step for more accurate bbox to crop face
    # if args.bbox_init == 'two':
        # roi_box = parse_roi_box_from_landmark(pts68)
        # img_step2 = crop_img(img_ori, roi_box)
        # img_step2 = cv2.resize(img_step2, dsize=(std_size, std_size), interpolation=cv2.INTER_LINEAR)
        # input = transform(img_step2).unsqueeze(0)
        # with torch.no_grad():
            # if args.mode == 'gpu':
                # input = input.cuda()
            # param = model(input)
            # param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # pts68 = predict_68pts(param, roi_box)
     #trim xyz to xy
    return img_ori,pts68
     
#endof 3DDFA

#faceswap feature
#...........................................
#...........................................

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
                               # Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    NOSE_POINTS + MOUTH_POINTS,
]
COLOUR_CORRECT_BLUR_FRAC = 0.6
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
                         
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))
def getImageInfo(image_path,faceid):
    print(image_path)
    image= cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    print('%s face detected!' % len(rects))
    #for (i, rect) in enumerate(rects):
    #(x, y, w, h) = utils.rect_to_bb(rect)
    return image,rects[faceid]
 
def restore_image(orgImage,prePts, ovimg ,pre_ovPts, idx,transform,model):
    im1, landmarks1 = get_landmarks3DDFA(orgImage,prePts,transform,model)
    #im2, landmarks2 = get_landmarks3DDFA(ovimg,pre_ovPts,transform,model)
    im2, landmarks2 = get_landmarks(ovimg)
    #convert to facewarp need
    drawMark1=np.matrix(landmarks1[:-1,:].T.astype(int))
    #drawMark2=np.matrix(landmarks2[:-1,:].T.astype(int))
    drawMark2=landmarks2
    
    M = transformation_from_points(drawMark1[ALIGN_POINTS],drawMark2[ALIGN_POINTS])
    mask = get_face_mask(im2, drawMark2)
    warped_mask = warp_im(mask, M, im1.shape)

    combined_mask = np.max([get_face_mask(im1, drawMark1), warped_mask],axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    out=get_face_mask(im1, drawMark1)*255+im1
    cv2.imwrite("../temp/mask/maskr_{:04d}.png".format(idx),out)
    warped_corrected_im2 = correct_colours(im1, warped_im2, drawMark1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im,landmarks1,landmarks2

#...........................................
#...........................................
