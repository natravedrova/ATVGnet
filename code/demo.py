import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import librosa
from models import AT_net, AT_single
from models import VG_net 
import cv2
import scipy.misc
import utils
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes

import dlib

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--lstm",
                     default=True)
    parser.add_argument("--vg_model",
                     type=str,
                     default="../model/generator_23.pth")
    parser.add_argument("--at_model",
                     type=str,
                     # default="/u/lchen63/lrw/model/audio2lmark_pca/audio2lmark_24.pth")
                     default="../model/atnet_lstm_18.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="../results")
                    # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')test
    parser.add_argument('-i','--in_file', type=str, default='../audio/test.wav')
    parser.add_argument('-d','--data_path', type=str, default='../basics')
    parser.add_argument('-p','--person', type=str, default='../image/musk1.jpg')
    parser.add_argument('--device_ids', type=str, default='2')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--faceid', type=int, default=0)
    parser.add_argument('--headonly', type=int, default=0)
    parser.add_argument('--seq', type=int, default=0)
    return parser.parse_args()
config = parse_args()

ms_img = np.load('../basics/mean_shape_img.npy')
ms_norm = np.load('../basics/mean_shape_norm.npy')
S = np.load('../basics/S.npy')

MSK = np.reshape(ms_norm, [1, 68*2])
SK = np.reshape(S, [1, S.shape[0], 68*2])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')
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

PREV_FACE_RECT=[0,0]
FACE_DIST = 20
def get_landmarks(im,org_rect,id):
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    global PREV_FACE_RECT
    #im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,im.shape[0] * SCALE_FACTOR))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if id==0 and PREV_FACE_RECT==[0,0] :
        PREV_FACE_RECT=[[p.x, p.y] for p in predictor(im, org_rect).parts()][0]
    if len(rects) > 1:
        print('more than 1 face detected! %s' % len(rects))
    if len(rects) == 0:
        raise NoFaces
    for idx,rect in enumerate(rects):
        ldmark=[[p.x, p.y] for p in predictor(im, rect).parts()]
        if id==0:
            xx=[PREV_FACE_RECT[0]-ldmark[0][0],PREV_FACE_RECT[1]-ldmark[0][1]]
            dist=np.linalg.norm(xx)
            if dist>FACE_DIST:
                print('rects too far %s' % dist)
                continue
            PREV_FACE_RECT=ldmark[0]
        return im,np.matrix(ldmark)

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
def restore_image(orgImage,rect, img , idx):
    #(x, y, w, h) = utils.rect_to_bb(rect)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #rgb = cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)
    #orgImage[y:y + h, x:x+w]=rgb
    im1, landmarks1 = get_landmarks(orgImage,rect,0)
    ovimg = img.astype('uint8')
    im2, landmarks2 = get_landmarks(ovimg,rect,1)
    M = transformation_from_points(landmarks1[ALIGN_POINTS],landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)

    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    print(landmarks1)
    out=get_face_mask(im1, landmarks1)*255+im1
    cv2.imwrite("../temp/mask/maskr_{:04d}.png".format(idx),out)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return output_im
#...........................................
#...........................................

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def normLmarks(lmarks):
    norm_list = []
    idx = -1
    max_openness = 0.2
    mouthParams = np.zeros((1, 100))
    mouthParams[:, 1] = -0.06
    tmp = deepcopy(MSK)
    tmp[:, 48*2:] += np.dot(mouthParams, SK)[0, :, 48*2:]
    open_mouth_params = np.reshape(np.dot(S, tmp[0, :] - MSK[0, :]), (1, 100))
    if len(lmarks.shape) == 2:
        lmarks = lmarks.reshape(1,68,2)
    for i in range(lmarks.shape[0]):
        mtx1, mtx2, disparity = procrustes(ms_img, lmarks[i, :, :])
        mtx1 = np.reshape(mtx1, [1, 136])
        mtx2 = np.reshape(mtx2, [1, 136])
        norm_list.append(mtx2[0, :])
    pred_seq = []
    init_params = np.reshape(np.dot(S, norm_list[idx] - mtx1[0, :]), (1, 100))
    for i in range(lmarks.shape[0]):
        params = np.reshape(np.dot(S, norm_list[i] - mtx1[0, :]), (1, 100)) - init_params - open_mouth_params
        predicted = np.dot(params, SK)[0, :, :] + MSK
        pred_seq.append(predicted[0, :])
    return np.array(pred_seq), np.array(norm_list), 1
    
def getImageInfo(image_path):
    print(image_path)
    image= cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    print('%s face detected!' % len(rects))
    #for (i, rect) in enumerate(rects):
    #(x, y, w, h) = utils.rect_to_bb(rect)
    return image,rects[config.faceid]
 
def crop_image2(image,rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    shape = utils.shape_to_np(shape)
    (x, y, w, h) = utils.rect_to_bb(rect)
    center_x = x + int(0.5 * w)
    center_y = y + int(0.5 * h)
    r = int(0.64 * h)
    new_x = center_x - r
    new_y = center_y - r
    roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
    roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
    scale =  163. / (2 * r)

    shape = ((shape - np.array([new_x,new_y])) * scale)

    return roi, shape 

def crop_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        (x, y, w, h) = utils.rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)

        scale =  163. / (2 * r)

        shape = ((shape - np.array([new_x,new_y])) * scale)

        return roi, shape 
def generator_demo_example_lips(img_path,image,rect):
    name = img_path.split('/')[-1]
    name = name.split('.')[-1]
    landmark_path = os.path.join('../image/', name+'.npy') 
    region_path = os.path.join('../image/', name+ '_region.jpg') 
    roi, landmark= crop_image2(image,rect)

    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

        # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
        template = np.load( '../basics/base_68.npy')
    else:
        template = np.load( '../basics/base_68_close.npy')
    # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
    pts2 = np.float32(template[27:45,:])
    #pts2 = np.float32(template[0:0,:])
    # pts2 = np.float32(template[17:35,:])
    # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
    pts1 = np.float32(landmark[27:45,:])
    #pts1 = np.float32(landmark[0:0,:])
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))

    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129,1:129,:]
    #dst = cv2.resize(dst, (128,128))
    cv2.imwrite(region_path, dst)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    #rects = detector(gray, 1)
    shape = predictor(gray, rect)
    shape = utils.shape_to_np(shape)
    shape, _ ,_ = normLmarks(shape)
    np.save(landmark_path, shape)
    lmark= shape.reshape(68,2)
    name = region_path.replace('region.jpg','lmark.png')
    
    utils.plot_flmarks(lmark, name, (-0.2, 0.2), (-0.2, 0.2), 'x', 'y', figsize=(10, 10))
    return dst, lmark
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    if os.path.exists('../temp'):
        try:
            shutil.rmtree('../temp')
            while os.path.exists('../temp'):
                time.sleep(1)
        except:
            while os.path.exists('../temp'):
                time.sleep(1)
    try:
        os.mkdir('../temp')
        os.mkdir('../temp/img')
        os.mkdir('../temp/motion')
        os.mkdir('../temp/attention')
        os.mkdir('../temp/mask')
    except:
        print ('already have dir')
    pca = torch.FloatTensor( np.load('../basics/U_lrw1.npy')[:,:6]).cuda()
    mean =torch.FloatTensor( np.load('../basics/mean_lrw1.npy')).cuda()
    decoder = VG_net()
    encoder = AT_net()
    if config.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    state_dict2 = multi2single(config.vg_model, 1)

    # state_dict2 = torch.load(config.video_model, map_location=lambda storage, loc: storage)
    decoder.load_state_dict(state_dict2)

    state_dict = multi2single(config.at_model, 1)
    encoder.load_state_dict(state_dict)

    encoder.eval()
    decoder.eval()
    test_file = config.in_file
    orgPath=config.person
    if config.seq:
        orgPath = config.person.format(0)
    orgImage, rect = getImageInfo(orgPath)
    example_image, example_landmark = generator_demo_example_lips( orgPath,orgImage,rect)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])        
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    example_image = transform(example_image)
    example_landmark =  example_landmark.reshape((1,example_landmark.shape[0]* example_landmark.shape[1]))
    if config.cuda:
        example_image = Variable(example_image.view(1,3,128,128)).cuda()
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)) ).cuda()
    else:
        example_image = Variable(example_image.view(1,3,128,128))
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)))
    # Load speech and extract features
    example_landmark = example_landmark * 5.0
    example_landmark  = example_landmark - mean.expand_as(example_landmark)
    example_landmark = torch.mm(example_landmark,  pca)
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    sound, _ = librosa.load(test_file, sr=44100)

    print ('=======================================')
    print ('Start to generate images')
    t =time.time()
    ind = 3
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        fake_lmark = encoder(example_landmark, input_mfcc)
        fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 6)
        example_landmark  = torch.mm( example_landmark, pca.t() ) 
        example_landmark = example_landmark + mean.expand_as(example_landmark)
        fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])).cuda() 
        fake_lmark = torch.mm( fake_lmark, pca.t() )
        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
    
        fake_lmark = fake_lmark.unsqueeze(0) 

        fake_ims, atts ,ms ,_ = decoder(example_image, fake_lmark, example_landmark )
        
        fake_lmark = fake_lmark.data.cpu().numpy()
        #np.save( os.path.join( config.sample_dir,  'obama_fake.npy'), fake_lmark)
        fake_lmark = np.reshape(fake_lmark, (fake_lmark.shape[1], 68, 2))
        #utils.write_video_wpts_wsound(fake_lmark, sound, 44100, config.sample_dir, 'fake', [-1.0, 1.0], [-1.0, 1.0])
        imgIdx=0
        imgDir=1
        for indx in range(fake_ims.size(1)):
            fake_im = fake_ims[:,indx]
            fake_store = fake_im.permute(0,2,3,1).data.cpu().numpy()[0]
            if not config.headonly:
                if config.seq:
                    seqPath=config.person.format(imgIdx)
                    if not os.path.exists(seqPath):
                        imgDir=-imgDir
                        imgIdx+=imgDir
                        seqPath=config.person.format(imgIdx)
                    print(seqPath)
                    newImage=cv2.imread(seqPath)
                    imgIdx+=imgDir
                    fake_store = restore_image(newImage,rect,fake_store,indx)
                else:
                    fake_store = restore_image(orgImage,rect,fake_store,indx)
                cv2.imwrite("{}/{:05d}.png".format(os.path.join('../', 'temp', 'img') ,indx ), fake_store)
            else:
                scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'img') ,indx ), fake_store)
            m = ms[:,indx]
            att = atts[:,indx]
            m = m.permute(0,2,3,1).data.cpu().numpy()[0]
            att = att.data.cpu().numpy()[0,0]

            scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'motion' ) ,indx ), m)
            scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'attention') ,indx ), att)

        print ( 'In total, generate {:d} images, cost time: {:03f} seconds'.format(fake_ims.size(1), time.time() - t) )
            
        
        video_name = os.path.join(config.sample_dir , 'results.mp4')
        utils.image_to_video(os.path.join('../', 'temp', 'img'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'results_a.mp4')))
        #shutil.rmtree('../temp')

test()

