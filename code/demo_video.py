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
import ddfa
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
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--faceid', type=int, default=0)
    return parser.parse_args()
config = parse_args()

ms_img = np.load('../basics/mean_shape_img.npy')
ms_norm = np.load('../basics/mean_shape_norm.npy')
S = np.load('../basics/S.npy')

MSK = np.reshape(ms_norm, [1, 68*2])
SK = np.reshape(S, [1, S.shape[0], 68*2])


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
    
def crop_image2(image,rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = ddfa.predictor(gray, rect)
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
    cv2.imwrite(region_path, dst)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    #rects = detector(gray, 1)
    shape = ddfa.predictor(gray, rect)
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
            while os.path.exists('../temp/img'):
                time.sleep(1)
        except:
            while os.path.exists('../temp/img'):
                time.sleep(1)
    try:
        os.mkdir('../temp')
    except:
        print ('already have dir')
    os.mkdir('../temp/img')
    os.mkdir('../temp/motion')
    os.mkdir('../temp/attention')
    os.mkdir('../temp/mask')
    os.mkdir('../temp/lmk')
    os.mkdir('../temp/head')
    os.mkdir('../temp/headlmk')
    
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
    orgPath = config.person.format(0)
    #init ddfa and more
    ddfaModel,ddfaTrans=ddfa.init3DDFA()
    orgImage, rect = ddfa.getImageInfo(orgPath,config.faceid)
    prePts=np.array([[p.x, p.y] for p in ddfa.predictor(orgImage, rect).parts()]).T
    pre_ovPts=[]
    #............
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
        for indx in range(fake_ims.size(1)):
            fake_im = fake_ims[:,indx]
            fake_store = fake_im.permute(0,2,3,1).data.cpu().numpy()[0]
            seqPath=config.person.format(indx)
            if not os.path.exists(seqPath):
                seqPath=config.person.format(0)
            print(seqPath)
            newImage=cv2.imread(seqPath)
            headimg=fake_store.copy()
            cv2.normalize(headimg, headimg, 0, 255, cv2.NORM_MINMAX)
            headimg = headimg.astype('uint8')
            
            fake_store,prePts,pre_ovPts = ddfa.restore_image(newImage,prePts,headimg,pre_ovPts,indx,ddfaTrans,ddfaModel)
            headimg=cv2.cvtColor(headimg,cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/{:05d}.png".format(os.path.join('../', 'temp', 'head') ,indx ), headimg)
            ddfa.drawPoints(headimg,pre_ovPts.tolist())
            #ddfa.drawPoints(headimg,pre_ovPts.T)
            cv2.imwrite("{}/{:05d}.png".format(os.path.join('../', 'temp', 'headlmk') ,indx ), headimg)
            
            cv2.imwrite("{}/{:05d}.png".format(os.path.join('../', 'temp', 'img') ,indx ), fake_store)
            ddfa.drawPoints(newImage,prePts.T)
            cv2.imwrite("{}/{:05d}.jpg".format(os.path.join('../', 'temp', 'lmk') ,indx ),newImage)
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
        video_name = os.path.join(config.sample_dir , 'landmark.mp4')
        utils.image_to_video(os.path.join('../', 'temp', 'lmk'), video_name )
        
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'results_a.mp4')))
        #shutil.rmtree('../temp')

test()

