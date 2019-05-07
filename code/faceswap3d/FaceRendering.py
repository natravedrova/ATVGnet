import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
excludeFace=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,47,90,
93,96,101,102,103,104,105,107,108,109,111,112,113,114,115,117,118,119,120,121,123,
124,125,127,128,129,130,131,133,134,135,137,138,139,140,141,147,148,149,150,151,152,153,
154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174]

meshExtra=np.array([[84,82,89],[84,87,82],[40,87,84],[83,87,40],[40,81,87],[83,81,40],[88,81,83]])

def setOrtho(w, h):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1000, 1000)
    glMatrixMode(GL_MODELVIEW)

def addTexture(img):
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId) 
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, img)
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) 
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) 
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    return textureId

class FaceRenderer:
    def __init__(self, targetImg, textureImg, textureCoords, mesh):
        self.h = targetImg.shape[0]
        self.w = targetImg.shape[1]

        pygame.init()
        pygame.display.set_mode((self.w, self.h), DOUBLEBUF|OPENGL)
        setOrtho(self.w, self.h)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D) 

        self.textureCoords = textureCoords
        self.textureCoords[0, :] /= textureImg.shape[1] 
        self.textureCoords[1, :] /= textureImg.shape[0]

        self.faceTexture = addTexture(textureImg)
        self.renderTexture = addTexture(targetImg)

        self.mesh = mesh

    def drawFace(self, vertices):
        glBindTexture(GL_TEXTURE_2D, self.faceTexture) 

        glBegin(GL_TRIANGLES)
        for i,triangle in enumerate(self.mesh):
            if i in excludeFace:
                continue    
            for vertex in triangle:
                glTexCoord2fv(self.textureCoords[:, vertex])
                glVertex3fv(vertices[:, vertex])
            
        glEnd()

    def render(self, vertices):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.drawFace(vertices)

        data = glReadPixels(0, 0, self.w, self.h, GL_BGR, GL_UNSIGNED_BYTE)
        renderedImg = np.fromstring(data, dtype=np.uint8)
        renderedImg = renderedImg.reshape((self.h, self.w, 3))
        for i in range(renderedImg.shape[2]):
            renderedImg[:, :, i] = np.flipud(renderedImg[:, :, i])

        pygame.display.flip()
        return renderedImg
    def setFaceTexture(self,textureImg):
        self.faceTexture = addTexture(textureImg)