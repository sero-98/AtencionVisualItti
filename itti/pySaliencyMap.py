import cv2
import numpy as np
from itti import pySaliencyMapDefs
class pySaliencyMap:


    # INITIALIZATION
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.prev_frame = None
        self.SM = None
        self.GaborKernel0   = np.array(pySaliencyMapDefs.GaborKernel_0)
        self.GaborKernel45  = np.array(pySaliencyMapDefs.GaborKernel_45)
        self.GaborKernel90  = np.array(pySaliencyMapDefs.GaborKernel_90)
        self.GaborKernel135 = np.array(pySaliencyMapDefs.GaborKernel_135)



    # EXTRACTING COLOR CHANNELS   (*)
    def SMExtractRGBI(self, inputImage):
        src = np.float32(inputImage) * 1./255
        (B, G, R) = cv2.split(src)
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return R, G, B, I



    # FEATURE MAPS
    ## CONSTRUCTING A GAUSSIAN PYRAMID
    ## src = I
    def FMCreateGaussianPyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1,9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst



    ## TALKING CENTER-SURROUND DIFFERENCES
    ### (tomando las diferencias entre el centro y el entorno)
    def FMCenterSurroundDiff(self, GaussianMaps):
        dst = list()
        for s in range(2,5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0])  ## (width, height)
            tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)      
            tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst



    ## CONSTRUCTING A GAUSSIAN PYRAMID + TALKING CENTER-SURROUND DIFFERENCES 
    def FMGaussianPyrCSD(self, src):
        GaussianMaps = self.FMCreateGaussianPyr(src)
        dst = self.FMCenterSurroundDiff(GaussianMaps)
        return dst



    ## intensity feature maps            (*)
    def IFMGetFM(self, I):
        return self.FMGaussianPyrCSD(I)



    ## colors feature maps                (*)
    def CFMGetFM(self, R, G, B):

        # max(R,G,B)
        tmp1 = cv2.max(R, G)
        RGBMax = cv2.max(B, tmp1)
        RGBMax[RGBMax <= 0] = 0.0001    # prevent dividing by 0

        # min(R,G)
        RGMin = cv2.min(R, G)

        # RG = (R-G)/max(R,G,B)
        RG = (R - G) / RGBMax
        
        # BY = (B-min(R,G)/max(R,G,B)
        BY = (B - RGMin) / RGBMax

        # clamp nagative values to 0
        RG[RG < 0] = 0
        BY[BY < 0] = 0

        # obtain feature maps in the same way as intensity
        RGFM = self.FMGaussianPyrCSD(RG)
        BYFM = self.FMGaussianPyrCSD(BY)

        # return
        return RGFM, BYFM



    ## orientation feature maps          (*)
    def OFMGetFM(self, src):
        # creating a Gaussian pyramid
        GaussianI = self.FMCreateGaussianPyr(src)
        # convoluting a Gabor filter with an intensity image to extract oriemtation features
        GaborOutput0   = [ np.empty((1,1)), np.empty((1,1)) ]  # dummy data: any kinds of np.array()s are OK
        GaborOutput45  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput90  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput135 = [ np.empty((1,1)), np.empty((1,1)) ]
        for j in range(2,9):
            GaborOutput0.append(   cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel0) )
            GaborOutput45.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel45) )
            GaborOutput90.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel90) )
            GaborOutput135.append( cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel135) )
        # calculating center-surround differences for every oriantation
        CSD0   = self.FMCenterSurroundDiff(GaborOutput0)
        CSD45  = self.FMCenterSurroundDiff(GaborOutput45)
        CSD90  = self.FMCenterSurroundDiff(GaborOutput90)
        CSD135 = self.FMCenterSurroundDiff(GaborOutput135)
        # concatenate
        dst = list(CSD0)
        dst.extend(CSD45)
        dst.extend(CSD90)
        dst.extend(CSD135)
        # return
        return dst



    ## motion feature maps      (*)
    def MFMGetFM(self, src):
        # convert scale
        I8U = np.uint8(255 * src)
        cv2.waitKey(10)
        # calculating optical flows
        if self.prev_frame is not None:
            farne_pyr_scale= pySaliencyMapDefs.farne_pyr_scale
            farne_levels = pySaliencyMapDefs.farne_levels
            farne_winsize = pySaliencyMapDefs.farne_winsize
            farne_iterations = pySaliencyMapDefs.farne_iterations
            farne_poly_n = pySaliencyMapDefs.farne_poly_n
            farne_poly_sigma = pySaliencyMapDefs.farne_poly_sigma
            farne_flags = pySaliencyMapDefs.farne_flags
            flow = cv2.calcOpticalFlowFarneback(\
                prev = self.prev_frame, \
                next = I8U, \
                pyr_scale = farne_pyr_scale, \
                levels = farne_levels, \
                winsize = farne_winsize, \
                iterations = farne_iterations, \
                poly_n = farne_poly_n, \
                poly_sigma = farne_poly_sigma, \
                flags = farne_flags, \
                flow = None \
            )
            flowx = flow[...,0]
            flowy = flow[...,1]
        else:
            flowx = np.zeros(I8U.shape)
            flowy = np.zeros(I8U.shape)
        # create Gaussian pyramids
        dst_x = self.FMGaussianPyrCSD(flowx)
        dst_y = self.FMGaussianPyrCSD(flowy)
        # update the current frame
        self.prev_frame = np.uint8(I8U)
        # return
        return dst_x, dst_y



    # conspicuity maps
    ## standard range normalization  (*)
    def SMRangeNormalize(self, src):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src)
        if maxx!=minn:
            dst = src/(maxx-minn) + minn/(minn-maxx)
        else:
            dst = src - minn
        return dst



    ## computing an average of local maxima
    def SMAvgLocalMax(self, src):
        # size
        stepsize = pySaliencyMapDefs.default_step_local
        width = src.shape[1]
        height = src.shape[0]
        # find local maxima
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[y:y+stepsize, x:x+stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        # averaging over all the local regions
        return lmaxmean / numlocal



    ## normalization specific for the saliency map model
    def SMNormalization(self, src):
        dst = self.SMRangeNormalize(src)
        lmaxmean = self.SMAvgLocalMax(dst)
        normcoeff = (1-lmaxmean)*(1-lmaxmean)
        return dst * normcoeff



    ## normalizing feature maps
    def normalizeFeatureMaps(self, FM):
        NFM = list()
        for i in range(0,6):
            normalizedImage = self.SMNormalization(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            NFM.append(nownfm)
        return NFM



    ## intensity conspicuity map    (*)
    def ICMGetCM(self, IFM):
        NIFM = self.normalizeFeatureMaps(IFM)
        ICM = sum(NIFM)
        return ICM



    ## color conspicuity map        (*)
    def CCMGetCM(self, CFM_RG, CFM_BY):
        # extracting a conspicuity map for every color opponent pair
        CCM_RG = self.ICMGetCM(CFM_RG)
        CCM_BY = self.ICMGetCM(CFM_BY)
        # merge
        CCM = CCM_RG + CCM_BY
        # return
        return CCM



    ## orientation conspicuity map   (*)
    def OCMGetCM(self, OFM):
        OCM = np.zeros((self.height, self.width))
        for i in range (0,4):
            # slicing
            nowofm = OFM[i*6:(i+1)*6]  # angle = i*45
            # extracting a conspicuity map for every angle
            NOFM = self.ICMGetCM(nowofm)
            # normalize
            NOFM2 = self.SMNormalization(NOFM)
            # accumulate
            OCM += NOFM2
        return OCM



    ## motion conspicuity map        (*)
    def MCMGetCM(self, MFM_X, MFM_Y):   
        return self.CCMGetCM(MFM_X, MFM_Y)



###########################################################
###########################################################
###########################################################

    # promedio de una matriz
    def promedio(self, matriz, i):
        elementos = 0
        sumatoria = 0
        for fila in matriz:
            for elemento in fila:
                sumatoria += elemento
                elementos += 1
        promedio = sumatoria / elementos

        return promedio

    # segmentacion de una matriz
    def segmetacion(self, matriz):
        t1size = matriz.shape
        #print(t1size)

        t1_columnas  = t1size[1] 
        t1_filas = t1size[0] 

        t1_filas_bordes = int(t1_filas/4)
        t1_columnas_bordes = int(t1_columnas/3)

        datos1 = self.SM[0:t1_filas_bordes,0:t1_columnas_bordes]
        datos2 = self.SM[0:t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]
        datos3 = self.SM[0:t1_filas_bordes,2*t1_columnas_bordes:4*t1_columnas_bordes]

        datos4 = self.SM[t1_filas_bordes:2*t1_filas_bordes,0:t1_columnas_bordes]
        datos5 = self.SM[t1_filas_bordes:2*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]
        datos6 = self.SM[t1_filas_bordes:2*t1_filas_bordes,2*t1_columnas_bordes:4*t1_columnas_bordes]

        datos7 = self.SM[2*t1_filas_bordes:3*t1_filas_bordes,0:t1_columnas_bordes]
        datos8 = self.SM[2*t1_filas_bordes:3*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]
        datos9 = self.SM[2*t1_filas_bordes:3*t1_filas_bordes,2*t1_columnas_bordes:4*t1_columnas_bordes]

        datos10 = self.SM[3*t1_filas_bordes:4*t1_filas_bordes,0:t1_columnas_bordes]
        datos11 = self.SM[3*t1_filas_bordes:4*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]
        datos12 = self.SM[3*t1_filas_bordes:4*t1_filas_bordes,2*t1_columnas_bordes:4*t1_columnas_bordes]


        #print("...")

        my_list = []
        my_list.append(self.promedio(datos1, 1))
        my_list.append(self.promedio(datos2, 2))
        my_list.append(self.promedio(datos3, 3))
        my_list.append(self.promedio(datos4, 4))
        my_list.append(self.promedio(datos5, 5))
        my_list.append(self.promedio(datos6, 6))
        my_list.append(self.promedio(datos7, 7))
        my_list.append(self.promedio(datos8, 8))
        my_list.append(self.promedio(datos9, 9))
        my_list.append(self.promedio(datos10, 10))
        my_list.append(self.promedio(datos11, 11))
        my_list.append(self.promedio(datos12, 12))

        StrA = "-".join([str(_) for _ in my_list])
        
        return StrA

    def segmetacionAbajo(self, matriz): 
        t1size = matriz.shape
        print(t1size)

        t1_columnas  = t1size[1] 
        t1_filas = t1size[0] 

        t1_filas_bordes = int(t1_filas/5) #4
        t1_columnas_bordes = int(t1_columnas/2) #3

        datos1 = self.SM[0:t1_filas_bordes,0:t1_columnas_bordes]
        datos2 = self.SM[0:t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]

        datos3 = self.SM[t1_filas_bordes:2*t1_filas_bordes,0:t1_columnas_bordes]
        datos4 = self.SM[t1_filas_bordes:2*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]

        datos5 = self.SM[2*t1_filas_bordes:3*t1_filas_bordes,0:t1_columnas_bordes]
        datos6 = self.SM[2*t1_filas_bordes:3*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]

        datos7 = self.SM[3*t1_filas_bordes:4*t1_filas_bordes,0:t1_columnas_bordes]
        datos8 = self.SM[3*t1_filas_bordes:4*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]

        datos9 = self.SM[4*t1_filas_bordes:5*t1_filas_bordes,0:t1_columnas_bordes]
        datos10 = self.SM[4*t1_filas_bordes:5*t1_filas_bordes,t1_columnas_bordes:2*t1_columnas_bordes]

        my_list = []
        my_list.append(self.promedio(datos1, 1))
        my_list.append(self.promedio(datos2, 2))
        my_list.append(self.promedio(datos3, 3))
        my_list.append(self.promedio(datos4, 4))
        my_list.append(self.promedio(datos5, 5))
        my_list.append(self.promedio(datos6, 6))
        my_list.append(self.promedio(datos7, 7))
        my_list.append(self.promedio(datos8, 8))
        my_list.append(self.promedio(datos9, 9))
        my_list.append(self.promedio(datos10, 10))

        StrAbajo = "--".join([str(_) for _ in my_list])

        return StrAbajo

    
    def segmetacionFull(self, matriz):    
        t1size = matriz.shape
        print(t1size)

        t1_columnas  = t1size[1] 
        t1_filas = t1size[0] 

        t1_filas_arriba = int(t1_filas/8) #5 f1
        t1_columnas_arriba = int(t1_columnas/3) #2 c1

        t1_filas_abajo = int(t1_filas/10) #4 f2
        t1_columnas_abajo = int(t1_columnas/2) #3 c2

        datos1 = self.SM[0:t1_filas_arriba,0:t1_columnas_arriba]
        datos2 = self.SM[0:t1_filas_arriba,t1_columnas_arriba:2*t1_columnas_arriba]
        datos3 = self.SM[0:t1_filas_arriba,2*t1_columnas_arriba:3*t1_columnas_arriba]

        datos4 = self.SM[t1_filas_arriba:2*t1_filas_arriba,0:t1_columnas_arriba]
        datos5 = self.SM[t1_filas_arriba:2*t1_filas_arriba,t1_columnas_arriba:2*t1_columnas_arriba]
        datos6 = self.SM[t1_filas_arriba:2*t1_filas_arriba,2*t1_columnas_arriba:3*t1_columnas_arriba]

        datos7 = self.SM[2*t1_filas_arriba:3*t1_filas_arriba,0:t1_columnas_arriba]
        datos8 = self.SM[2*t1_filas_arriba:3*t1_filas_arriba,t1_columnas_arriba:2*t1_columnas_arriba]
        datos9 = self.SM[2*t1_filas_arriba:3*t1_filas_arriba,2*t1_columnas_arriba:3*t1_columnas_arriba]

        datos10 = self.SM[3*t1_filas_arriba:4*t1_filas_arriba,0:t1_columnas_arriba]
        datos11 = self.SM[3*t1_filas_arriba:4*t1_filas_arriba,t1_columnas_arriba:2*t1_columnas_arriba]
        datos12 = self.SM[3*t1_filas_arriba:4*t1_filas_arriba,2*t1_columnas_arriba:3*t1_columnas_arriba]


        datos13 = self.SM[4*t1_filas_arriba:6*t1_filas_abajo,0:t1_columnas_abajo]
        datos14 = self.SM[4*t1_filas_arriba:6*t1_filas_abajo,t1_columnas_abajo:2*t1_columnas_abajo]

        datos15 = self.SM[6*t1_filas_abajo:7*t1_filas_abajo,0:t1_columnas_abajo]
        datos16 = self.SM[6*t1_filas_abajo:7*t1_filas_abajo,t1_columnas_abajo:2*t1_columnas_abajo]

        datos17 = self.SM[7*t1_filas_abajo:8*t1_filas_abajo,0:t1_columnas_abajo]
        datos18 = self.SM[7*t1_filas_abajo:8*t1_filas_abajo,t1_columnas_abajo:2*t1_columnas_abajo]

        datos19 = self.SM[8*t1_filas_abajo:9*t1_filas_abajo,0:t1_columnas_abajo]
        datos20 = self.SM[8*t1_filas_abajo:9*t1_filas_abajo,t1_columnas_abajo:2*t1_columnas_abajo]

        datos21 = self.SM[9*t1_filas_abajo:10*t1_filas_abajo,0:t1_columnas_abajo]
        datos22 = self.SM[9*t1_filas_abajo:10*t1_filas_abajo,t1_columnas_abajo:2*t1_columnas_abajo]

        my_list = []
        my_list.append(self.promedio(datos1, 1))
        my_list.append(self.promedio(datos2, 2))
        my_list.append(self.promedio(datos3, 3))
        my_list.append(self.promedio(datos4, 4))
        my_list.append(self.promedio(datos5, 5))
        my_list.append(self.promedio(datos6, 6))
        my_list.append(self.promedio(datos7, 7))
        my_list.append(self.promedio(datos8, 8))
        my_list.append(self.promedio(datos9, 9))
        my_list.append(self.promedio(datos10, 10))
        my_list.append(self.promedio(datos11, 11))
        my_list.append(self.promedio(datos12, 12))

        my_list.append(self.promedio(datos13, 13))
        my_list.append(self.promedio(datos14, 14))
        my_list.append(self.promedio(datos15, 15))
        my_list.append(self.promedio(datos16, 16))
        my_list.append(self.promedio(datos17, 17))
        my_list.append(self.promedio(datos18, 18))
        my_list.append(self.promedio(datos19, 19))
        my_list.append(self.promedio(datos20, 20))
        my_list.append(self.promedio(datos21, 21))
        my_list.append(self.promedio(datos22, 22))


        StrFull = "-".join([str(_) for _ in my_list])

        return StrFull


    # core
    def SMGetSM_Arriba(self, src):
        # definitions
        size = src.shape
        width  = size[1]
        height = size[0]

        # extracting individual color channels
        R, G, B, I = self.SMExtractRGBI(src)

        # extracting feature maps
        IFM = self.IFMGetFM(I)
        CFM_RG, CFM_BY = self.CFMGetFM(R, G, B)
        OFM = self.OFMGetFM(I)
        MFM_X, MFM_Y = self.MFMGetFM(I)

        # extracting conspicuity maps
        ICM = self.ICMGetCM(IFM)
        CCM = self.CCMGetCM(CFM_RG, CFM_BY)
        OCM = self.OCMGetCM(OFM)
        MCM = self.MCMGetCM(MFM_X, MFM_Y)

        # adding all the conspicuity maps to form a saliency map
        wi = pySaliencyMapDefs.weight_intensity
        wc = pySaliencyMapDefs.weight_color
        wo = pySaliencyMapDefs.weight_orientation
        SMMat = wi*ICM + wc*CCM + wo*OCM

        normalizedSM = self.SMRangeNormalize(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)
        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)

        segmentacion = self.segmetacion(self.SM)

        # return
        return segmentacion

    def SMGetSM_Abajo(self, src):
        # definitions
        size = src.shape
        width  = size[1]
        height = size[0]
        # check
#        if(width != self.width or height != self.height):
#            sys.exit("size mismatch")

        # extracting individual color channels
        R, G, B, I = self.SMExtractRGBI(src)

        # extracting feature maps
        IFM = self.IFMGetFM(I)
        CFM_RG, CFM_BY = self.CFMGetFM(R, G, B)
        OFM = self.OFMGetFM(I)
        MFM_X, MFM_Y = self.MFMGetFM(I)

        # extracting conspicuity maps
        ICM = self.ICMGetCM(IFM)
        CCM = self.CCMGetCM(CFM_RG, CFM_BY)
        OCM = self.OCMGetCM(OFM)
        MCM = self.MCMGetCM(MFM_X, MFM_Y)

        # adding all the conspicuity maps to form a saliency map
        wi = pySaliencyMapDefs.weight_intensity
        wc = pySaliencyMapDefs.weight_color
        wo = pySaliencyMapDefs.weight_orientation
        #wm = pySaliencyMapDefs.weight_motion
        #SMMat = wi*ICM + wc*CCM + wo*OCM + wm*MCM
        SMMat = wi*ICM + wc*CCM + wo*OCM

        normalizedSM = self.SMRangeNormalize(SMMat)

        normalizedSM2 = normalizedSM.astype(np.float32)

        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)

        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        

        segmentacion = self.segmetacionAbajo(self.SM)
        print("########################")
        print("segmentacion", segmentacion)
        print("########################")

        # return
        return segmentacion

    def SMGetSM_Full(self, src):
        # definitions
        size = src.shape
        width  = size[1]
        height = size[0]
        # check
#        if(width != self.width or height != self.height):
#            sys.exit("size mismatch")

        # extracting individual color channels
        R, G, B, I = self.SMExtractRGBI(src)

        # extracting feature maps
        IFM = self.IFMGetFM(I)
        CFM_RG, CFM_BY = self.CFMGetFM(R, G, B)
        OFM = self.OFMGetFM(I)
        MFM_X, MFM_Y = self.MFMGetFM(I)

        # extracting conspicuity maps
        ICM = self.ICMGetCM(IFM)
        CCM = self.CCMGetCM(CFM_RG, CFM_BY)
        OCM = self.OCMGetCM(OFM)
        MCM = self.MCMGetCM(MFM_X, MFM_Y)

        # adding all the conspicuity maps to form a saliency map
        wi = pySaliencyMapDefs.weight_intensity
        wc = pySaliencyMapDefs.weight_color
        wo = pySaliencyMapDefs.weight_orientation
        #wm = pySaliencyMapDefs.weight_motion
        #SMMat = wi*ICM + wc*CCM + wo*OCM + wm*MCM
        SMMat = wi*ICM + wc*CCM + wo*OCM

        normalizedSM = self.SMRangeNormalize(SMMat)

        normalizedSM2 = normalizedSM.astype(np.float32)

        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)

        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        

        segmentacion = self.segmetacionFull(self.SM)
        print("########################")
        print("segmentacion", segmentacion)
        print("########################")

        # return
        return segmentacion

