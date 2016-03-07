#!/usr/bin/python
from __future__ import print_function, division # py3 compatibility
import numpy as np

import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.expanduser('~/whales/scripts/signal-processing/')))
import signalTools_beta as sT

sys.path.append(os.path.abspath(os.path.expanduser('~/whales/scripts/orchive/')))
import MLtools_beta as myML
import aupTextFile2mtlFile as a2m # annotation tools

from sklearn import cross_validation
from sklearn import svm

parser = argparse.ArgumentParser(description = 'trains-test-predict with a SVC')#, add_help=False)
                
## IN FILES                
parser.add_argument("-trW", "--trainWav", type=str, required=True, 
                    help = "wav file for training (required)")
                    
parser.add_argument("-trA","--trainAnn", type=str, required=True,
                    help = "txt annotations of the trainWav. "
                    "audacity txt annotatoins. Gaps are filled with 'b' (required)")   

parser.add_argument("-teW", "--testWav", type=str, nargs='*', 
                    help = "wav file for training")

parser.add_argument("-teA", "--testAnn", type=str, nargs='*',
                    help = "txt annotations of the trainWav")   
                    
parser.add_argument("-pW", "--predictWav", type=str, nargs='*',
                    help = "wav file for prediction")    
                    
parser.add_argument("-trWC", "--trainColl", type=str, nargs='*',
                    help = "text file with the paths to the wav and the corresponding"
                    "annotations file in the same line")                      


																				
## CLF PARAMETERS

clfArgs = parser.add_argument_group('clfArgs', 'SVM classifier parameters')

                    
clfArgs.add_argument("-C", "--SVCerrorC", type=float,  default=1, 
                    help = "error parameter of the SVC")                     
                    
clfArgs.add_argument("-cw", "--classWeight", default=None,
                    help = "weight for the classifier clases,"
                    "'auto' : automatically adjust weights inversely"
                    "proportional to class frequencies") 

clfArgs.add_argument("-tSz", "--testSize", type=float, default=0.6,
                    help = "size (percent) of the test set to be used to train "
                    "the SVC") 
                    
## FEATURES PARAMETERS
                    
featureArgs = parser.add_argument_group('featureArgs', 'feature extraction parameters')
                    
featureArgs.add_argument("-tw", "--textureWinSz", type=float, default=0.2,
                         help = "Size of the texture window [s]")                     

featureArgs.add_argument('-fty', '--featureType', default = 'cepstral', type=str,
                         help='type of festures to extract [spectral, cepstral, chroma]')
featureArgs.add_argument('-fp', '--featureExtrationParams',
                         type=str, nargs='*',
                         help="parameters for the feature extractio.'"
                        "ex: NCEPS' 13 'NFFTpower' '10' "
                        "! Depend on the feature extractor")
                        
## MORE OUTPUTS
outArgs = parser.add_argument_group('outArgs', 'output file options and parameters')                   
                        
outArgs.add_argument("-sF", "--summaryFi", dest='summaryFile', action='store_true',
                    help = "append feature extraction and clf details to the summary file")
outArgs.add_argument("-no-sF", "--no-summaryFi", dest='summaryFile', action='store_false',
                    help="don't append to summary file")
outArgs.set_defaults(summaryFile=True) # default uses Heikes call names
						
outArgs.add_argument("-ik", "--iterKey", type=str, default=(None, None), nargs=2,
                    help = "iteration key <parameter_name>, <value>"
						"This is ment when several test are executed iteratively"
                         "to anlaise the behaviour of the classifier at different"
                         "values of a parameter. An output file is automatically created"
                         "in the working directory where the scores are appended"
						"Ex: -ik NFFTpower 9") 
						
outArgs.add_argument("-wD", "--workingDir", type=str, default='',
                    help = "directory where output is sent. Defaults sets the cwd") 
                    
outArgs.add_argument("-pl","--plots", dest='plots', action='store_true',
                    help='plot the features, ground thruth and predictions for '
                    'train and test sets')
outArgs.add_argument("-no-pl", "--no-plots", dest='plots', action='store_false',
                    help="don't create plots")
outArgs.set_defaults(plots=True) # default makes plots
       
outArgs.add_argument("-ann","--annotations", dest='annotations', action='store_true',
                    help='generate annotations form clf predictions')
outArgs.add_argument("-no-ann", "--no-annotations", dest='annotations', action='store_false',
                    help="don't generate annotations")
outArgs.set_defaults(annotations=True) # default generates annotations                                  
                        
outArgs.add_argument("-WAP","--wavAnnPairs", dest='wavAnnPairs', action='store_true',
                    help='output a text file with the wav-annotation pairs')
outArgs.add_argument("-no-WAP", "--no-wavAnnPairs", dest='annotations', action='store_false',
                    help="don't output text file with wav-annotation pairs")
outArgs.set_defaults(wavAnnPairs=False) # default generates annotations                           
                        

######### ASSIGMENTS ##########
args = parser.parse_args()

###### FILES AND DIRS
### working dir
wDir=os.path.abspath(os.path.expanduser(args.workingDir))
### train
trainWav = os.path.abspath(os.path.expanduser(args.trainWav))
trainAnn = os.path.abspath(os.path.expanduser(args.trainAnn))
### test
testWav = args.testWav 
testAnn = args.testAnn 
if testWav is not None:
    assert len(testWav) == len(testAnn), "test: # annotations don't match # wavs"
    testWav = [os.path.abspath(os.path.expanduser(f)) for f in testWav]
    testAnn = [os.path.abspath(os.path.expanduser(f)) for f in testAnn]
else:
    testWav=[]
    testAnn=[]
### predict
## wav files    
predictWav = args.predictWav
if predictWav is not None: # somethingis given
    predictWav = [os.path.abspath(os.path.expanduser(f)) for f in predictWav]
else:
    predictWav = []
## collection of wav files in a text file
predictWavColl = args.predictWavColl
if predictWavColl is not None: # somethingis given
    for f in predictWavColl:
        predictWav+=[line.strip() for line in open(os.path.abspath(os.path.expanduser(f)), 'r') ]
  
print("PREDICT WAV COLLECTION:", predictWav)
#sys.exit()
### sumary file
summaryFi = args.summaryFile
### wav-annotations file
wavAnnPairsFi = args.summaryFile # bool
### iterKey
iterKey = args.iterKey				
###### CLF
SVCerrorC = args.SVCerrorC
classWeight = args.classWeight
testSize = args.testSize
### feature texturization
textureWinSz = args.textureWinSz
### fe extraction
featureType = args.featureType
featureExtrationParams=args.featureExtrationParams
print("HOLA", featureExtrationParams)
if featureExtrationParams is not None:
    i = iter(featureExtrationParams)
    featureExtrationParams = dict(zip(i, i))
else:
    featureExtrationParams = dict()

genAnnotations = args.annotations
genPlots = args.plots
wavAnnPairs = args.wavAnnPairs

print("working dir:", wDir)
print("TRAIN:", trainWav, trainAnn)
print("TEST:", testWav, testAnn)
print("SVM:", SVCerrorC, classWeight, testSize)
print("PREDICT:", predictWav)
print("size of the texture win:", textureWinSz )
print("feature type:", featureType)
print("featureExtrationParams:", featureExtrationParams)
print("iterKey:", iterKey[0], iterKey[1])
print("iterKey:", iterKey[0], iterKey[1])
print("GENEARATE: \nplots", genPlots, "\nannotations", genAnnotations, 
      "\nwavAnnPairs", wavAnnPairs)
######## FILE HANDLING -- out --> ########
## train
#trWavDN = os.path.dirname(trainWav)
trbN = os.path.splitext(os.path.basename(trainWav))[0]
## dirs
imgD = os.path.join(wDir, 'images')
if not os.path.isdir( imgD ): os.mkdir( imgD )
annD = trImgD = os.path.join(wDir, 'annotations')
if not os.path.isdir( annD ): os.mkdir( annD )
## sumary file
if summaryFi:
	summaryFi = os.path.join(wDir, 'clfsummary.txt')
print("OUT", summaryFi)

## sumary file
if wavAnnPairsFi:
	wavAnnPairsFi = os.path.join(wDir, 'wavAnnPairsFi.txt')
print("OUT", wavAnnPairsFi)

####### TRAIN #######
## load
waveForm, fs = sT.wav2waveform(trainWav)
annotLi_t = sT.aupTxt2annTu(trainAnn) ## in sample units
## feature extraction
featureExtrationParams['featExtrFun'] = sT.featureExtractionFun(featureType)
## train
M0, y0_names, featN, fExStr = sT.waveform2featMatrix(waveForm, fs,
						   textWS=textureWinSz,
                            annotations=annotLi_t,                           
                            **featureExtrationParams)

fExStr = featureType + fExStr
trainO = myML.dataObj(M0, attrNames=featN, target=y0_names, std_features=True)
trM = trainO.X
y_tr_num = trainO.y
labsD = trainO.targetNumDict()
## annotation sections we are interested on 
annSections = labsD.values()
annSections.remove('b')

## preprocessing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(trM,
										y_tr_num, test_size=testSize, 
										random_state=0)
## train clf
kernel = 'linear'
clf = svm.SVC(kernel=kernel, C=SVCerrorC, class_weight=classWeight).fit(X_train, y_train)
clfStr = fExStr + '-C%d'%SVCerrorC

#### PRINT clf scores
## cv test
wavStr = 'fs %d (%d)'%(fs, len(waveForm))
sumFeatStr = "insatances: " + str(trainO.m_instances) + " attributes: " + str(trainO.n_attr)
DiStr = ', '.join(['%s : %s'%(str(k), str(v))  for k, v in labsD.items()])
clsStr = DiStr + '\n%s'%', '.join(['%s (%s)'%(str(ky), str(ky_freq)) for ky, ky_freq in trainO.targetFrequencies()])
myML.printClfScores(summaryFi, clf, X_test, y_test, 
	l0='##  - - - < < < %s > > > - - -\nTEST CV%.3f\n%s\n%s\n%s\n%s\n%s'
	%(time.strftime("%c"), testSize, trainWav, wavStr, sumFeatStr, clsStr, clfStr))
## full train set
#thisStr += "\nTarget dict" + '-'.join([item for item in labsD.items()])
s=myML.printClfScores(summaryFi, clf, trM, y_tr_num, 
					l0='## full train set')

## predictions
y_tr_pred = clf.predict(trM) 
tf = 1.0*len(waveForm)/fs

if genAnnotations:     
    #print('HOLA', tf, len(M_p), len(y_p_names), len(y_p_pred))    
    outTxt = os.path.join(annD, trbN + '-pred%s'%fExStr + '.txt' )
    a2m.predictions2txt(trainO.target2tNames(y_tr_pred), outTxt, tf, sections=annSections)
    
if genPlots:     
    outPl = os.path.join(imgD, trbN + '-TRAINpred%s'%fExStr + '.png')
    myML.plXy(trM.T, np.vstack(( y_tr_num, y_tr_pred)), plTitle='predictions %.2f'%s, y_ix_names=labsD, outFig=outPl)#, )
    print("OUT:", outPl)
    
if wavAnnPairsFi:
    with open(wavAnnPairsFi, 'a') as f:
        f.write("%s\t%s\t%s\n"%(trainWav, trainAnn, outTxt))

    
## iterator file
if iterKey[0] :
    iterFile = os.path.join(wDir, 'CVtest-' + trbN +'-'+ iterKey[0] + '.txt'	)
    myML.printIterClfScores(iterFile, clf, X_test, y_test, "%s"%(iterKey[1]), comments=clsStr)
    print("OUT",iterFile)
	## 
    iterFile = os.path.join(wDir, 'CVtestAll-' + trbN + '-' + iterKey[0] + '.txt')
    myML.printIterClfScores(iterFile, clf, trM, y_tr_num, "%s"%(iterKey[1]), comments=clsStr)
    print("OUT:", iterFile)
            
####### TEST #######
for i in range(len(testWav)):    
    wF = testWav[i]
	## FILE handling
    #teWavDN = os.path.dirname(teW) # wavs dir
    tebN = os.path.splitext(os.path.basename(wF))[0] # base name
    #teImgD = dirN.replace('/wav', '/images')
	#if not os.path.isdir( teImgD ): os.mkdir( teImgD )
	### waveform					
    waveForm_t, fs_t = sT.wav2waveform(wF)
    tf = 1.0*len(waveForm_t)/fs_t

	## annotations											
    annLi_te = sT.aupTxt2annTu(testAnn[i]) ## in sample units
	### feature extraction
    M0_t, y_te_names, featN_t, fExStr_t = sT.waveform2featMatrix(waveForm_t, fs_t,
                                              textWS=textureWinSz,
											annotations=annLi_te, 
											**featureExtrationParams)

    testO = myML.dataObj(M0_t, featN_t, y_te_names, std_features=True)

	## numeric labels
    y_te_num = trainO.tNames2target(y_te_names) # read labels into the train dictionary
	## preprocess - scale features
    M_t = testO.X
    y_te_pred = clf.predict(M_t) 
	
	## print scores
    wavStr = 'fs %d (%d)'%(fs_t, len(waveForm_t))
    featStr=fExStr_t
    featStr+="\ninstances: " + str(testO.m_instances) + " attributes: " + str(testO.n_attr)
    ClassFreqs = ', '.join(['%s (%s)'%(str(ky), str(ky_freq)) for ky, ky_freq in testO.targetFrequencies()])

    s=myML.printClfScores(summaryFi, clf, M_t, y_te_num, 
						l0='## TEST %s\n%s\n%s\n%s'%(wF, wavStr, featStr, ClassFreqs))
					
	## print -iter scores
    if iterKey[0]:					
		iterFile = os.path.join(wDir, 'TEST-' + tebN + '-'+ iterKey[0] + '.txt'	)
		s=myML.printIterClfScores(iterFile, clf, M_t, y_te_num, "%s"%iterKey[1], 
                            comments=DiStr+'\n'+ClassFreqs)
		print(iterFile)
	
	####  <<<<< generate anotations >>>>>>
    if genAnnotations:     
         outTxt = os.path.join(annD, tebN + '-TESTpred%s'%fExStr + '.txt' )
         a2m.predictions2txt(trainO.target2tNames(y_te_pred), outTxt, tf, sections=annSections)
         print("OUT:", outTxt)	

    ####  <<<<< generate images >>>>>>
    if genPlots:     
        outPl = os.path.join(imgD, tebN + '-TESTpred%s'%fExStr + '.png')
        myML.plXy(M_t.T, np.vstack(( y_te_num, y_te_pred)), 
                  plTitle='predictions %.2f'%s, y_ix_names=labsD, outFig=outPl)#, )
                  
    if wavAnnPairsFi:
        with open(wavAnnPairsFi, 'a') as f:
            f.write("%s\t%s\t%s\n"%(wF, testAnn[i], outTxt))
    

###### PREDICT ######
for pW in predictWav:   
    print(pW)
    bN = os.path.splitext(os.path.basename(pW))[0] # base name
    waveForm_t, fs_t = sT.wav2waveform(pW)
    ##### feature extraction
    M_p, y_p_names, featN_p, fExStr_t = sT.waveform2featMatrix(waveForm_t, fs_t,
                                            textWS=textureWinSz,
                                            annotations=None, 
                                            **featureExtrationParams)
	
    predO = myML.dataObj(M_p, featN_p, target=None, std_features=True)											
    ## predictions	
    M_p = predO.X
    y_p_pred = clf.predict(M_p) 
    ## create audacity annotations from predictions
    tf = 1.0*len(waveForm_t)/fs_t
    outTxt = os.path.join(annD, bN + '-Ppred%s'%fExStr + '.txt' )
    a2m.predictions2txt(trainO.target2tNames(y_p_pred), outTxt, tf, sections=annSections)
    print("OUT:", outTxt)										
    ## images
    if genPlots:     
        outPl = os.path.join(imgD, bN +'-Ppred%s'%fExStr + '.png')
        myML.plXy(M_p.T,  y_p_pred, plTitle='predictions', y_ix_names=labsD, outFig=outPl)#, )
        print("OUT:", outPl)	
        
    if wavAnnPairsFi: # write wav - annotations pairs file
        with open(wavAnnPairsFi, 'a') as f:
            f.write("%s\t%s\t%s\n"%(pW, str('p'),outTxt))
            


sys.exit()

