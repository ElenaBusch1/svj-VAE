import h5py
import os,sys
import uproot
import numpy as np
import optparse
import random
#import ROOT

#inputFileString = "v9.1/v9p1_PFNv6_totalBkgALL_skim0.hdf5"
#outputFile = "v9.1/v9p1_PFNv6_totalBkgALL_skim0_downSample.root"
#inputFileString = "v9.2/v9p2_PFNv6_dataAll.hdf5"
#outputFile = "v9.2/v9p2_PFNv6_dataAll_downSample_5GeVBins_LowStat.root"
inputFileString = "../v12.5/v12p5_PFNv12_bkgALL.hdf5"
outputFile = "../v12.5/v12p5_PFNv12_bkgALL_downSampleTest.root"

score_cut = 0.6
#score_name='multi_reco_transformed_log10_sig'
score_name='score'
jet2_cut = 0.05
downSample = 1
scaleDown = False
addGaussian = False
signalScaleFactor=1
dataSRStats = 54181 #54181
xmin = 1500.
xmax = 6000.
tmp = 0.
scale = 1.


binning = np.linspace(1500,6000,901)
#binning = [1500, 1550, 1601.0, 1652.0, 1704.0, 1758.0, 1813.0, 1870.0, 1930.0, 1993.0, 2059.0, 2129.0, 2204.0, 2284.0, 2371.0, 2465.0, 2568.0, 2681.0, 2805.0, 2943.0, 3097.0, 3269.0, 3464.0, 3685.0, 3937.0, 4226.0, 4560.0, 4947.0, 5399.0, 6000.0]
nbins = len(binning)-1

def get_weighted_elements(weights, seed, nEvents):
    my_weight_array = np.abs(weights)
    np.random.seed(seed)
    idx = np.random.choice( my_weight_array.size, size=nEvents, p=my_weight_array/float(my_weight_array.sum()),replace=False) # IMPORTANT that replace=False so that event is picked only once
    return idx

def makeHist(inputFileString, outputFile):
  inputFile = h5py.File(inputFileString,'r')
  
  mtHistData = []
  mtHistWeight = []
  
  mtData = inputFile['data']["mT_jj"]
  scoreData = inputFile['data'][score_name]
  jet2_WidthData = inputFile['data']['jet2_Width']
  weightData = inputFile['data']["weight"]
  if "515" in inputFileString:
    weightData = weightData*signalScaleFactor
  
  num = 0
  mtData_PS = mtData[jet2_WidthData>-1] # dummy cut for formatting
  mtData_CR = mtData[jet2_WidthData<jet2_cut]
  print("CR unscaled events", len(mtData_CR[(mtData_CR>xmin) & (mtData_CR<xmax)]))
  mtData_VR = mtData[(jet2_WidthData>jet2_cut) & (scoreData<score_cut)]
  print("VR unscaled events: ", len(mtData_VR[(mtData_VR>xmin) & (mtData_VR<xmax)]))
  if "data" not in inputFileString:
    print("Not data - constructing SR")
    isData = False
    mtData_SR = mtData[(jet2_WidthData>jet2_cut) & (scoreData>score_cut)]
    print("SR unscaled events: ", len(mtData_SR[(mtData_SR>xmin) & (mtData_SR<xmax)]))
  else:
    isData = True
    print("Data file - no SR")
  
  if not isData:
    weightData_PS = weightData[jet2_WidthData>-1] #dummy cut for formatting
    weightData_CR = weightData[jet2_WidthData<jet2_cut]
    weightData_VR = weightData[(jet2_WidthData>jet2_cut) & (scoreData<score_cut)]
    weightData_SR = weightData[(jet2_WidthData>jet2_cut) & (scoreData>score_cut)]
  
  else:
    weightData_PS = np.ones(len(mtData_PS))
    weightData_CR = np.ones(len(mtData_CR))
    weightData_VR = np.ones(len(mtData_VR))
    #weightData_SR = np.ones(len(mtData_SR))
  
  if addGaussian:
    sig = np.random.normal(2000,200,30000)
    sig_weight = np.ones(len(sig))
    mtData_CR = np.concatenate((mtData_CR,sig))
    print("Gaussian signal events injected: ", len(sig[sig>1500]))
    weightData_CR = np.concatenate((weightData_CR,sig_weight))

  if downSample > 0:
    print("Down Sampling")
    PS_hists = []
    CR_hists = []
    VR_hists = []
    SR_hists = []
    #random.shuffle(mtData_PS)
    #random.shuffle(mtData_CR)
    #random.shuffle(mtData_VR)
    #if not isData: print("WARNING!!! Just shuffled MC weighted samples without shuffling weights")
    for i in range(downSample):
      # PS, CR, VR
      for weightData, mtData, hists in zip([weightData_PS, weightData_CR, weightData_VR], [mtData_PS, mtData_CR, mtData_VR], [PS_hists, CR_hists, VR_hists]):
        weights = np.array([])
        mts = np.array([])
        while weights.sum() < dataSRStats:
          idx = get_weighted_elements(weightData, i, 1000)
          weights = np.append(weights, weightData[idx])
          mts = np.append(mts, mtData[idx])
          weightData[idx] = 0
          if weightData.sum == 0: break
        print("Weighted selected events: ", weights.sum())
        hists.append(np.histogram(mts,binning,(xmin,xmax),weights=weights))
      #SR
      if not isData:
        weights = np.array([])
        mts = np.array([])
        while weights.sum() < dataSRStats:
          idx = get_weighted_elements(weightData_SR, i, 1000)
          weights = np.append(weights, weightData_SR[idx])
          mts = np.append(mts, mtData_SR[idx])
          weightData_SR[idx] = 0
          if weightData_SR.sum == 0: break
        print("Weighted selected events: ", weights.sum())
        SR_hists.append(np.histogram(mts,binning,(xmin,xmax),weights=weights))
  else:
    mtHist_PS = np.histogram(mtData_PS,binning,(xmin,xmax),weights=weightData_PS)
    mtHist_CR = np.histogram(mtData_CR,binning,(xmin,xmax),weights=weightData_CR)
    mtHist_VR = np.histogram(mtData_VR,binning,(xmin,xmax),weights=weightData_VR)
    if not isData:
      mtHist_SR = np.histogram(mtData_SR,binning,(xmin,xmax),weights=weightData_SR)
  
  if scaleDown:
    hists = [mtHist_PS, mtHist_CR, mtHist_VR]
    if not isData: hists.append(mtHist_SR)
    for h in hists:
      h_sum = np.sum(h[0])
      print(h_sum)
      for b in range(nbins):
        h_vals = h[0]
        n_bins = h[1]
        h_vals[b] = round(h_vals[b] * dataSRStats/h_sum)
      print(np.sum(h[0]))
  
  print("Writing to ", outputFile)
  with uproot.recreate(outputFile) as outputFile:
    if downSample > 0:
      for i in range(downSample):
        outputFile["mT_PS"+str(i)] = PS_hists[i]
        print(outputFile["mT_PS"+str(i)].errors())
        outputFile["mT_CR"+str(i)] = CR_hists[i]
        outputFile["mT_VR"+str(i)] = VR_hists[i]
        if not isData:
          outputFile["mT_SR"+str(i)] = SR_hists[i]
    else:
      outputFile["mT_PS"] = mtHist_PS
      outputFile["mT_CR"] = mtHist_CR
      outputFile["mT_VR"] = mtHist_VR
      if not isData:
        outputFile["mT_SR"] = mtHist_SR
  
  inputFile.close()
  print("File closed")

#for i in range(515495,515523):
#  inputFileString = "v9.2/v9p2_PFNv6_"+str(i)+".hdf5"
#  outputFile = "v9.2/v9p2_PFNv6_"+str(i)+"_fine.root"
#  makeHist(inputFileString, outputFile)

#inputFileString = "v9.2/v9p2_vANTELOPE_bkgAll_log10_0-67_jet2_width.hdf5"
#outputFile = "v9.2/v9p2_vANTELOPE_bkgAll_log10_0-67_jet2_width.root"
makeHist(inputFileString, outputFile)
  
