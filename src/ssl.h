//my includes
#include "manageObjectInputMap.h"
#include "manageObjectCnn.h"
#include "manageObjectDepthMap.h"
#include "manageDepthMapPerformance.h"
#include "stereoAlgorithms.h"
#include "loadConfiguration.h"


//C includes
#include <iostream>
#include <fstream>
#include <cmath> 
#include <stdlib.h>
#include <ctime>  

#define NYUDataset 0
#define MyDataset 1
#define ZED 2
#define LEARN 1
#define NOT_LEARN 0
#define STEREO_PERFORMANCE 99.5
void setupStart();

float scaleGT;
float scaleOriginalCnnMap;
float scaleSSLCnnMap;
bool mergeFromConfidenceMap;
int  thresholdConfidence;
float stdNoise;
manageObjectCnn * solver;
manageObjectCnn * cnn;
manageObjectInputMap * inputImage;
manageObjectInputMap * inputImageRight;
manageObjectInputMap * inputDepthMap;
manageObjectInputMap* cheapDepth;
manageObjectInputMap * inputConfidenceMap;
manageObjectDepthMap depthCnn;
manageDepthMapPerformance * performanOriginalCnnMap;
manageDepthMapPerformance * performanCnnMap;
manageDepthMapPerformance * performanceMergedMap;
manageDepthMapPerformance * performanceSecondMap;
manageDepthMapPerformance * performanceStereoMap;
displayObjectDepthMap displayDepthColorMap;
displayObjectDepthMap displayDepthCnnColorMap;
displayObjectDepthMap displayDepthOriginalCnnColorMap;
displayObjectDepthMap displayDepthCnnSSLColorMap;
displayObjectDepthMap displayDepthStereoMap;
displayObjectDepthMap displayMergedMap;
displayObjectDepthMap displaySecondMap;
stereoBMOpencv bmAlgorithm;
manageZEDObject * zedCamObject;
