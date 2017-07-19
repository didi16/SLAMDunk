// My includes
#include "loadConfiguration.h"

// OpenCV includes
#include "opencv2/opencv.hpp"


class confidenceMergeAlgorithm
{
public:
	confidenceMergeAlgorithm();
	~confidenceMergeAlgorithm();

 	void setmonoDepthMap(cv::Mat inputMonoDepthMap);
 	void setstereoDepthMap(cv::Mat inputStereoDepthMap);
 	void setConfidenceMap(cv::Mat inputConfidenceDepthMap);
 	void setPixels2BeMerged(std::vector<cv::Point_<int>> inputPixels2BeMerged);
 	cv::Mat getFinalDepthMap();
 	cv::Mat getSecondMap();
 	void merge();
 	void setScaleMonoDepthMap(float newScale);
 	void setScaleStereoDepthMap(float newScale);

 private:

 	cv::Mat monoDepthMap;
 	cv::Mat stereoDepthMap;
 	cv::Mat finalDepthMap;
 	cv::Mat confidenceMap;
	cv::Mat secondMap;
 	float scaleMonoDepthMap = 6.0;
 	float scaleStereoDepthMap = 10.0/255.0;
 	double computeWeight(double conf, double maxConf);
};