#ifndef BOSSU_RAIN_GAUGE
#define BOSSU_RAIN_GAUGE


// MIT License
// 
// Copyright(c) 2019 Aalborg University
// Joakim Bruslund Haurum & Chris Holmberg Bahnsen, May 2019
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// An open source implementation of the estimation of rain intensity algorithm as 
// presented by J. Bossu, N. Hauti�re, and J.P. Tarel in "Rain or snow detection 
// in image sequences through use of a histogram of orientation of streaks." 
// appearing in International Journal of Computer Vision, 2011
//
// Overall, the algorithm comprises of the following elements:
// 1. Apply a standard Gaussian Mixture Model (Stauffer-Grimson) background 
//    subtraction algorithm to segment moving objects, including raindrops, 
//	  from the background
// 2. Use the Garg-Nayar intensity constraint to select possible rain 
//	  rain pixels from the foreground. 
// 3. Perform connected component analysis on the possible rain pixels and use
//	  size selection to further narrow the search. These pixels are dubbed 
//	  "candidate pixels"
// 4. Compute a histogram of the orientation of the candidate pixels, using the
//	  moments of the retrieved connected components
//		a) Use the moments of the connected components to estimate the orientation
//		   of the BLOB
//		b) Based on the estimate, make a Gaussian with fixed standard deviation
//		   whose mass is added to the histogram of orientation. This histogram is 
//		   named "Histogram of Orientation of Streaks" in the original article
// 5. Model the accumulated histogram using a mixture distribution of 1 Gaussian
//    and 1 uniform distribution in the range [0-179] degrees. 
//		a) Use an Expectation-Maximization scheme to estimate the mixture 
//		   distribution. 
//		   (Maybe use the built-in OpenCV EM class? There are indications that 
//			the authors have used this in their implementation)
// 6. Test the observed histogram against the estimated distributions
//    with a Goodness-Of-Fit / Kolmogorv-Smirnov test
// 7. If GOF test is passed, use a Kalman filter for each of the three parameters
//    of the mixture distribution to smooth the model temporally
//		a) The three parameters are: (1) Gaussian Mean, (2) Gaussian std.dev,
//		   (3) mass ratio of Gaussian distribution to uniform distribution 
// 8. Detect the rain intensity from the mixture model, if the Kalman estimated
//	  Gaussian mixture proportion is above the defined threshold
//		a) Rain intensity =approx mass ratio of Gaussian distribution to 
//		   uniform distribution multiplied by the unnormalized surface of the 
//		   Gaussian-Uniform distribution (which correlates to the number of 
//		   segmented BLOBs)
// 

#include <iostream>
#include <fstream>

#include <opencv2\opencv.hpp>
//#include <Windows.h>

struct BossuRainParameters {
	// Grey scale threshold from which candidate rain pixels are generated
	int c;

	// Maximum BLOB size of connected component in order to classify as rain pixel
	int maximumBlobSize;

	// Minimum BLOB size of connected component in order to classify as rain pixel
	int minimumBlobSize;

	// Gaussian std.dev scalar for estimating the Histogram of Orientation of Streaks
	float dm;

	// Maximum discrepency allowed between the observed histogram and estimated distribution
	float maxGoFDifference;

	// Minimum Gaussian sufrace in order to classify image as rainy
	float minimumGaussianSurface;

	// Maximum number of iterations for the Expectation-Maximization algorithm
	int emMaxIterations;

	// Options
	bool saveImg;
	bool verbose;
	bool debug;

	
};

class BossuRainIntensityMeasurer {
public:
	BossuRainIntensityMeasurer(std::string inputVideo, std::string filePath, std::string settingsFile, std::string outputFolder, BossuRainParameters rainParams = BossuRainIntensityMeasurer::getDefaultParameters());

	int detectRain();

	static BossuRainParameters loadParameters(std::string filePath);
	int saveParameters(std::string filePath);

	static BossuRainParameters getDefaultParameters();

private:

	void computeOrientationHistogram(
		const cv::Mat& image,
		const int numberOfBLOBs,
		std::vector<double>& histogram,
		double& surface,
		double dm);
	void estimateGaussianUniformMixtureDistribution(const std::vector<double>& histogram,
		const double surface,
		double& gaussianMean,
		double& gaussianStdDev,
		double& gaussianMixtureProportion);
	void plotDistributions(const std::vector<double>& histogram,
		const double gaussianMean,
		const double gaussianStdDev,
		const double gaussianMixtureProportion,
		const double kalmanGaussianMean,
		const double kalmanGaussianStdDev,
		const double kalmanGaussianMixtureProportion);
	double goodnessOfFitTest(const std::vector<double>& histogram,
		const double surface,
		const double gaussianMean,
		const double gaussianStdDev,
		const double gaussianMixtureProportion);
	void plotGoodnessOfFitTest(const std::vector<double>& histogram,
		const double surface,
		const double gaussianMean,
		const double gaussianStdDev,
		const double gaussianMixtureProportion);

	double uniformDist(double a, double b, double pos);
	double gaussianDist(double mean, double stdDev, double pos);
	

	std::string inputVideo, filePath, settingsFile, outputFolder;

	BossuRainParameters rainParams;
};

#endif // !BOSSU_RAIN_GAUGE

