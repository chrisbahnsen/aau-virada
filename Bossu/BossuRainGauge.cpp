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
// See BossuRainGauge.h for further explanation


#include "BossuRainGauge.h"

using namespace std;
using namespace cv;

#define USERAD true

static const double RAD2DEG = 180. / CV_PI;
static const double DEG2RAD = CV_PI/180.;
static const double HALF_PI = CV_PI/2.;
static const double TWO_PI = CV_PI * 2.;
static const double SQRT_TWO_PI = sqrt(TWO_PI);
static const double SQRT_TWO = sqrt(2.);

#if USERAD
static const double defaultMean = HALF_PI;
static const double defaultStdDev = 0.;
static const double defaultMixtureProportion = 0.;
static const double uniformMaxRange = CV_PI;
#else
static const double defaultMean = 90.;
static const double defaultStdDev = 0.;
static const double defaultMixtureProportion = 0.;
static const double uniformMaxRange = 180.;
#endif

static const double uniformStepSize = uniformMaxRange/180.;
static const int warmUpFrames = 500;




BossuRainIntensityMeasurer::BossuRainIntensityMeasurer(std::string inputVideo, std::string filePath, std::string settingsFile, std::string outputFolder, BossuRainParameters rainParams)
{
	this->inputVideo = inputVideo;
	this->filePath = filePath;
	this->settingsFile = settingsFile;
	this->outputFolder = outputFolder;
	this->rainParams = rainParams;
}

int BossuRainIntensityMeasurer::detectRain()
{

	cout << "c: " << rainParams.c << endl;
	cout << "minimumBlobSize: " << rainParams.minimumBlobSize << endl;
	cout << "maximumBlobSize: " << rainParams.maximumBlobSize << endl;
	cout << "dm: " << rainParams.dm << endl;
	cout << "maxGoFDifference: " << rainParams.maxGoFDifference << endl;
	cout << "minimumGaussianSurface: " << rainParams.minimumGaussianSurface << endl;
	cout << "emIterations: " << rainParams.emMaxIterations << endl;
	cout << "saveImg: " << rainParams.saveImg << endl;
	cout << "verbose: " << rainParams.verbose << endl;
	cout << "debug: " << rainParams.debug << endl;

	// Open video
	VideoCapture cap(this->filePath + this->inputVideo);

	// Initialize MOG background subtraction model
	cv::Ptr<BackgroundSubtractorMOG2> backgroundSubtractor = 
		cv::createBackgroundSubtractorMOG2(500, 16.0, false);

	// Create file, write header
	ofstream resultsFile;
	resultsFile.open(this->outputFolder + "/" + this->inputVideo + "_" + "Results" + ".csv", ios::out | ios::trunc);

	std::string header;
	header += string("settingsFile") + ";" + "InputVideo" + "; " + "Frame#" + "; " +
		"GaussMean" + ";" + "GaussStdDev" + ";" +
		"GaussMixProp" + ";" + "Goodness-Of-Fit Value" + ";" +
		"kalmanGaussMean" + ";" + "kalmanGaussStdDev" + ";" +
		"kalmanGaussMixProp" + ";" + "Rain Intensity" + ";" + "Kalman Rain Intensity" + ";" + "EM Rain Detected" + ";" "Kalman Rain Detected" + "\n";
	 
	resultsFile << header;

	double kalmanMeanTmp = 0.0;
	double kalmanStdDevTmp = 0.0;
	double kalmanMixPropTmp = 0.0; 

	if (cap.isOpened()) {
		Mat frame, foregroundMask, backgroundImage;


		// Use the first few frames to initialize the Gaussian Mixture Model.
		for (auto i = 0; i < warmUpFrames; ++i) {
			cap.read(frame);

			backgroundSubtractor->apply(frame, foregroundMask);
			backgroundSubtractor->getBackgroundImage(backgroundImage);
			resultsFile << "\n"; // Write blank frame
		}

		// Set up Kalman filter to smooth the Gaussian-Uniform mixture distribution
		cv::KalmanFilter KF(6, 3, 0, CV_64F);
		
		KF.transitionMatrix = (Mat_<double>(6, 6) << 
			1, 0, 0, 1, 0, 0,
			0, 1, 0, 0, 1, 0,
			0, 0, 1, 0, 0, 1,
			0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 1);
		setIdentity(KF.measurementMatrix);
		setIdentity(KF.processNoiseCov, Scalar::all(0.01)); // According to Bossu et al, 2011, p. 10, right column
		setIdentity(KF.measurementNoiseCov, Scalar::all(0.1)); // According to Bossu et al, 2011, p. 10, right column
		setIdentity(KF.errorCovPost, Scalar::all(1.));
		

		cout << "measurementMatrix = " << endl << " " << KF.measurementMatrix << endl << endl;
		cout << "processNoiseCov = " << endl << " " << KF.processNoiseCov << endl << endl;
		cout << "measurementNoiseCov = " << endl << " " << KF.measurementNoiseCov  << endl << endl;
		cout << "errorCovPost = " << endl << " " << KF.errorCovPost << endl << endl;
		
		int vidLength = cap.get(CV_CAP_PROP_FRAME_COUNT);
		int frameCounter = warmUpFrames;
		while (cap.grab()) {
			frameCounter++;
			Mat foregroundImage;

			if (frameCounter % 100 == 0)
				cout << "Frame: " << frameCounter << "/" << vidLength << endl;

			// Continue while there are frames to retrieve
			cap.retrieve(frame);

			backgroundSubtractor->apply(frame, foregroundMask);
			backgroundSubtractor->getBackgroundImage(backgroundImage);

			// Construct the foreground image from the mask
			frame.copyTo(foregroundImage, foregroundMask);

			// Operate on grayscale images from now on
			Mat grayForegroundImage, grayBackgroundImage;
			cvtColor(foregroundImage, grayForegroundImage, COLOR_BGR2GRAY);
			cvtColor(backgroundImage, grayBackgroundImage, COLOR_BGR2GRAY);

			// Use the Garg-Nayar intensity constraint to select candidate rain pixels
			// Bossu et al., 2011, pp. 4, equation 8
			Mat diffImg = grayForegroundImage - grayBackgroundImage;
			Mat candidateRainMask;

			threshold(diffImg, candidateRainMask, rainParams.c, 255, CV_THRESH_BINARY);

			// We now have the candidate rain pixels. Use connected component analysis
			// to filter out too small or large connected components in candidateRainMask
			Mat ccLabels, ccMask;
			int ccCount = connectedComponents(candidateRainMask, ccLabels, 8);

			int approvedCCs = 0;
			Mat approvedCCsMat = Mat::zeros(candidateRainMask.rows, candidateRainMask.cols, CV_32S);
			for (int i = 1; i < ccCount; i++) {
				cv::compare(ccLabels, Scalar(i), ccMask, CMP_EQ);
				int blobSize = cv::countNonZero(ccMask);

				if ((blobSize > rainParams.maximumBlobSize) ||
					(blobSize < rainParams.minimumBlobSize)) {
					continue;
				}
				else {
					ccMask /= 255;
					ccMask.convertTo(ccMask, CV_32S);
					approvedCCs++;
					approvedCCsMat = approvedCCsMat + approvedCCs * ccMask;
				}
			}

			if (rainParams.debug) {
				int deletedCCs = ccCount - approvedCCs - 1;
				std::cout << "Deleted " << deletedCCs << " connected components, out of " << ccCount - 1 << " with dm: " << rainParams.dm << endl;
			}

			// For visualizing the kept connected components
			Mat ccFiltered;
			if (rainParams.debug || rainParams.saveImg){
				normalize(approvedCCsMat, ccFiltered, 255, 0, NORM_MINMAX);
				ccFiltered.convertTo(ccFiltered, CV_8UC1);
				threshold(ccFiltered, ccFiltered, 0, 255, CV_THRESH_BINARY);
			}

			if (rainParams.saveImg) {
				imwrite("Image.png", frame);
				imwrite("backgroundImg.png", grayBackgroundImage);
				imwrite("foregroundImage.png", grayForegroundImage);
				imwrite("diffImg.png", diffImg);
				imwrite("Candidate.png", candidateRainMask);
				imwrite("Filtered.png", ccFiltered);
			}
			if (rainParams.debug) {
				imshow("Image", frame);
				imshow("backgroundImg", grayBackgroundImage);
				imshow("foregroundImage", grayForegroundImage);
				imshow("diffImg", diffImg);
				imshow("Candidate", candidateRainMask);
				imshow("Filtered", ccFiltered);
			}

			if (approvedCCs > 0) {
				// 4. Compute the Histogram of Orientation of Streaks (HOS) from the connected components
				vector<double> histogram;
				double surface;
				computeOrientationHistogram(approvedCCsMat, approvedCCs, histogram, surface, rainParams.dm);

				// 5. Model the accumulated histogram using a mixture distribution of a Gaussian
				//    and a Uniform distribution in the range [0-179] degrees.
				double gaussianMean, gaussianStdDev, gaussianMixtureProportion;

				estimateGaussianUniformMixtureDistribution(histogram, surface,
					gaussianMean, gaussianStdDev, gaussianMixtureProportion);

				// 6. Goodness-Of-Fit test between the observed histogram and estimated mixture distribution
				double ksTest = goodnessOfFitTest(histogram, surface, gaussianMean, gaussianStdDev, gaussianMixtureProportion);

				if (rainParams.debug || rainParams.saveImg)
					plotGoodnessOfFitTest(histogram, surface, gaussianMean, gaussianStdDev, gaussianMixtureProportion);

				// 7. Use a Kalman filter for each of the three parameters of the mixture
				//	  distribution to smooth the model temporally
				//    (Should only update if the Goodness-OF-Fit test is within the defined threshold, but we still enter here for plotting reasons)

				double kalmanGaussianMean;
				double kalmanGaussianStdDev;
				double kalmanGaussianMixtureProportion;

				// NOTE: If kalman predict is moved out of this if statement, kalman rain intensity/mixture propotion WILL go into negative numbers if not updated reguraly
				if (ksTest <= rainParams.maxGoFDifference) {

					Mat kalmanPredict = KF.predict();

					Mat measurement = (Mat_<double>(3, 1) <<
						gaussianMean, gaussianStdDev, gaussianMixtureProportion);
					Mat estimated = KF.correct(measurement);

					kalmanGaussianMean = estimated.at<double>(0);
					kalmanGaussianStdDev = estimated.at<double>(1);
					kalmanGaussianMixtureProportion = estimated.at<double>(2);

					kalmanMeanTmp = kalmanGaussianMean;
					kalmanStdDevTmp = kalmanGaussianStdDev;
					kalmanMixPropTmp = kalmanGaussianMixtureProportion;

					if (rainParams.verbose)
						cout << "Updating Kalman filter" << endl;
				}
				else {

					kalmanGaussianMean = kalmanMeanTmp;
					kalmanGaussianStdDev = kalmanStdDevTmp;
					kalmanGaussianMixtureProportion = kalmanMixPropTmp;

					if (rainParams.verbose)
						cout << "Not updating Kalman filter" << endl;
				}

				if (rainParams.verbose) {
					cout << "EM Estimated: Mean: " << gaussianMean << ", std.dev: " <<
						gaussianStdDev << ", mix.prop: " << gaussianMixtureProportion << endl;
					cout << "Kalman:       Mean: " << kalmanGaussianMean << ", std.dev: " <<
						kalmanGaussianStdDev << ", mix.prop: " << kalmanGaussianMixtureProportion << endl;
				}


				if (rainParams.saveImg || rainParams.debug)
				{
					plotDistributions(histogram, gaussianMean, gaussianStdDev,
						gaussianMixtureProportion,
						kalmanGaussianMean, kalmanGaussianStdDev, kalmanGaussianMixtureProportion);
				}

				// 8. Detect the rain intensity from the mixture model
				// Now that we have estimated the distribution and the filtered distribution,
				// compute an estimate of the rain intensity
				// (Should only be calculated if kalmanGaussianMixtureProportion is above the threshold. Still calculated for plotting reasons)

				// Step 1: Compute the sum (surface) of the histogram (already computed in 4. computation of HOS)
				// Step 2: Compute the rain intensity R on both the estimate and filtered estimate
				// This value should be proportional to actual rain intensity. 
				double R = surface * gaussianMixtureProportion;
				double kalmanR = surface * kalmanGaussianMixtureProportion;

				double rain = ((ksTest <= rainParams.maxGoFDifference) && (gaussianMixtureProportion >= rainParams.minimumGaussianSurface));
				double kalmanRain = ((ksTest <= rainParams.maxGoFDifference) && (kalmanGaussianMixtureProportion >= rainParams.minimumGaussianSurface));

				if (rainParams.verbose) {
					cout << "Is rain detected (EM): " << rain << endl;
					cout << "Is rain detected (Kalman): " << kalmanRain << endl;
					cout << "EM Estimated Rain Intensity: " << R << endl;
					cout << "Kalman Estimated Rain Intensity: " << kalmanR << endl;
					cout << "Number of rain pixels: " << cv::countNonZero(approvedCCsMat) << endl;
					cout << "Number of BLOBs: " << approvedCCs << endl;
					cout << "Surface of HOS histogram: " << surface << endl;
				}

				resultsFile <<
					this->outputFolder + "/" + this->settingsFile + ";" +
					this->inputVideo + "; " +
					to_string(frameCounter) + "; " +
					to_string(gaussianMean) + ";" +
					to_string(gaussianStdDev) + ";" +
					to_string(gaussianMixtureProportion) + ";" +
					to_string(ksTest) + ";" +
					to_string(kalmanGaussianMean) + ";" +
					to_string(kalmanGaussianStdDev) + ";" +
					to_string(kalmanGaussianMixtureProportion) + ";" +
					to_string(R) + ";" +
					to_string(kalmanR) + ";" +
					to_string(rain) + ";" + 
					to_string(kalmanRain) + "\n";
			}
			else {
				if (rainParams.verbose || rainParams.debug)
					cout << "0 CONNECTED COMPONENTS FOUND" << endl;

				resultsFile <<
					this->outputFolder + "/" + this->settingsFile + ";" +
					this->inputVideo + "; " +
					to_string(frameCounter) + "; " +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(-1) + ";" +
					to_string(0) + ";" +
					to_string(0) + ";" +
					to_string(0) + ";" +
					to_string(0) + "\n";
			}


			if (rainParams.verbose)
				cout << "\n" << endl;
			if (rainParams.debug)
				waitKey(0);
		}

	}
	resultsFile.close();
	return 0;
}

BossuRainParameters BossuRainIntensityMeasurer::loadParameters(std::string filePath)
{
	BossuRainParameters newParams = getDefaultParameters();

	FileStorage fs(filePath, FileStorage::READ);

	if (fs.isOpened()) {
		int tmpInt;
		fs["c"] >> tmpInt;
		if (tmpInt != 0) {
			newParams.c = tmpInt;
		}

		fs["minimumBlobSize"] >> tmpInt;
		if (tmpInt >= 0) {
			newParams.minimumBlobSize = tmpInt;
		}

		fs["maximumBlobSize"] >> tmpInt;
		if (tmpInt > 0) {
			newParams.maximumBlobSize = tmpInt;
		}

		float tmpFloat;
		fs["dm"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.dm = tmpFloat;
		}

		fs["maxGoFDifference"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.maxGoFDifference = tmpFloat;
		}

		fs["minimumGaussianSurface"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.minimumGaussianSurface = tmpFloat;
		}

		fs["emMaxIterations"] >> tmpInt;
		if (tmpInt > 0) {
			newParams.emMaxIterations = tmpInt;
		}


		fs["saveImg"] >> newParams.saveImg;
		fs["verbose"] >> newParams.verbose;
		fs["debug"] >> newParams.debug;
	}

	return newParams;
}

int BossuRainIntensityMeasurer::saveParameters(std::string filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);

	if (fs.isOpened()) {
		fs << "c" << rainParams.c;
		fs << "minimumBlobSize" << rainParams.minimumBlobSize;
		fs << "maximumBlobSize" << rainParams.maximumBlobSize;
		fs << "dm" << rainParams.dm;
		fs << "maxGoFDifference" << rainParams.maxGoFDifference;
		fs << "minimumGaussianSurface" << rainParams.minimumGaussianSurface;
		fs << "emMaxIterations" << rainParams.emMaxIterations;
		fs << "saveImg" << rainParams.saveImg;
		fs << "verbose" << rainParams.verbose;
		fs << "debug" << rainParams.debug;
	}
	else {
		return 1;
	}
}

BossuRainParameters BossuRainIntensityMeasurer::getDefaultParameters()
{
	BossuRainParameters defaultParams;

	defaultParams.c = 3;
	defaultParams.dm = 1.;
	defaultParams.emMaxIterations = 100;
	defaultParams.minimumBlobSize = 4;
	defaultParams.maximumBlobSize = 50;
	defaultParams.maxGoFDifference = 0.06;
	defaultParams.minimumGaussianSurface = 0.35;
	defaultParams.saveImg = true;
	defaultParams.verbose = true;
	defaultParams.debug = true;

	
	return defaultParams;
}

void BossuRainIntensityMeasurer::computeOrientationHistogram(
	const Mat& image,
	const int numberOfBLOBs,
	std::vector<double>& histogram,
	double& surface,
	double dm)
{
	// Compute the moments from the connected components and get the orientation of the BLOB
	histogram.clear();
	histogram.resize(180);

	Mat ccMask;
	for (int i = 1; i <= numberOfBLOBs; i++) {
		cv::compare(image, Scalar(i), ccMask, CMP_EQ);

		// Calculate the central second-order moments as per Bossu et al, 2011, pp. 6, equation 13
		// OpenCV central moments are denoted mu, and doesn't divide by m00 by default
		Moments mu = moments(ccMask, true);
		mu.mu02 /= mu.m00;
		mu.mu11 /= mu.m00;
		mu.mu20 /= mu.m00;

		// Compute the major semiaxis of the ellipse equivalent to the BLOB
		// In order to do so, we must compute eigenvalues of the matrix
		// | mu20 mu11 |
		// | mu11 mu02 |
		// Bossu et al, 2011, pp. 6, equation 16
		Mat momentsMat = Mat(2, 2, CV_64FC1);
		momentsMat.at<double>(0, 0) = mu.mu20;
		momentsMat.at<double>(1, 0) = mu.mu11;
		momentsMat.at<double>(0, 1) = mu.mu11;
		momentsMat.at<double>(1, 1) = mu.mu02;

		Mat eigenvalues;
		eigen(momentsMat, eigenvalues);


		// Extract the largest eigenvalue and compute the major semi-axis according to
		// Bossu et al, 2011, pp. 6, equation 14
		double a = sqrt(eigenvalues.at<double>(0, 0));
		a = a > 0 ? a : -a;

		// Bossu et al, 2011, pp. 6, equation 17
		double orientation = 0.5 * (atan2(2 * mu.mu11, (mu.mu02 - mu.mu20)));

		//Convert from [-\pi / 2, \pi /2] to [0, \pi]
		orientation += HALF_PI;

		// Compute the uncertainty of the estimate to be used as standard deviation
		// for Gaussian
		double estimateUncertaintyNominator = sqrt(pow(mu.mu02 - mu.mu20, 2) + 2 * pow(mu.mu11, 2)) * dm;
		double estimateUncertaintyDenominator = pow(mu.mu02 - mu.mu20, 2) + 4 * pow(mu.mu11, 2);

		double estimateUncertainty = estimateUncertaintyDenominator > 0 ?
			estimateUncertaintyNominator / estimateUncertaintyDenominator :
			1;
		if (rainParams.debug) {
			std::cout << "Orient (Deg): " << orientation * RAD2DEG << ", unct: " << estimateUncertainty << ", major semiaxis: " << a << endl;
			std::cout << "mu00: " << mu.m00 << ", mu20: " << mu.mu20 << ", mu02 : " << mu.mu02 << ", mu11 : " << mu.mu11 << ", lambda_1 : " << eigenvalues.at<double>(0, 0) << ", lambda_2 : " << eigenvalues.at<double>(1, 0) << ", BLOB size : " << countNonZero(ccMask) << endl << endl;
		}

		// Compute the Gaussian (Parzen) estimate of the true orientation and 
		// add to the histogram
		for (int angle = 0; angle < histogram.size(); ++angle) {
			double angle_val = USERAD ? angle * DEG2RAD : angle;
			double orientation_val = USERAD ? orientation : orientation * RAD2DEG;

			histogram[angle] += a / (estimateUncertainty * SQRT_TWO_PI) *
				exp(-0.5 * pow((angle_val - orientation_val) / estimateUncertainty, 2.));
		}
	}

	// Calculate the sum/surface of the unnormalized histogram
	surface = 0;
	for (auto& n : histogram) {
		surface += n;
	}

	if(rainParams.debug)
		cout << "Surface of histogram: " << surface << endl;
}

void BossuRainIntensityMeasurer::estimateGaussianUniformMixtureDistribution(const std::vector<double>& histogram, 
	const double surface,
	double & gaussianMean, 
	double & gaussianStdDev, 
	double & gaussianMixtureProportion)
{
	// Estimate a Gaussian-Uniform mixture distribution using the data in the histogram. 
	// Use the Expectation-Maximization algorithm as provided by Bossu et al, 2011, page 8, equation 24-25

	// Initialize the EM algorithm as described by Bossu et al, 2011, page 8, upper right column (We actually use the method from Bossu et al, 2009  pp. 2, eq 6-8   ( http://perso.lcpc.fr/hautiere.nicolas/pdf/2009/hautiere-gretsi09b.pdf ) )
	// We find the median value of the histogram and use it to estimate initial values of
	// gaussianMean, gaussianStdDev, and gaussianMixtureProportion

	//Sort the histogram in ascending order
	std::vector<double> histogram_sorted = histogram;
	std::sort(histogram_sorted.begin(), histogram_sorted.end());

	// Find the median value in the sorted histogram.
	double median = 0;

	if (histogram_sorted.size() % 2 == 0)
		median = (histogram_sorted[histogram_sorted.size() / 2 - 1] + histogram_sorted[histogram_sorted.size() / 2]) / 2;
	else
		median = histogram_sorted[histogram_sorted.size() / 2];
	
	if (rainParams.debug)
		cout << "Largest histogram value: " << histogram_sorted[histogram_sorted.size() - 1] << " Median value: " << median << endl;



	// Now that we have found the median, only use entries in the histogram equal to or above
	// the median to calculate the mean, std.dev and proportion estimate
	// i.e. subtract median and only use positive values
	double sumAboveMedian = 0, observationSum = 0, mixtureProportionAboveMedian = 0, sumOfSqDiffToMean = 0;
	double uniformMass = uniformDist(0, uniformMaxRange, uniformStepSize);

	for (auto angle = 0; angle < histogram.size(); ++angle) {
		double val = histogram[angle] - median;
		if (val < 0.0)
			continue;

		double angle_val = USERAD ? angle * DEG2RAD : angle;
		sumAboveMedian += angle_val * val;
		observationSum += val;
	}

	double initialMean = observationSum > 0 ? sumAboveMedian / observationSum : defaultMean;
	double initialMixtureProportion = surface > 0 ? observationSum / surface : defaultMixtureProportion;

	for (auto angle = 0; angle < histogram.size(); ++angle) {
		double val = histogram[angle] - median;
		if (val < 0.0)
			continue;

		double angle_val = USERAD ? angle * DEG2RAD : angle;
		sumOfSqDiffToMean += pow(angle_val - initialMean, 2) * val;
	}

	double initialStdDev = observationSum > 0 ? sqrt(sumOfSqDiffToMean / observationSum) : defaultStdDev;


	if(rainParams.debug)
		cout << "SumAboveMean: " << sumAboveMedian << ", ObservationSum: " << observationSum << ", initialMean: " << initialMean << ", sumOfSquareDiff: " << sumOfSqDiffToMean << ", initialStdDev: " << initialStdDev << ", initialMixProp: " << initialMixtureProportion << ", uniform Dist est.: " << uniformMass << endl;
	

	// Now that we have the initial values, we may start the EM algorithm
	vector<double> estimatedMixtureProportion{ initialMixtureProportion };
	vector<double> estimatedGaussianMean{ initialMean };
	vector<double> estimatedGaussianStdDev{ initialStdDev };
	vector<double> z;

	if (rainParams.verbose) {
		std::cout << "Mean: " << estimatedGaussianMean.back()
			<< ", stdDev: " << estimatedGaussianStdDev.back() 
			<< ", mixProp: " <<	estimatedMixtureProportion.back() 
			<< endl;
	}

	double mixtureProportionDenominator = surface;
	for (auto i = 1; i <= rainParams.emMaxIterations; ++i) {

		// Expectation step
		// Bossu et al, 2011, pp. 8, equation 24
		z.clear();
		double zNominator = (1. - estimatedMixtureProportion.back()) * uniformMass; //Constant for all angles in the range [0; \pi]
		for (double angle = 0; angle < 180.; ++angle) {
			double angle_val = USERAD ? angle * DEG2RAD : angle;

			// NOTE: zDenominator should in theory always be non-zero, due to the infinite support of the Gaussian distribution. 
			//       However due to the finite range of the floats/doubles this is not true in our case. 
			double zDenominator = zNominator + (estimatedMixtureProportion.back() * gaussianDist(estimatedGaussianMean.back(), estimatedGaussianStdDev.back(), angle_val));
			double zVal = zDenominator > 0 ? zNominator / zDenominator : 0.;
			z.push_back(zVal);
		}

		// Maximization step
		// Bossu et al, 2011, pp. 8, equation 25
		double meanNominator = 0;
		double meanDenominator = 0;
		double stdDevNominator = 0;
		double stdDevDenominator = 0;
		double mixtureProportionNominator = 0;

		for (int angle = 0; angle < 180; ++angle) {
			double angle_val = USERAD ? angle * DEG2RAD : angle;
			meanNominator += (1. - z[angle]) * angle_val * histogram[angle];
			meanDenominator += (1. - z[angle]) * histogram[angle];
		}
		double tmpGaussianMean = meanDenominator > 0 ? meanNominator / meanDenominator : defaultMean;
		estimatedGaussianMean.push_back(tmpGaussianMean);

		for (int angle = 0; angle < 180; ++angle) {
			double angle_val = USERAD ? angle * DEG2RAD : angle;
			stdDevNominator += ((1. - z[angle]) * pow(angle_val - estimatedGaussianMean.back(),2) * histogram[angle]);
		}

		stdDevDenominator = meanDenominator;
		mixtureProportionNominator = meanDenominator;

		double tmpGaussianStdDev = stdDevDenominator > 0 ? sqrt(stdDevNominator / stdDevDenominator) : defaultStdDev;
		estimatedGaussianStdDev.push_back(tmpGaussianStdDev);
		
		double tmpMixtureProportion = mixtureProportionDenominator > 0 ? mixtureProportionNominator /
			mixtureProportionDenominator : defaultMixtureProportion;
		estimatedMixtureProportion.push_back(tmpMixtureProportion);

		if ((i % 25 == 0) && rainParams.verbose) {
			std::cout << "EM step " << i << ": Mean: " << estimatedGaussianMean.back()
				<< ", stdDev: " << estimatedGaussianStdDev.back() << ", mixProp: " <<
				estimatedMixtureProportion.back() << endl;
		}
	}

	gaussianMean = estimatedGaussianMean.back();
	gaussianStdDev = estimatedGaussianStdDev.back();
	gaussianMixtureProportion = estimatedMixtureProportion.back();
}

void BossuRainIntensityMeasurer::plotDistributions(const std::vector<double>& histogram, const double gaussianMean, const double gaussianStdDev, const double gaussianMixtureProportion, const double kalmanGaussianMean, const double kalmanGaussianStdDev, const double kalmanGaussianMixtureProportion)
{
	// Create canvas to plot on
	vector<Mat> channels;

	for (auto i = 0; i < 3; ++i) {
		channels.push_back(Mat::ones(300, 180, CV_8UC1) * 125);
	}
	
	Mat figure;
	cv::merge(channels, figure);
	
	// Plot histogram
	// Find maximum value of histogram to scale
	double maxVal = 0;
	for (auto &val : histogram) {
		if (val > maxVal) {
			maxVal = val;
		}
	}

	double scale = maxVal > 0 ? 300 / maxVal : 1;

	// Constrain the scale value based on the scale needed to display a uniform distribution
	// with gaussianMixtureProportion == 0
	double uniformScale = 100 / uniformDist(0, uniformMaxRange, uniformStepSize);

	scale = uniformScale < scale ? uniformScale : scale;


	if (histogram.size() >= 180) {
		for (auto angle = 0; angle < 180; ++angle) {
			line(figure, Point(angle, 299), Point(angle, 300 - std::round(histogram[angle] * scale)), Scalar(255, 255, 255));
		}
	}

	if(rainParams.debug)
		imshow("Histogram", figure);

	if (rainParams.saveImg) 
		imwrite("Histogram.png", figure);

}

double BossuRainIntensityMeasurer::goodnessOfFitTest(const std::vector<double>& histogram,
	const double surface,
	const double gaussianMean,
	const double gaussianStdDev,
	const double gaussianMixtureProportion) {
	//Implementation of Goodness-Of-Fit / Kolmogrov-Smirnov test, pp. 8, equation 26
	
	std::vector<double> eCDF(180, 0);
	double D = 0.;
	double uniformMass = uniformDist(0, uniformMaxRange, uniformStepSize);

	//Compare the Emperical CDF with the CDF of the actual joint unifrom-Gaussian distribution
	//Save the largest distance between the two CDFs	

	for (auto angle = 0; angle < histogram.size(); ++angle) {
		double iterator = USERAD ? (angle + 1) * DEG2RAD : (angle + 1);
		double angle_val = USERAD ? angle * DEG2RAD : angle;

		//Calculate normalized eCDF value for the current angle 
		double prevECDF = angle > 0 ? eCDF[angle - 1] : 0.;
		eCDF[angle] = histogram[angle] / surface + prevECDF;

		//Calculate distribution CDFs
		double normalCDF = 0.5 * (1. + erf((angle_val - gaussianMean) / (gaussianStdDev*SQRT_TWO)));
		double uniformCDF = iterator * uniformMass;
		double combinedCDF = gaussianMixtureProportion*normalCDF + (1. - gaussianMixtureProportion)*uniformCDF;

		//Calcualte the absolute difference between the combined CDF and eCDF
		double diff = abs(combinedCDF - eCDF[angle]);
		if (diff != diff)
			D = 1.;
		else if (diff > D)
			D = diff;

		if (rainParams.debug)
			cout << "ECDF: " << eCDF[angle] << ", Gauss CDF: " << normalCDF << ", Uniform CDF: " << uniformCDF << ", combinedCDF: " << combinedCDF << ", diff: " << diff << endl;
	}

	if (rainParams.verbose)
		cout << "Goodness-Of-Fit test resulted in D: " << D << endl;

	return D;
}

void BossuRainIntensityMeasurer::plotGoodnessOfFitTest(const std::vector<double>& histogram,
	const double surface,
	const double gaussianMean,
	const double gaussianStdDev,
	const double gaussianMixtureProportion) {
	//Plot Goodness-Of-Fit / Kolmogrov-Smirnov test, pp. 8, equation 26
	//Plot the Emperical CDF and the CDF of the actual joint unifrom-Gaussian distribution

	//Calculate the Emperical CDF of the Histogram
	std::vector<double> eCDF(180, 0);
	double uniformMass = uniformDist(0, uniformMaxRange, uniformStepSize);

	// Create canvas to plot on
	vector<Mat> channels;

	for (auto i = 0; i < 3; ++i) {
		channels.push_back(Mat::ones(300, 180, CV_8UC1) * 125);
	}

	Mat ECDFFigure, uCDFFigure, nCDFFigure, cCDFFigure;
	cv::merge(channels, ECDFFigure);
	ECDFFigure.copyTo(uCDFFigure);
	ECDFFigure.copyTo(nCDFFigure);
	ECDFFigure.copyTo(cCDFFigure);
	double scale = 300;

	for (auto angle = 0; angle < histogram.size(); ++angle) {
		double iterator = USERAD ? (angle + 1) * DEG2RAD : (angle + 1);
		double angle_val = USERAD ? angle * DEG2RAD : angle;

		//Calculate normalized eCDF value for the current angle 
		double prevECDF = angle > 0 ? eCDF[angle - 1] : 0.;
		eCDF[angle] = histogram[angle] / surface + prevECDF;

		//Calculate distribution CDFs
		double normalCDF = 0.5 * (1. + erf((angle_val - gaussianMean) / (gaussianStdDev*SQRT_TWO)));
		double uniformCDF = iterator * uniformMass;
		double combinedCDF = gaussianMixtureProportion*normalCDF + (1. - gaussianMixtureProportion)*uniformCDF;

		line(ECDFFigure, Point(angle, 299), Point(angle, 300 - std::round(eCDF[angle] * scale)), Scalar(255, 255, 255));
		line(uCDFFigure, Point(angle, 299), Point(angle, 300 - std::round(uniformCDF * scale)), Scalar(255, 255, 255));
		line(nCDFFigure, Point(angle, 299), Point(angle, 300 - std::round(normalCDF * scale)), Scalar(255, 255, 255));
		line(cCDFFigure, Point(angle, 299), Point(angle, 300 - std::round(combinedCDF * scale)), Scalar(255, 255, 255));
	}

	if (rainParams.saveImg) {
		imwrite("ECDF.png", ECDFFigure);
		imwrite("Uniform CDF.png", uCDFFigure);
		imwrite("Normal CDF.png", nCDFFigure);
		imwrite("Combined CDF.png", cCDFFigure);
	}
	if (rainParams.debug) {
		imshow("ECDF", ECDFFigure);
		imshow("Uniform CDF", uCDFFigure);
		imshow("Normal CDF", nCDFFigure);
		imshow("Combined CDF", cCDFFigure);
	}
}

double BossuRainIntensityMeasurer::uniformDist(double a, double b, double pos)
{
	assert(b > a);

	if (pos >= a && pos <= b) {
		return 1. / (b - a);
	}

	return 0.;
}

double BossuRainIntensityMeasurer::gaussianDist(double mean, double stdDev, double pos)
{
	double result = 0;

	if (stdDev != 0) {
		double variance = pow(stdDev, 2);
		result = 1 / sqrt(TWO_PI * variance) *
			exp(-pow(pos - mean, 2) / (2 * variance));
	}
	else if (mean == pos) {
		result = 1.;
	}

	return result;
}

int main(int argc, char** argv)
{
	const string keys =
		"{help h            |       | Print help message}"
		"{fileName fn       |       | Input video file to process}"
		"{filePath fp       |       | Filepath of the input file}"
		"{outputFolder of   |       | Output folder of processed files}"
		"{settingsFile sf   |       | File from which settings are loaded/saved}"
		"{saveSettings s    | 0     | Save settings to settingsFile (0,1)}"	
		"{saveImage i       | 0     | Save images from intermediate processing}"
		"{verbose v         | 0     | Write additional debug information to console}"
		"{debug d           | 0     | Enables debug mode. Writes extra information to console and shows intermediate images}"
		;

	cv::CommandLineParser cmd(argc, argv, keys);

	if (argc <= 1 || (cmd.has("help"))) {
		std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
		std::cout << "Available options: " << std::endl;
		cmd.printMessage();
		cmd.printErrors();
		std::cout << "Current arguments: " << endl;
		std::cout << **argv << endl;
		return 1;
	}

	for (int i = 0; i < argc; i++)
		cout << argv[i] << endl;
		
	std::string filename = cmd.get<std::string>("fileName");
	std::string filePath = cmd.get<std::string>("filePath");
	std::string outputFolder = cmd.get<string>("outputFolder");
	std::string settingsFile = cmd.get<string>("settingsFile");

	if ((settingsFile == "") && (cmd.get<int>("saveSettings") != 0))
		settingsFile = filename + "_" + "Settings.txt";

	BossuRainParameters defaultParams = BossuRainIntensityMeasurer::getDefaultParameters();

	if (cmd.has("settingsFile")) {
		defaultParams = BossuRainIntensityMeasurer::loadParameters(outputFolder + "/" + settingsFile);
	}

	// Set parameters here
	defaultParams.saveImg = cmd.get<int>("saveImage") != 0;
	defaultParams.verbose = cmd.get<int>("verbose") != 0;
	defaultParams.debug = cmd.get<int>("debug") != 0;

	BossuRainIntensityMeasurer bossuIntensityMeasurer(filename, filePath, settingsFile, outputFolder, defaultParams);

	// Save final settings
	if (cmd.get<int>("saveSettings") != 0) {
		bossuIntensityMeasurer.saveParameters(outputFolder + "/" + settingsFile);
	}

	bossuIntensityMeasurer.detectRain();


}
 