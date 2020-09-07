#include "pch.h"
#include "OpenCVHelper.h"
#include "MemoryBuffer.h"
#include <iostream>
#include <stdio.h>
#include <codecvt>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\opencv.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing\frontal_face_detector.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing\render_face_detections.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\gui_widgets.h>
#include <cvt/wstring>
#include <thread>
#include "svm-predict.c"
#include <time.h>

using namespace Microsoft::WRL;
using namespace OpenCVBridge;
using namespace Platform;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Storage::Streams;
using namespace Windows::Foundation;
using namespace cv;

OpenCVHelper::OpenCVHelper()
{
    pMOG2 = createBackgroundSubtractorMOG2();
	try
	{
		detector = dlib::get_frontal_face_detector();
		//Load face binary data
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model; // face detection model
		paragraph.clear();
		setBackgroundTrig = false;
		setSubtitleTrig = false;
		const char* model_path = "model0502.txt"; // set svm model path
		model = svm_load_model(model_path);
		svm_type = svm_get_svm_type(model); // SVM Type : enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR }
		nr_class = svm_get_nr_class(model);

		for (int i = 0; i < 10; ++i) {
			std::vector<int> element(10);
			visualBuffer.push_back(element);
		}
		// Init visual buffer
		for (int i = 0; i < 10; ++i) {
			visualBuffer[i].clear();
		}
		visualBuffer.clear();
		subtitleBuf.resize(10, "");
	}
	catch (dlib::serialization_error& e)
	{
		return ;
	}
	catch (std::exception& e)
	{
		OutputDebugStringA("Initialization failed.");
		OutputDebugStringA(e.what());
	}
}

void OpenCVHelper::Main(SoftwareBitmap^ input, SoftwareBitmap^ output, Platform::String^ result)
{
	Mat inputMat, outputMat, grayMat, resizeMat;	//Initialization image array
	
	stdext::cvt::wstring_convert<std::codecvt_utf8<wchar_t>> convert;	// converts C# String to C++ String
	std::string stringUtf8 = convert.to_bytes(result->Data());
	const char* rawCstring = stringUtf8.c_str();
	std::string audioStream = rawCstring;
	if ((audioStream != "") && (audioStream != oldSubtitle)) {
		split(audioStream, audioBuffer, ' ');
		sentence.words.resize(audioBuffer.size());				// resize vector buffer
		for (int i = 0; i < audioBuffer.size(); ++i) {
			std::transform(audioBuffer[i].begin(), audioBuffer[i].end(), audioBuffer[i].begin(), ::tolower);
			sentence.words[i].text = audioBuffer[i];
			sentence.words[i].ponetics = dict(audioBuffer[i]);	// translate into phonetic symbol
		}
		paragraph.push_back(sentence);
		matchingFLAG = true;
	}
    if (!(TryConvert(input, inputMat) && TryConvert(output, outputMat)))	//Check frame is empty
    {
        return;
    }
	
	inputMat.copyTo(outputMat);
	
	unsigned char* pPixels = nullptr;

	cv::Mat bgrMat(input->PixelHeight, input->PixelWidth, CV_8UC3);	// Create BGR frame
	cv::cvtColor(outputMat, bgrMat, COLOR_BGRA2BGR);				// Copy frame into BGR frame
	cv::resize(bgrMat, resizeMat, cv::Size(), 0.5, 0.5);			// Resize frame to boost the face detection

	try
	{
		dlib::array2d<unsigned char> img_gray;
		dlib::cv_image<dlib::bgr_pixel> cvImage(resizeMat);			//CV::MAT to Dlib
		dlib::assign_image(img_gray, cvImage);						//Get gray-scale image
		std::vector<dlib::rectangle> faces = detector(img_gray);	//Face detection
		std::vector<dlib::full_object_detection> shapes;

		if (faces.size() != 0) {
			int* cx = new int[faces.size()]();
			int* cy = new int[faces.size()]();
			for (unsigned long i = 0; i < faces.size(); ++i)
			{
				setBackgroundTrig = true;
				std::vector <dlib::rectangle> facesBuf = faces;

				int* idx = new int[faces.size()];						//re-ordering rectangle region
				for (int i = 0; i < faces.size(); ++i) {
					idx[i] = dlib::center(facesBuf[i]).x();
				}
				std::sort(idx, idx + faces.size());
				for (int i = 0; i < faces.size(); ++i) {
					for (int j = 0; j < faces.size(); ++j) {
						if (idx[i] == dlib::center(facesBuf[j]).x())
							faces[i] = facesBuf[j];
					}
				}
				delete[] idx;											//Flush(), ordering indices

				double** arr_X = new double*[faces.size()];				// Set mouth landmark array of X pos in each speaker candidates
				double** arr_Y = new double*[faces.size()];				// Set mouth landmark array of Y pos in each speaker candidates
				double** featureSpeaker = new double*[faces.size()];	// Set F_{i} for each speaker candidates
				double* avgLandmark_X = new double[faces.size()]();		// Set avg landmark position of X pos
				double* avgLandmark_Y = new double[faces.size()]();		// Set avg landmark position of Y pos
				double* mThreshold_new = new double[faces.size()]();	// Set SVM Threshold
				double* mThreshold_old = new double[faces.size()]();
				for (int i = 0; i < faces.size(); ++i) {
					arr_X[i] = new double[20];
					arr_Y[i] = new double[20];
					featureSpeaker[i] = new double[20];
					memset(arr_X[i], 0.0, sizeof(double) * 20);
					memset(arr_Y[i], 0.0, sizeof(double) * 20);
					memset(featureSpeaker[i], 0.0, sizeof(double) * 20);
				}

				dlib::full_object_detection shape = pose_model(img_gray, faces[i]);		//Facial landmark detection
				std::vector<cv::Point> landmark;
				double* maxDist = new double[faces.size()]();

				for (int j = 0; j < faces.size(); ++j)
				{
					double xcenter = (shape.part(61).x() + shape.part(63).x()) / 2;
					double ycenter = (shape.part(61).y() + shape.part(67).y()) / 2;
					cx[j] = xcenter;
					cy[j] = ycenter;

					for (int i = 48; i < 68; ++i) {
						arr_X[j][i - 48] = shape.part(i).x();
						arr_Y[j][i - 48] = shape.part(i).y();
						avgLandmark_X[j] += shape.part(i).x();
						avgLandmark_Y[j] += shape.part(i).y();
					}

					double magnitude1 = sqrt(arr_X[j][13] * arr_X[j][13] + arr_Y[j][13] * arr_Y[j][13]);
					double magnitude2 = sqrt(arr_X[j][19] * arr_X[j][19] + arr_Y[j][19] * arr_Y[j][19]);
					double unitX1 = arr_X[j][13] / magnitude1;
					double unitY1 = arr_Y[j][13] / magnitude1;
					double unitX2 = arr_X[j][19] / magnitude2;
					double unitY2 = arr_Y[j][19] / magnitude2;

					avgLandmark_X[j] = (1.0 / 20.0) * avgLandmark_X[j];	//Compute average of X position
					avgLandmark_Y[j] = (1.0 / 20.0) * avgLandmark_Y[j]; //Compute average of Y position

					for (int i = 0; i < 20; ++i) {
						featureSpeaker[j][i] = L2Dist(arr_X[j][i], arr_Y[j][i], avgLandmark_X[j], avgLandmark_Y[j]);
						maxDist[j] = maxDist[j] + featureSpeaker[j][i] * featureSpeaker[j][i];
					}
					maxDist[j] = sqrt(maxDist[j]);

					for (int i = 0; i < 20; ++i) {						//Normalize
						featureSpeaker[j][i] = featureSpeaker[j][i] / maxDist[j];
					}

					if (trig == 1) {										//Compute t-1 and t distances
						if (faceNumBuf > faces.size()) {
							setSubtitleTrig = true;
							cx[j] = faceXposBuf[j];
							cy[j] = faceYposBuf[j];
						}
						else if (faceNumBuf < faces.size()) {
							setSubtitleTrig = false;
						}
						mThreshold_new[j] = L2Dist(unitX1, unitY1, unitX2, unitY2);
						cv::putText(outputMat, std::to_string(mThreshold_new[j]), cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 3);
						//value debug 0.013 , 0.030 -> 0.02
						d1d2[j] = fabs(mThreshold_new[j] - mThreshold_old[j]);
						mThreshold_old[j] = mThreshold_new[j];
						faceXposBuf[j] = cx[j];
						faceYposBuf[j] = cy[j];
					}
					if (trig == 0) {
						faceNumBuf = faces.size();	//Remember previous face number to robust face detection
						faceXposBuf[j] = cx[j];
						faceYposBuf[j] = cy[j];
						mThreshold_old[j] = L2Dist(unitX1, unitY1, unitX2, unitY2);	//Compute vector length here
						trig = 1;
					}
					if (d1d2[j] > 0.04) {				//Online inference
						kkkk38(svmresult, featureSpeaker, faces.size(), j, model);
						if (svmResultBuf[j] != svmresult[j]) {
							visualBuffer[j].push_back((int)svmresult[j]);
							svmInference = svmresult[j];
						}
						svmResultBuf[j] = svmresult[j];
					}
					else {
						if (svmResultBuf[j] != 0) {
							visualBuffer[j].push_back(0);
							//svmInference = 0.0;
							svmResultBuf[j] = 0;
						}
					}
				}
				for (unsigned long j = 0; j < shape.num_parts(); ++j)	//Drawing purpose
				{
					landmark.push_back(cv::Point(shape.part(j).x() * (1.0 / scale), shape.part(j).y() * (1.0 / scale)));
					if (j >= 48)
						cv::circle(outputMat, landmark[j], 2.0, Scalar(0, 255, 0, 255), 1, 8);
				}
				//Flush all arrays
				for (int i = 0; i < faces.size(); ++i) {
					delete[] arr_X[i];
				}
				delete[] arr_X;
				for (int i = 0; i < faces.size(); ++i) {
					delete[] arr_Y[i];
				}
				delete[] arr_Y;
				for (int i = 0; i < faces.size(); ++i) {
					delete[] featureSpeaker[i];
				}
				delete[] maxDist;
				delete[] avgLandmark_X;
				delete[] avgLandmark_Y;
				delete[] mThreshold_new;
				delete[] mThreshold_old;
			}
			//Speaker matching
			if ((matchingFLAG == true)) { 
				oldSubtitle = audioStream;
				if (wordPivot == 0) { 
					paragraph[audioPivot].words[wordPivot].text[0] = std::toupper(paragraph[audioPivot].words[wordPivot].text[0]);
				}
				for (int i = 0; i < faces.size(); ++i) { //Visual-audio size equalization
					if (visualBuffer[i].empty())
						visualBuffer[i].push_back(0);
					if (paragraph[audioPivot].words[wordPivot].ponetics.size() > visualBuffer[i].size()) {
						int offset = paragraph[audioPivot].words[wordPivot].ponetics.size() - visualBuffer[i].size();
						visualBuffer[i].resize(visualBuffer[i].size() + offset);
					}	// visual buffer smaller than audio buffer.. 
					else if (paragraph[audioPivot].words[wordPivot].ponetics.size() < visualBuffer[i].size()) {
						int offset = paragraph[audioPivot].words[wordPivot].ponetics.size() - visualBuffer[i].size();
						visualBuffer[i].resize(paragraph[audioPivot].words[wordPivot].ponetics.size());
					}	// visual buffer greater than audio buffer..	
				}	//Done equalization
				bbb = paragraph[audioPivot].words[wordPivot].ponetics.size();
				double* spDistances = new double[faces.size()]();	//Visual-audio matching
				for (int j = 0; j < faces.size(); ++j) {
					for (int i = 0; i < visualBuffer[j].size(); ++i) {
						spDistances[j] += simCheck(std::stoi(paragraph[audioPivot].words[wordPivot].ponetics[i]), (int)visualBuffer[j][i]);
					}
					spDistances[j] = spDistances[j] / ( (double) bbb);
				}
				activeSpeakerIdx = -1;	//Default as -1, off-screen
				
				double minDist = minimumDistance(spDistances, faces.size(), activeSpeakerIdx);	// Find minimum distance in each candidates
				globalMinDist = activeSpeakerIdx;
				delete[] spDistances;
					
				if (minDist >= 0.023) {
					if (wordPivot == 0) {	//Active speaker processing
						subtitleBuf[activeSpeakerIdx] = paragraph[audioPivot].words[wordPivot].text;
						subtitleBuf[activeSpeakerIdx] += " ";
					}
					else {
						subtitleBuf[activeSpeakerIdx] += paragraph[audioPivot].words[wordPivot].text;
						subtitleBuf[activeSpeakerIdx] += " ";
					}
				}
				else {
					if (wordPivot == 0) {	//Off-screen proessing
						subtitleBuf[subtitleBuf.size() - 1] = paragraph[audioPivot].words[wordPivot].text;
						subtitleBuf[subtitleBuf.size() - 1] += " ";
					}
					else {
						subtitleBuf[subtitleBuf.size() - 1] += paragraph[audioPivot].words[wordPivot].text;
						subtitleBuf[subtitleBuf.size() - 1] += " ";
					}
				}
				wordPivot++;
				if (wordPivot > paragraph[audioPivot].words.size() - 1) {
					wordPivot = 0;
					audioPivot++;
					for (int i = 0; i < 10; ++i) {
						visualBuffer[i].clear();
					}
					visualBuffer.clear();
					matchingFLAG = false;
				}
			}//Speaker matching
			if(activeSpeakerIdx != -1)
				render_sub(outputMat, subtitleBuf[activeSpeakerIdx], cx[activeSpeakerIdx], cy[activeSpeakerIdx]);	//Render off-screen subtitle
			
			render_sub(outputMat, subtitleBuf[subtitleBuf.size() - 1], 100, 100);	//Render off-screen subtitle
			render_sub(outputMat, std::to_string(activeSpeakerIdx), 100, 300);
		}
		if(faces.size() == 0) {
			if (matchingFLAG == true) {
				oldSubtitle = audioStream;
				if (wordPivot == 0) {
					paragraph[audioPivot].words[wordPivot].text[0] = std::toupper(paragraph[audioPivot].words[wordPivot].text[0]);
					subtitleBuf[subtitleBuf.size() - 1] = paragraph[audioPivot].words[wordPivot].text;
					subtitleBuf[subtitleBuf.size() - 1] += " ";
				}
				else {
					subtitleBuf[subtitleBuf.size() - 1] += paragraph[audioPivot].words[wordPivot].text;
					subtitleBuf[subtitleBuf.size() - 1] += " ";
				}
				wordPivot++;
				if (wordPivot > paragraph[audioPivot].words.size() - 1) {
					wordPivot = 0;
					audioPivot++;
					matchingFLAG = false;
					for (int i = 0; i < 10; ++i) {
						visualBuffer[i].clear();
					}
					visualBuffer.clear();
				}
			}
			render_sub(outputMat, subtitleBuf[subtitleBuf.size() - 1], 100, 100);
		}
		render_sub(outputMat, subtitleBuf[subtitleBuf.size() - 1], 100, 100);
		render_sub(outputMat, std::to_string(globalMinDist), 100, 300);
	}
	catch (std::exception& e)
	{
		OutputDebugStringA("Face detection failed!");
		OutputDebugStringA(e.what());
	}
}

//Extra functions...

bool OpenCVHelper::TryConvert(SoftwareBitmap^ from, Mat& convertedMat)
{
    unsigned char* pPixels = nullptr;
    unsigned int capacity = 0;
    if (!GetPointerToPixelData(from, &pPixels, &capacity))
    {
        return false;
    }

    Mat mat(from->PixelHeight,
        from->PixelWidth,
        CV_8UC4, // assume input SoftwareBitmap is BGRA8
        (void*)pPixels);

    // shallow copy because we want convertedMat.data = pPixels
    // don't use .copyTo or .clone
    convertedMat = mat;
    return true;
}

bool OpenCVHelper::GetPointerToPixelData(SoftwareBitmap^ bitmap, unsigned char** pPixelData, unsigned int* capacity)
{
    BitmapBuffer^ bmpBuffer = bitmap->LockBuffer(BitmapBufferAccessMode::ReadWrite);
    IMemoryBufferReference^ reference = bmpBuffer->CreateReference();

    ComPtr<IMemoryBufferByteAccess> pBufferByteAccess;
    if ((reinterpret_cast<IInspectable*>(reference)->QueryInterface(IID_PPV_ARGS(&pBufferByteAccess))) != S_OK)
    {
        return false;
    }

    if (pBufferByteAccess->GetBuffer(pPixelData, capacity) != S_OK)
    {
        return false;
    }
    return true;
}

size_t OpenCVHelper::split(const std::string& txt, std::vector<std::string>& strs, char ch)
{
	size_t pos = txt.find(ch);
	size_t initialPos = 0;
	strs.clear();

	// Decompose statement
	while (pos != std::string::npos) {
		strs.push_back(txt.substr(initialPos, pos - initialPos));
		initialPos = pos + 1;

		pos = txt.find(ch, initialPos);
	}

	// Add the last one
	strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));

	return strs.size();
}

std::vector<std::string> OpenCVHelper::dict(std::string word) {
	std::string search;
	int offset;
	std::string line;
	std::ifstream MyFile;
	MyFile.open("phonemes.txt");
	std::vector<std::string> dummy;
	dummy.push_back("0");
	if ((word[word.size() - 1] == ',') || (word[word.size() - 1] == '.') || (word[word.size() - 1] == '?')) {
		word.erase(word.size() - 1);
	}
	search = word;
	if (MyFile.is_open())
	{
		while (!MyFile.eof())
		{
			std::vector<std::string> splitLine;
			getline(MyFile, line);
			if ((offset = line.find(search, 0)) != std::string::npos)
			{
				split(line, splitLine, ' ');
				if (splitLine[0] == word) {
					splitLine.erase(splitLine.begin());
					return splitLine;
				}
				else {
					splitLine.clear();
					continue;
				}
			}
		}
		MyFile.close();
	}
	return dummy;
}

double OpenCVHelper::L2Dist(double x1, double y1, double x2, double y2) {
	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

double OpenCVHelper::simCheck(int audio_num, int visual_num) {
	//	num of vowels <-> reference distances
	double refDist[14][20] = {
	{0.199828, 0.603603, 1.0, 0.064485, 0.375752, 0.393715, 0.945623, 0.421621, 0.339384, 0.085379, 0.352686, 0.649066, 0.829157, 0.137087, 0.031703, 0.244974, 0.782807, 0.245046, 0.051487, 0.238043},
	{0.999686, 0.711431, 0.572693, 0.436316, 1.0, 0.679251, 0.950438, 0.772914, 0.664542, 0.633640, 0.653787, 0.783823, 0.811758, 0.299419, 0.129099, 0.300145, 0.761321, 0.321312, 0.179766, 0.305108},
	{1.0, 0.747948, 0.583350, 0.444292, 0.581249, 0.693751, 0.857365, 0.755571, 0.716527, 0.702008, 0.720944, 0.817335, 0.865471, 0.037489, 0.162413, 0.321267, 0.715233, 0.144497, 0.240821, 0.346016},
	{1.0, 0.678473, 0.480377, 0.346871, 0.488260, 0.644703, 0.918806, 0.713753, 0.568364, 0.541972, 0.578718, 0.750297, 0.844333, 0.272905, 0.102663, 0.274456, 0.761652, 0.286253, 0.175176, 0.287188},
	{1.0, 0.653001, 0.427299, 0.292155, 0.430497, 0.610024, 0.906085, 0.671982, 0.530959, 0.500814, 0.537368, 0.717511, 0.850240, 0.158653, 0.077236, 0.257140, 0.763439, 0.273755, 0.161123, 0.272361},
	{0.837281, 0.701518, 0.557549, 0.422148, 0.561081, 1.0, 0.897884, 0.766947, 0.682821, 0.660801, 0.687434, 0.814455, 0.820497, 0.304830, 0.141289, 0.304749, 0.709058, 0.323859, 0.206737, 0.313533},
	{0.499982, 0.711950, 0.548634, 0.412740, 0.552250, 0.675828, 1.0, 0.782458, 0.670737, 0.638511, 0.659652, 0.791595, 0.998418, 0.307780, 0.013521, 0.304019, 0.777021, 0.334236, 0.209455, 0.108610},
	{0.699922, 0.707259, 0.551329, 0.417396, 0.554764, 0.672977, 1.0, 0.776247, 0.673654, 0.653204, 0.677714, 0.812087, 0.920912, 0.296233, 0.124557, 0.292714, 0.757481, 0.316003, 0.093762, 0.307255},
	{1.0, 0.728874, 0.587487, 0.451488, 0.594775, 0.668525, 0.906640, 0.796629, 0.732854, 0.718462, 0.737312, 0.846157, 0.808057, 0.292780, 0.113775, 0.291255, 0.702636, 0.306261, 0.185339, 0.304875},
	{0.782241, 0.873032, 0.868003, 0.738107, 0.856285, 0.838979, 0.900836, 0.943764, 1.0, 0.981947, 0.978960, 0.910134, 0.732138, 0.541761, 0.448558, 0.524605, 0.759330, 0.588703, 0.543145, 0.544005},
	{1.0, 0.764570, 0.650423, 0.529627, 0.666030, 0.783469, 0.973423, 0.826415, 0.744457, 0.716580, 0.726217, 0.808324, 0.941496, 0.345375, 0.232935, 0.361675, 0.805601, 0.374707, 0.091751, 0.437686},
	{0.599749, 0.711705, 0.518697, 0.379880, 1.0, 0.699310, 0.962259, 0.753687, 0.629467, 0.589562, 0.610578, 0.750184, 0.843869, 0.294535, 0.122520, 0.300065, 0.803674, 0.326617, 0.178114, 0.988306},
	{0.399990, 0.779487, 0.707233, 0.573342, 0.703760, 0.748062, 0.936411, 0.894931, 0.878409, 0.877493, 0.871731, 0.917559, 0.789958, 0.307729, 0.165574, 0.102195, 1.0, 0.328715, 0.249542, 0.232609},
	{1.0, 0.717685, 0.588549, 0.461893, 0.591661, 0.692389, 0.913961, 0.850456, 0.806575, 0.789678, 0.793266, 0.868129, 0.786842, 0.278324, 0.074925, 0.281798, 0.708155, 0.306877, 0.157045, 0.283185} }; 
	double result;
	if ((visual_num == 0) || (audio_num == 0)) {
		result = 0.0083; // Penalty
	}
	else {
		result = FeatureL2Dist(refDist[visual_num - 1], refDist[audio_num]);
	}
	return result;
}

double OpenCVHelper::minimumDistance(double* spDist, int faceNum, int& activeSpeakerIdx) {
	double minDist = spDist[0];
	activeSpeakerIdx = -1;
	for (int i = 0; i < faceNum; ++i) {
		if ((spDist[i] > 0.023)) {
			minDist = spDist[i];
			activeSpeakerIdx = i;
		}
	}
	return minDist;
}

double OpenCVHelper::FeatureL2Dist(double* f_a, double* f_b) {
	double result = 0.0;
	for (int i = 0; i < 20; ++i) {
		result = result + (f_a[i] - f_b[i]) * (f_a[i] - f_b[i]);
	}
	result = sqrtf(result) * (1.0/20.0);
	return result;
}

void OpenCVHelper::render_sub(cv::Mat frame, std::string sub, int cx, int cy) {
	int thickness = 2;
	cv::Point location(cx, cy);
	int font = FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.8;

	putText(frame, sub, location, font, fontScale, Scalar(0, 255, 255), thickness);
}