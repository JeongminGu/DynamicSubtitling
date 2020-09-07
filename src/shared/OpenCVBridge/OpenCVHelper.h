#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video.hpp>

#include <C:\Project\dlib-19.17\dlib-19.17\dlib\opencv.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing\frontal_face_detector.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing\render_face_detections.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\image_processing.h>
#include <C:\Project\dlib-19.17\dlib-19.17\dlib\gui_widgets.h>

#include <thread>
#include <vector>
#include <iostream>

namespace OpenCVBridge
{
	public ref class OpenCVHelper sealed
	{
	public:
		OpenCVHelper();
		// Image processing operators
		void Main(Windows::Graphics::Imaging::SoftwareBitmap^ input, Windows::Graphics::Imaging::SoftwareBitmap^ output, Platform::String^ result);

	private:
		struct Word {
			std::string text;
			std::vector<std::string> ponetics;
			int speakerIdx;
		};

		struct Sentence {
			std::vector<Word> words;
		};
		Word word;
		Sentence sentence;
		std::vector <Sentence> paragraph;
		std::vector<std::string> audioBuffer;
		std::string rawCstring = " ";
		std::string oldSubtitle = " ";
		double est_value;
		float distances[30] = { 0.0f, };
		double di1[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
		double di2[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
		double d1d2[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
		double faceXposBuf[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
		double faceYposBuf[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
		double svmresult[5] = { 999.0, 999.0, 999.0, 999.0, 999.0 };
		double svmResultBuf[5] = { 999.0, 999.0, 999.0, 999.0, 999.0 };
		std::vector< std::vector<int> > visualBuffer;
		bool matchingFLAG = false;
		int wordPivot = 0;
		int audioPivot = 0;
		int activeSpeakerIdx = -1;
		std::vector<std::string> subtitleBuf;
		std::vector<std::string> completeAudioBuf;
		int trig = 0;
		int faceNumBuf = 0;
		bool setBackgroundTrig;
		bool setSubtitleTrig;
		float sum_distance;
		double scale = 0.5;
		bool svm_trig = true;
		cv::Mat fgMaskMOG2;
		cv::Ptr<cv::BackgroundSubtractor> pMOG2;
		//face detector member variant
		dlib::frontal_face_detector detector;
		dlib::shape_predictor pose_model;
		dlib::array2d<unsigned char> img_gray;
		// helper functions for getting a cv::Mat from SoftwareBitmap
		bool GetPointerToPixelData(Windows::Graphics::Imaging::SoftwareBitmap^ bitmap, unsigned char** pPixelData, unsigned int* capacity);
		bool TryConvert(Windows::Graphics::Imaging::SoftwareBitmap^ from, cv::Mat& convertedMat);
		size_t split(const std::string& txt, std::vector<std::string>& strs, char ch);
		std::vector<std::string> dict(std::string word);
		double L2Dist(double x1, double y1, double x2, double y2);
		double simCheck(int audio_num, int visual_num);
		double OpenCVHelper::minimumDistance(double* spDist, int faceNum, int& activeSpeakerIdx);
		double FeatureL2Dist(double* f_a, double* f_b);
		void render_sub(cv::Mat frame, std::string sub, int cx, int cy);
		bool setActiveSpeakTrig = true;
		bool setDebugtrig = false;
		int aaa = 0;
		int bbb = 0;
		double globalMinDist = -1.0;
		double svmInference = -1.0;
	};
}
