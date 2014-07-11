//!<
//!< OpenCV version > git clone 2014/Jun/03
//!<

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <err.h>

#include <cv.hpp> //!< OpenCV C++ class. not "cv.h" which is for C.

#include "elas.h" //!< Geiger's ACCV2010 paper implementation

#include <boost/thread/thread.hpp>
#include <boost/graph/graph_concepts.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
//#include "tracklets.h"


float NCC(cv::Mat& currentFrame, cv::Mat& targetFrame, cv::Mat& curPixel, cv::Mat& tarPixel,int patchSize)
{
	cv::Mat result;
	
	//*******************************************
	// use ROI to get image patch
	//*******************************************
	float u = curPixel.at<float>(0, 0);
	float v = curPixel.at<float>(0, 1);
	cv::Mat curPatch;
	cv::Rect curPosition(u - patchSize/2, v - patchSize /2, patchSize, patchSize);
	currentFrame(curPosition).copyTo(curPatch);
	
	
	u = tarPixel.at<float>(0, 0);
	v = tarPixel.at<float>(0, 1);
	cv::Mat tarPatch;
	cv::Rect tarPosition(u - patchSize/2, v - patchSize /2, patchSize, patchSize);

	
	
	cv::matchTemplate(curPatch, tarPatch, result, CV_TM_CCORR_NORMED);
	return result.at<double>(patchSize/2, patchSize/2);
};


	
void loadCalibrationKITTI(std::string infile, float baseline, float focus, float u0, float v0 )
{
	// calib_cam_to_cam.txt: Camera-to-camera calibration
	// --------------------------------------------------
	// 
	//   - S_xx: 1x2 size of image xx before rectification
	//   - K_xx: 3x3 calibration matrix of camera xx before rectification
	//   - D_xx: 1x5 distortion vector of camera xx before rectification
	//   - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
	//   - T_xx: 3x1 translation vector of camera xx (extrinsic)
	//   - S_rect_xx: 1x2 size of image xx after rectification
	//   - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
	//   - P_rect_xx: 3x4 projection matrix after rectification
	// 
	// Note: When using this dataset you will most likely need to access only
	// P_rect_xx, as this matrix is valid for the rectified image sequences.
	
	// load calibration file
	fstream input(infile.c_str(), ios::in);
	if(!input.good()){
		cerr << "Could not read file: " << infile << endl;
		exit(EXIT_FAILURE);
	}
	
	// read matrix P, for the two colored camera
	
	input.close();
};


void loadPositionKITTI(std::string infile, std::vector<cv::Mat>& rotationList, std::vector<cv::Mat>& translationList)
{

};


int main( int /*argc*/, char** /*argv*/ )
{

    //*******************************************
    //Load calibration matrix, camera position, 
    //and image pairs. Init variables.
    //*******************************************
    std::string infile;
    
    //!< load camera calibration info
	//need to learn about the format of KITTI
	float baseline;
	float focus;
	float u0;
	float v0;
	loadCalibrationKITTI(infile, baseline, focus, u0, v0 );
  

    //!< load camera position info
	cv::Mat rotation(3, 3, CV_32F);
	cv::Mat translation(1, 3, CV_32F);
	std::vector<cv::Mat> rotationList;
	std::vector<cv::Mat> translationList;
	loadPositionKITTI(infile, rotationList, translationList);
	
	
	//!< load image pairs
	std::vector<cv::Mat> disparityList;
	std::vector<cv::Mat> LRefectedList;
	std::vector<cv::Mat> RRefectedList;
	std::vector<cv::Mat> LNormalizedList;
	
	
	//!<load images obtained by left color camera
	{
		std::string dirname = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/";
		struct dirent **namelist;
		int fileNum = scandir(dirname.c_str(), &namelist, NULL, alphasort);
		if(fileNum == -1) {
			err(EXIT_FAILURE, "%s", dirname.c_str());
		}
		//(void) printf ("%d\n", fileNum);
		for (int i = 2; i < fileNum; ++i) 
		{
			//(void) printf("%s\n", namelist[i]->d_name);
			
			//std::cout<<dirname + namelist[i]->d_name<<std::endl;
			cv::Mat pic = cv::imread(dirname + namelist[i]->d_name, 1);
			LRefectedList.push_back(pic);
			
			free(namelist[i]);
		}
		free(namelist);
	}
	
	
	//!<load images obtained by right color camera
	{
		std::string dirname = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/";
		struct dirent **namelist;
		int fileNum = scandir(dirname.c_str(), &namelist, NULL, alphasort);
		if(fileNum == -1) {
			err(EXIT_FAILURE, "%s", dirname.c_str());
		}
		//(void) printf ("%d\n", fileNum);
		for (int i = 2; i < fileNum; ++i) 
		{
			//(void) printf("%s\n", namelist[i]->d_name);
			
			//std::cout<<dirname + namelist[i]->d_name<<std::endl;
			cv::Mat pic = cv::imread(dirname + namelist[i]->d_name, 1);
			RRefectedList.push_back(pic);
			
			free(namelist[i]);
		}
		free(namelist);
	}
	
	return 0;
    

	//!< init variables
	
	Elas::parameters param;
	param.postprocess_only_left = true;
	param.disp_min = 0;
	param.disp_max = (9 + 1) * 16;
	const int32_t dims[3] = {640, 480, 640};
	Elas elas( param );
	
	//init paramaters
	float Tcov = 0.5;
	float Tdist = 0.5;
	float Tphoto = 0.7;
	float patchSize = 7;
	float pointingError = 0.5;
	float matchingError = 1.0;
	
	
	pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
	pcl::PointCloud<pcl::PointXYZRGB> currentFrameCloud;
	

	cv::Mat K(3, 3, CV_32F);
	K.at<float>(0,0) = focus;
	K.at<float>(1,1) = focus;
	K.at<float>(2,2) = 1.0;
	K.at<float>(0,2) = u0;
	K.at<float>(1,2) = v0;
	
	
	const int m = 3; 
	const int r = (m - 1) / 2;
	
    while ( true )
    {
		//keyframe selected by minimum distance of camera motion
		//maybe I can use some simple frame skip to instead it
		static int framecnt = 0;
		framecnt++;
		//bool isKeyframe;
		//if ( !isKeyframe )
		//	continue;
		
		//*******************************************
		//Dense stereo matching by ELAS
		//*******************************************
		cv::Mat imageDisparity;
		cv::Mat& leftRefected = LRefectedList.at(framecnt);
		cv::Mat& rightRefected = RRefectedList.at(framecnt);
		elas.process(leftRefected.data, rightRefected.data, (float*)imageDisparity.data,(float*)imageDisparity.data, dims);
		
		
		//!<This vector is work like that:
		//!<record m frames into list, choose the center one as keyframe, do sth
		//!<once finished, discard the list and wait for new m frames
		disparityList.push_back(imageDisparity);
		
		
		
		//*******************************************
		//claculate the Normalized zero mean and unit variance 
		//left rectified RGB image
		//*******************************************
		//!<convert color image to grey scale 
		cv::Mat gray;
		cv::Mat grayNormalized;
		cv::cvtColor(leftRefected,gray,CV_BGR2GRAY);
		cv::Scalar mean = cv::mean(gray);
		gray = gray - mean;
		
		double minVal, maxVal;
		minMaxLoc( gray, &minVal, &maxVal );
		gray.convertTo(grayNormalized, CV_32F, 1/(maxVal - minVal), -0.5f);
		
		LNormalizedList.push_back(grayNormalized);
		
		if (framecnt % m != 0)
			continue;
		

		
		
		
		for ( int i = 0; i < leftRefected.rows; i++ )
		{
			for ( int j = 0; j < leftRefected.cols; j++ )
			{
				//*******************************************
				//Geometric check
				//*******************************************
				
				//float p = LRefectedList[framecnt - m + r].at<float>(i, j);
				float d = disparityList[framecnt - m + r].at<float>(i, j);
				
				//!<compute 3D point
				float z = focus * (baseline / d);
				float x = z * (j - u0) / focus;
				float y = z * (i - v0) / focus;
				cv::Mat Yi(1, 3, CV_32F, 0);
				Yi.at<float>(0,0) = x;
				Yi.at<float>(0,1) = y;
				Yi.at<float>(0,2) = z;
				
				//!<compute covariance Pi
				//Pi = Ji * Si * Ji'
				cv::Mat Si(3, 3, CV_32F, 0);
				Si.at<float>(0,0) = pointingError * pointingError;
				Si.at<float>(1,1) = pointingError * pointingError;
				Si.at<float>(2,2) = matchingError * matchingError;
				
				cv::Mat Ji(3, 3, CV_32F, 0);
				Si.at<float>(0,0) = baseline / d;
				Si.at<float>(0,2) = - (j * baseline / (d*d) );
				Si.at<float>(1,1) = baseline / d;
				Si.at<float>(1,2) = - (i * baseline / (d*d) );
				Si.at<float>(2,2) = - (focus * baseline / (d*d) );
				
				cv::Mat Pi(3, 3, CV_32F, 0);
				Pi = Ji * Si * Ji.t();
				float w = Pi.at<float>(0,0) + Pi.at<float>(1,1) + Pi.at<float>(2,2);
				
				//!<check if wi < Tcov, the first check 
				if (w > Tcov)
					continue;
				
				
				//reproject 3D point to other view to check the disparity, the second check
				std::vector<float> weightList;
				std::vector<cv::Mat> YiList;
				bool flgReproject = true;
				for (int k = 0; k < m; k++)
				{
					cv::Mat& rotation = rotationList.at(framecnt - m + k);
					cv::Mat& translation = translationList.at(framecnt - m + k);
					cv::Mat& disparity = disparityList.at(framecnt - m + k);
					cv::Mat Ui(1, 3, CV_32F, 0);
					Ui = K * (rotation * (Yi - translation) );
					
					//Now we got the pixel's position, let's check the disparity
					float u = Ui.at<float>(0,0);
					float v = Ui.at<float>(0,1);
					float d = disparity.at<float>(u, v);
					
					//check if disparity is valid
					if ( !(d > 0 && d < param.disp_max))
					{
						flgReproject = false;
						break;
					}

					//!<check if disparity is low uncertainty
					//!<Use the same method with wi's check
					//!<since the Si is already set, we need only to refill the Ji
					
					Si.at<float>(0,0) = baseline / d;
					Si.at<float>(0,2) = - (j * baseline / (d*d) );
					Si.at<float>(1,1) = baseline / d;
					Si.at<float>(1,2) = - (i * baseline / (d*d) );
					Si.at<float>(2,2) = - (focus * baseline / (d*d) );
					
					Pi = Ji * Si * Ji.t();
					float w = Pi.at<float>(0,0) + Pi.at<float>(1,1) + Pi.at<float>(2,2);
					
					if (w > Tcov)
					{
						flgReproject = true;
						break;
					}
					weightList.push_back(w);
					
					
					//!<
					//!<recompute point's 3D coordinate with this frame's disparity
					float z = focus * (baseline / d);
					float x = z * (j - u0) / focus;
					float y = z * (i - v0) / focus;
					cv::Mat Yi(1, 3, CV_32F, 0);
					Yi.at<float>(0,0) = x;
					Yi.at<float>(0,1) = y;
					Yi.at<float>(0,2) = z;
					YiList.push_back(Yi);
				}
				
				
				if(flgReproject == false)
					continue;
				
				
				//!<check 3D difference between all reconstructed 3D points is within Tdist
				float mindist = 99999;
				for(int j = 0; j < globalCloud.size(); j++)
				{
					float x1 = globalCloud.at(j).x;
					float y1 = globalCloud.at(j).y;
					float z1 = globalCloud.at(j).z;
					float dis = (x-x1) * (x-x1) + (y-y1) * (y-y1) + (z-z1) * (z-z1);
					if (dis < mindist)
						mindist = dis;
				}
				if (mindist > Tdist)
					continue;
				
				
				
				//*******************************************
				//Photometric check
				//*******************************************
				//Let's do this with a grey scale image first
				std::vector<float> NCCSList;
				std::vector<cv::Vec3b> colorList;
				float gp = 0;
				for (int k = 0; k < m; k++)
				{
					cv::Mat curPixel;
					cv::Mat tarPixel;
				
				
					//calculate 2D position in target frame
					cv::Mat& rotation = rotationList.at(framecnt - m + k);
					cv::Mat& translation = translationList.at(framecnt - m + k);
					cv::Mat& disparity = disparityList.at(framecnt - m + k);
					cv::Mat Ui(1, 3, CV_32F, 0);
					Ui = K * (rotation * (Yi - translation) );
					tarPixel = Ui;
					
					//save pixel value in target frame
					float u = tarPixel.at<float>(0, 0);
					float v = tarPixel.at<float>(0, 1);
					cv::Vec3b bgr = LRefectedList[framecnt - m + k].at<cv::Vec3b>(u,v);
					colorList.push_back(bgr);
					
					
					//calculate 2D position in current frame
					rotation = rotationList.at(framecnt - m + r);
					translation = translationList.at(framecnt - m + r);
					disparity = disparityList.at(framecnt - m + r);
					Ui = K * (rotation * (Yi - translation) );
					curPixel = Ui;
				
					float NCCScore = NCC(LNormalizedList.at(framecnt -m + r), LNormalizedList.at(framecnt - m + k), curPixel, tarPixel, patchSize);
					gp = gp + NCCScore;
					
				}
				gp = gp / 3;
				if (gp > Tphoto)
					continue;
				
				
				//<!if the pixel passed the both test, fuse it in to point cloud
				pcl::PointXYZRGB point;
				point.x = 0;
				point.y = 0;
				point.z = 0;
				float weightSum = 0;
				
				int pb = 0;
				int pg = 0;
				int pr = 0;
				for (int i = 0; i < m; i++)
				{
					point.x += YiList[i].at<float>(0, 0) * weightList[i];
					point.y += YiList[i].at<float>(0, 1) * weightList[i];
					point.z += YiList[i].at<float>(0, 2) * weightList[i];
					weightSum = weightSum + weightList[i];
					
					pb = colorList[i].val[0] * weightList[i];
					pg = colorList[i].val[1] * weightList[i];
					pr = colorList[i].val[2] * weightList[i];
					
				}
				point.x = point.x / weightSum;
				point.y = point.y / weightSum;
				point.z = point.z / weightSum;
				pb /= weightSum;
				pg /= weightSum;
				pr /= weightSum;
				
				uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
				point.rgb = *reinterpret_cast<float*>(&rgb);
				
				currentFrameCloud.push_back(point);
			}
		}
		
		
		//*******************************************
		//Remove outliers
		//*******************************************
		for (int i = 0; i < currentFrameCloud.size(); i++)
		{
			//something can be done by PCL, lucky
			
			
			//!<Use radius removal filter
			pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
			// build the filter
			outrem.setInputCloud(currentFrameCloud.makeShared());
			outrem.setRadiusSearch(0.8);
			outrem.setMinNeighborsInRadius (2);
			// apply filter
			outrem.filter (currentFrameCloud);
			
			//!<Use voxel grid to down sample
			pcl::VoxelGrid<pcl::PointXYZRGB> sor;
			sor.setInputCloud (globalCloud.makeShared());
			sor.setLeafSize (0.01f, 0.01f, 0.01f);
			sor.filter (globalCloud);
		}
		
		//try save all those information to memory first
		//disparityList.clear();
		for (int i = 0; i < currentFrameCloud.size(); i++)
		{
			globalCloud.push_back(currentFrameCloud.at(i));
		}
	}
	
	
	//*******************************************
	//Voxel grid filtering
	//*******************************************
	
	//!<Use voxel grid to down sample
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud (globalCloud.makeShared());
	vox.setLeafSize (0.01f, 0.01f, 0.01f);
	vox.filter (globalCloud);
	
	pcl::io::savePCDFileASCII ("scene.pcd", globalCloud);
	
    return 0;
}


