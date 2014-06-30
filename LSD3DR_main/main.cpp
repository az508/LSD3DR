//!<
//!< OpenCV version > git clone 2014/Jun/03
//!<

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <cv.hpp> //!< OpenCV C++ class. not "cv.h" which is for C.

#include "elas.h" //!< Geiger's ACCV2010 paper implementation

#include <boost/thread/thread.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>



//! \brief load calibration info used by cv::remap()
//!
//! \return cv::Mat& mx1
//! \return cv::Mat& my1
//! \return cv::Mat& mx2
//! \return cv::Mat& my2
//!
//!
void loadCalibrationInfo(cv::Mat& mx1,
                         cv::Mat& my1,
                         cv::Mat& mx2,
                         cv::Mat& my2,
			 cv::Mat& Q
			)
{
    cv::FileStorage fs;
    
    fs.open("../calibrationResult/Q.xml",   cv::FileStorage::READ);
    fs["Q"] >> Q;
    fs.release();

    fs.open("../calibrationResult/mx1.xml", cv::FileStorage::READ);
    fs["mx1"] >> mx1;
    fs.release();

    fs.open("../calibrationResult/mx2.xml", cv::FileStorage::READ);
    fs["mx2"] >> mx2;
    fs.release();

    fs.open("../calibrationResult/my1.xml", cv::FileStorage::READ);
    fs["my1"] >> my1;
    fs.release();

    fs.open("../calibrationResult/my2.xml", cv::FileStorage::READ);
    fs["my2"] >> my2;
    fs.release();
}


int main( int /*argc*/, char** /*argv*/ )
{

    //*******************************************
    //Load calibration matrix, camera position, 
    //and image pairs. Init variables.
    //*******************************************
    
    
    //!< load camera calibration info
    cv::Mat mx1, my1, mx2, my2, Q;
    loadCalibrationInfo(mx1, my1, mx2, my2, Q);
	//Get the interesting parameters from Q
	double Q03, Q13, Q23, Q32, Q33;
	Q03 = Q.at<double>(0,3);
	Q13 = Q.at<double>(1,3);
	Q23 = Q.at<double>(2,3);
	Q32 = Q.at<double>(3,2);
	Q33 = Q.at<double>(3,3);
  

    //!< load camera position info
	cv::Mat rotation(3, 3, CV_32F)[5];
	cv::Mat translation(1, 3, CV_32F)[5];
	
	
	//!< load image pairs
    

	//!< init variables
	cv::Mat imageDisparity(480, 640, CV_32F);
	cv::Mat leftRefected(480, 640, CV_32F);
	cv::Mat rightRefected(480, 640, CV_32F);
	cv::Mat leftNormalized(480, 640, CV_32F);

    
	Elas::parameters param;
	param.postprocess_only_left = true;
	param.disp_min = 0;
	param.disp_max = (9 + 1) * 16;
	const int32_t dims[3] = {640, 480, 640};
	Elas elas( param );
	
	float baseline;
	float focus;

	//init paramaters
	float Tcov;
	float Tdist;
	float Tphoto;
	float patchSize;
	float pointingError;
	float matchingError;
	
    while ( true )
    {
		
		bool isKeyframe;
		if ( !isKeyframe )
			continue;
		
		//*******************************************
		//Dense stereo matching by ELAS
		//*******************************************
		elas.process(leftRefected.data, leftRefected.data, (float*)imageDisparity.data,(float*)imageDisparity.data, dims);

		//*******************************************
		//Geometric check
		//*******************************************
		for ( int i = 0; i < leftRefected.rows; i++ )
		{
			for ( int j = 0; j < leftRefected.cols; j++ )
			{
				float p = leftRefected.at<float>(i, j);
				float d = imageDisparity.at<float>(i, j);
				
				//!<compute 3D point
				
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
				float w = cv::trace(Pi);
				
				//!<if wi < Tcov, 
				//reproject 3D point to other view to check the disparity
				if (w > Tcov)
					continue;
				
				
			}
		}


		//*******************************************
		//Photometric check
		//*******************************************

		//*******************************************
		//Remove outliers
		//*******************************************

	}
	
	
	//*******************************************
	//Voxel grid filtering
	//*******************************************
    return 0;
}


