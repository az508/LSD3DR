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


    //*******************************************
    //Take homogeneous transformation matrix,
    //Output 3 by 3 rotation and 1 by 3 translation marix
    //*******************************************
void homogeneous2RT( cv::Mat& H, cv::Mat& R, cv::Mat& T)
{
	T.at<float>(0, 0) = H.at<float>(0, 3);
	T.at<float>(0, 1) = H.at<float>(1, 3);
	T.at<float>(0, 2) = H.at<float>(2, 3);
	
	R.at<float>(0, 0) = H.at<float>(0, 0);
	R.at<float>(0, 1) = H.at<float>(0, 1);
	R.at<float>(0, 2) = H.at<float>(0, 2);
	R.at<float>(1, 0) = H.at<float>(1, 0);
	R.at<float>(1, 1) = H.at<float>(1, 1);
	R.at<float>(1, 2) = H.at<float>(1, 2);
	R.at<float>(2, 0) = H.at<float>(2, 0);
	R.at<float>(2, 1) = H.at<float>(2, 1);
	R.at<float>(2, 2) = H.at<float>(2, 2);
};


    //*******************************************
    //Take 3 by 3 rotation and 1 by 3 translation marix,
    //Output homogeneous transformation matrix
    //*******************************************
void RT2homogeneous( cv::Mat& H, cv::Mat& R, cv::Mat& T)
{
	
	H.at<float>(0, 0) = R.at<float>(0, 0);	H.at<float>(0, 1) = R.at<float>(0, 0);	H.at<float>(0, 2) = R.at<float>(0, 0);	H.at<float>(0, 3) = R.at<float>(0, 0);
	H.at<float>(1, 0) = R.at<float>(0, 0);	H.at<float>(1, 1) = R.at<float>(0, 0);	H.at<float>(1, 2) = R.at<float>(0, 0);	H.at<float>(1, 3) = R.at<float>(0, 0);
	H.at<float>(2, 0) = R.at<float>(0, 0);	H.at<float>(2, 1) = R.at<float>(0, 0);	H.at<float>(2, 2) = R.at<float>(0, 0);	H.at<float>(2, 3) = R.at<float>(0, 0);
	H.at<float>(3, 0) = 0				;	H.at<float>(3, 1) = 0				;	H.at<float>(3, 2) = 0				;	H.at<float>(3, 3) = 1				;

};

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
	targetFrame(tarPosition).copyTo(tarPatch);
	
	
	cv::matchTemplate(curPatch, tarPatch, result, CV_TM_CCORR_NORMED);
	return result.at<float>(patchSize/2, patchSize/2);
};


	
void loadCalibrationKITTI(std::string infile, float& baseline, float& focus, float& u0, float& v0, cv::Mat& R_cam0grayTOcam2color, cv::Mat& T_cam0grayTOcam2color, cv::Mat& R_cam2TOrectcam2 )
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
	
	std::string line;
	std::getline(input, line);//skip calib_time
	std::getline(input, line);//skip calib_dist
	//!<skip two grey scale camera
	for (int i = 0; i < 2; i++)
	{
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		std::getline(input, line);//skip R
		std::getline(input, line);//skip T
		
		std::getline(input, line);//skip Sr
		std::getline(input, line);//skip Rr
		std::getline(input, line);//skip Pr
	}
	//!<get calibration of two color camera

	float fb2, fb3;
	//left color
	{
		std::string matrixName;
		float dummy;
		
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		//read R
		input>>matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_cam0grayTOcam2color.at<float>(i, j);
			}
		}
		//read T
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_cam0grayTOcam2color.at<float>(0, i);
		}
		std::getline(input, line);//to next line
		std::getline(input, line);//skip Sr
		
		//read Rr
		input>>matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_cam2TOrectcam2.at<float>(i, j);
			}
		}
		
		//read Pr
		input>>matrixName;
		input>>focus;
		input>>dummy;
		input>>u0;
		input>>fb2;
		input>>dummy;
		input>>dummy;
		input>>v0;
		
		std::getline(input, line);//skip Pr
	}
	
	//right color
	float focus3;
	{
		std::getline(input, line);//skip S
		std::getline(input, line);//skip K
		std::getline(input, line);//skip D
		
		std::getline(input, line);//skip R
		std::getline(input, line);//skip T
		
		std::getline(input, line);//skip Sr
		std::getline(input, line);//skip Rr
		
		//read Pr
		std::string matrixName;
		float dummy;
		input>>matrixName;
		input>>focus3;
		input>>dummy;
		input>>dummy;
		input>>fb3;
		input>>dummy;
		input>>dummy;
		input>>dummy;
		
		std::getline(input, line);//skip Pr
	}
	
	baseline = fabs(fb3 / (-focus3) - fb2 / (-focus));

	input.close();
};





int main( int /*argc*/, char** /*argv*/ )
{

    //*******************************************
    //Load calibration matrix, camera position, 
    //and image pairs. Init variables.
    //*******************************************
    std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_cam_to_cam.txt" ;
    
    //!< load camera calibration info
	//need to learn about the format of KITTI
	float baseline;
	float focus;
	float u0;
	float v0;
	cv::Mat R_cam0grayTOcam2color(3, 3, CV_32F);
	cv::Mat R_cam2TOrectcam2(3, 3, CV_32F);
	cv::Mat T_cam0grayTOcam2color(1, 3, CV_32F);
	loadCalibrationKITTI(infile, baseline, focus, u0, v0, R_cam0grayTOcam2color, T_cam0grayTOcam2color, R_cam2TOrectcam2 );
	std::cout<<" calib_cam_to_cam.txt loaded. "<<endl;
	//std::cout<<" R_cam0grayTOcam2color: "<<endl<<R_cam0grayTOcam2color<<endl;
	//std::cout<<" R_cam2TOrectcam2: "<<endl<<R_cam2TOrectcam2<<endl;
  
    //!< load camera position info
	std::vector<cv::Mat> rotationList;
	std::vector<cv::Mat> translationList;
	std::vector<cv::Mat> transformList;
	
	//!<get other calibration information
	//first imu to velo
	cv::Mat R_imu2velo(3, 3, CV_32F);
	cv::Mat T_imu2velo(1, 3, CV_32F);
	{
		std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_imu_to_velo.txt" ;
		fstream input(infile.c_str(), ios::in);
		if(!input.good()){
			cerr << "Could not read file: " << infile << endl;
			exit(EXIT_FAILURE);
		}
		
		std::string line, matrixName;
		std::getline(input, line);
		
		input >> matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_imu2velo.at<float>(i, j);
			}
		}
		
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_imu2velo.at<float>(0, i);
		}
		
		input.close();
	}
	std::cout<<" calib_imu_to_velo.txt loaded. "<<endl;
	//std::cout<<" R_imu2velo: "<<endl<<R_imu2velo<<endl;
	
	//then velo to camera
	cv::Mat R_velo2cam0(3, 3, CV_32F);
	cv::Mat T_velo2cam0(1, 3, CV_32F);
	{
		std::string infile = "/home/zhao/Project/KITTI/2011_09_26/calib_velo_to_cam.txt" ;
		fstream input(infile.c_str(), ios::in);
		if(!input.good()){
			cerr << "Could not read file: " << infile << endl;
			exit(EXIT_FAILURE);
		}
		
		std::string line, matrixName;
		std::getline(input, line);
		
		input >> matrixName;
		for (int i = 0; i < 3; i ++)
		{
			for (int j = 0; j < 3; j++)
			{
				input >> R_velo2cam0.at<float>(i, j);
			}
		}
		
		input >> matrixName;
		for (int i = 0; i < 3; i++)
		{
			input >> T_velo2cam0.at<float>(0, i);
		}
		
		input.close();
	}
	std::cout<<" calib_velo_to_cam.txt loaded. "<<endl;
	//std::cout<<" R_velo2cam0: "<<endl<<R_velo2cam0<<endl;
	
	//!<get camera position by translate velodyne position
	/*******************************************************************
	lat:   latitude of the oxts-unit (deg)
	lon:   longitude of the oxts-unit (deg)
	alt:   altitude of the oxts-unit (m)
	roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
	pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
	yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
	vn:    velocity towards north (m/s)
	ve:    velocity towards east (m/s)
	vf:    forward velocity, i.e. parallel to earth-surface (m/s)
	vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
	vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
	ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
	ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
	ay:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
	af:    forward acceleration (m/s^2)
	al:    leftward acceleration (m/s^2)
	au:    upward acceleration (m/s^2)
	wx:    angular rate around x (rad/s)
	wy:    angular rate around y (rad/s)
	wz:    angular rate around z (rad/s)
	wf:    angular rate around forward axis (rad/s)
	wl:    angular rate around leftward axis (rad/s)
	wu:    angular rate around upward axis (rad/s)
	pos_accuracy:  velocity accuracy (north/east in m)
	vel_accuracy:  velocity accuracy (north/east in m/s)
	navstat:       navigation status (see navstat_to_string)
	numsats:       number of satellites tracked by primary GPS receiver
	posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
	velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
	orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
	********************************************************************/
	float lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, pos_accuracy, vel_accuracy;
	int navstat, numsats, posmode, velmode, orimode;
	{
		std::string dirname = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0002_sync/oxts/data/";
		struct dirent **namelist;
		int fileNum = scandir(dirname.c_str(), &namelist, NULL, alphasort);
		if(fileNum == -1) {
			err(EXIT_FAILURE, "%s", dirname.c_str());
		}
		//(void) printf ("%d\n", fileNum);
		
		for (int i = 2; i < fileNum; ++i) 
		{
			fstream input(dirname + namelist[i]->d_name, ios::in);
			if(!input.good()){
				cerr << "Could not read file: " << infile << endl;
				exit(EXIT_FAILURE);
			}
			
			input>>lat>>lon>>alt>>roll>>pitch>>yaw>>vn>>ve>>vf>>vl>>vu>>ax>>ay>>az>>af>>al>>au>>wx>>wy>>wz>>wf>>wl>>wu>>pos_accuracy>>vel_accuracy;
			input>>navstat>>numsats>>posmode>>velmode>>orimode;
			
			input.close();
			free(namelist[i]);
			
			//!<compute mercator scale from latitude
			float scale = cos(lat * M_PI / 180.0);
			
			//!< converts lat/lon coordinates to mercator coordinates using mercator scale

			const float er = 6378137; //!< Earth radius [m]
			float mx = scale * er * (lon * M_PI / 180);
			float my = scale * er * log( tan( (90+lat) * M_PI / 360) );
			//!< (mx,my): is the location of IMU/GPS in the mercator map. The world coordinate system is based on the mercator map.
			
			//!< R,T between IMU/GPS coordinate system from i-th frame to the 1st frame
			cv::Mat R_imu2imu0(3, 3, CV_32F);
			cv::Mat T_imu2imu0(1, 3, CV_32F);
			
			T_imu2imu0.at<float>(0, 0) = mx;
			T_imu2imu0.at<float>(0, 1) = my;
			T_imu2imu0.at<float>(0, 2) = alt; //!< hight from the sea level? [m]
			//std::cout<<translation<<endl;
			
			
			//!< rotation matrix (OXTS RT3000 user manual, page 71/92)
			float rx = roll; // roll
			float ry = yaw; // pitch
			float rz = pitch; // heading 
			
			cv::Mat Rx = (cv::Mat_<float>(3,3) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx));// base => nav  (level oxts => rotated oxts)
			cv::Mat Ry = (cv::Mat_<float>(3,3) << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry));// base => nav  (level oxts => rotated oxts)
			cv::Mat Rz = (cv::Mat_<float>(3,3) << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1);// base => nav  (level oxts => rotated oxts)
			R_imu2imu0 = Rz*Ry*Rx;
			//std::cout<<rotation<<endl;
			
			//!< Now let's translate this IMUGPS position to left camera's
			{
				cv::Mat H_imu2velo(4, 4, CV_32F);
				RT2homogeneous(H_imu2velo, R_imu2velo, T_imu2velo);
				
				cv::Mat H_velo2cam0(4, 4, CV_32F);
				RT2homogeneous(H_velo2cam0, R_velo2cam0, T_velo2cam0);
				
				cv::Mat H_cam0grayTOcam2color(4, 4, CV_32F);
				RT2homogeneous(H_velo2cam0, R_velo2cam0, T_velo2cam0);
				
				cv::Mat H_cam2TOrectcam2(4, 4, CV_32F);
				cv::Mat T_cam2TOrectcam2(1, 3, CV_32F, (0,0,0));
				RT2homogeneous(H_cam2TOrectcam2, R_cam2TOrectcam2, T_cam2TOrectcam2);
				
				cv::Mat H_imu2imu0(4, 4, CV_32F);
				RT2homogeneous(H_imu2imu0, R_imu2imu0, T_imu2imu0);
				
				
				//<! now this matrix will trans ith frame's point(scanner frame) into world's frame(first frame's coordinate)
				
				cv::Mat  H_imu0TOrectcam2(4, 4, CV_32F);
				H_imu0TOrectcam2 = H_cam2TOrectcam2 * H_cam0grayTOcam2color * H_velo2cam0 * H_imu2velo * H_imu2imu0;
				//transformList.push_back(H_imu0TOrectcam2);
				
				cv::Mat T_imu0TOrectcam2(1, 3, CV_32F);
				cv::Mat R_imu0TOrectcam2(3, 3, CV_32F);
				
				
				
				homogeneous2RT(H_imu0TOrectcam2, R_imu0TOrectcam2, T_imu0TOrectcam2);
				
				//now Pc_i = H_imu0TOrectcam2 * Pw, so for Pw we need Pw = H_imu0TOrectcam2.inv() * Pc_i
				
				
				rotationList.push_back(R_imu0TOrectcam2);
				translationList.push_back(T_imu0TOrectcam2);

			}
			//translation = translation + T_imu2velo + T_velo2cam0 + T_cam0grayTOcam2color;
			//rotation = R_cam2TOrectcam2 * R_cam0grayTOcam2color * R_velo2cam0 * R_imu2velo * rotation;
			//rotation = rotation * (R_imu2velo);
			
			
			//std::cout<<translation<<endl;
		}
		free(namelist);
	}
	std::cout<<" camera position obtained. "<<endl;
	
	
	//!< load image pairs
	std::vector<cv::Mat> disparityList;
	std::vector<cv::Mat> LRefectedList;
	std::vector<cv::Mat> RRefectedList;
	std::vector<cv::Mat> LNormalizedList;
	
	int fileNum;
	//!<load images obtained by left color camera
	{
		std::string dirname = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/";
		struct dirent **namelist;
		fileNum = scandir(dirname.c_str(), &namelist, NULL, alphasort);
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
	std::cout<<" left color images loaded. "<<endl;
	
	
	//!<load images obtained by right color camera
	{
		std::string dirname = "/home/zhao/Project/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/";
		struct dirent **namelist;
		fileNum = scandir(dirname.c_str(), &namelist, NULL, alphasort);
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
	std::cout<<" right color images loaded. "<<endl;
	
	//!< fix the file num since we don't want . and ..
	fileNum = fileNum -2;
	
	//!< init variables
	Elas::parameters param;
	param.postprocess_only_left = true;
	param.disp_min = 0;
	param.disp_max = (39 + 1) * 16;
	const int32_t dims[3] = {1242, 375, 1242};
	Elas elas( param );
	
	//!<init paramaters
	float Tcov = 0.5;
	float Tdist = 0.5;
	float Tphoto = 0.7;
	float patchSize = 7;
	float pointingError = 0.5;
	float matchingError = 1.0;
	
	
	pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
	pcl::PointCloud<pcl::PointXYZRGB> currentFrameCloud;
	

	//!<camera matrix K for reproject 3D points to image
	cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
	K.at<float>(0,0) = focus;
	K.at<float>(1,1) = focus;
	K.at<float>(2,2) = 1.0;
	K.at<float>(0,2) = u0;
	K.at<float>(1,2) = v0;
	std::cout<<" K is "<<endl<<K<<endl;
	
	
	//!<check m stereo views nearby the key frame
	//!<r is the center view of those stereo view
	const int m = 3; 
	const int r = (m - 1) / 2 + 1;
	
	
    //boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //viewer->setBackgroundColor (0, 0, 0);
    //viewer->addPointCloud<pcl::PointXYZRGB> (globalCloud.makeShared(), "my points");
    //viewer->addCoordinateSystem (1.0, "global");
    //viewer->initCameraParameters ();
	
	std::cout<<" start main loop. "<<endl;
	
    while ( true )
    {
		//keyframe selected by minimum distance of camera motion
		//maybe I can use some simple frame skip to instead it
		static int framecnt = -1;
		framecnt++;
		fileNum = 8;
		if (framecnt >= fileNum)
			break;
		//bool isKeyframe;
		//if ( !isKeyframe )
		//	continue;
		
		//*******************************************
		//Dense stereo matching by ELAS
		//*******************************************
		cv::Mat imageDisparity(375, 1242, CV_32F);
		cv::Mat& leftRefected = LRefectedList.at(framecnt);
		cv::Mat& rightRefected = RRefectedList.at(framecnt);
		cv::Mat grayLeftRefected;
		cv::Mat grayRightRefected;
		cv::cvtColor(leftRefected,grayLeftRefected,CV_BGR2GRAY);
		cv::cvtColor(rightRefected,grayRightRefected,CV_BGR2GRAY);
		//grayLeftRefected.convertTo(grayLeftRefected,CV_8UC1);
		//grayRightRefected.convertTo(grayRightRefected,CV_8UC1);
		elas.process(grayLeftRefected.data, grayRightRefected.data, (float*)imageDisparity.data,(float*)imageDisparity.data, dims);
		
		
		//!<Record disparity for each frame
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
		
		if ((framecnt + 1) % m != 0)
			continue;
		
		currentFrameCloud.clear();

		
		//center of m stereo view is the keyframe
		int keyFrame = framecnt - m + r;
		std::cout<<" start keyframe at "<<keyFrame <<"."<<endl;
		//cv::imshow( "D1",  disparityList[framecnt - m + 1] );
		//cv::imshow( "D2",  disparityList[framecnt - m + 2] );
		//cv::imshow( "D3",  disparityList[framecnt - m + 3] );
		//cv::imshow( "L1",  LNormalizedList.at(framecnt -m + 1) );
		//cv::imshow( "L2",  grayLeftRefected );
		//cv::imshow( "L3",  LNormalizedList.at(framecnt -m + 3) );
		//cv::waitKey(10);
		
		std::cout<<rotationList[keyFrame]<<endl;
		std::cout<<translationList[keyFrame]<<endl;
		
		
		for ( int i = 0; i < leftRefected.rows; i++ )
		{
			for ( int j = 0; j < leftRefected.cols; j++ )
			{
				//*******************************************
				//Geometric check
				//*******************************************
				
				float d = disparityList[keyFrame].at<float>(i, j);
				if (d < 0)
					continue;
				
				//!<compute 3D point 'hi' under camera's coordinate system 
				float z = focus * (baseline / d);
				float x = z * (j - u0) / focus;
				float y = z * (i - v0) / focus;
				cv::Mat hi(1, 3, CV_32F);
				hi.at<float>(0,0) = x;
				hi.at<float>(0,1) = y;
				hi.at<float>(0,2) = z;
				
				
				//!<convert 'hi' to 'Yi', 3D point under world coordinate system
				cv::Mat Yi(1, 3, CV_32F);
				Yi = ( rotationList[keyFrame].inv() * hi.t() + translationList[keyFrame].t() ).t();
				//x = Yi.at<float>(0,0);
				//y = Yi.at<float>(0,1);
				//z = Yi.at<float>(0,2);
				//std::cout<<Yi<<endl;
				
				
				//!<compute covariance Pi as equation
				//Pi = Ji * Si * Ji'
				cv::Mat Si = cv::Mat::zeros(3, 3, CV_32F);
				Si.at<float>(0,0) = pointingError * pointingError;
				Si.at<float>(1,1) = pointingError * pointingError;
				Si.at<float>(2,2) = matchingError * matchingError;
				
				cv::Mat Ji = cv::Mat::zeros(3, 3, CV_32F);
				Ji.at<float>(0,0) = baseline / d;
				Ji.at<float>(0,2) = - (j * baseline / (d*d) );
				Ji.at<float>(1,1) = baseline / d;
				Ji.at<float>(1,2) = - (i * baseline / (d*d) );
				Ji.at<float>(2,2) = - (focus * baseline / (d*d) );
				
				cv::Mat Pi = cv::Mat::zeros(3, 3, CV_32F);
				Pi = Ji * Si * Ji.t();
				float w = Pi.at<float>(0,0) + Pi.at<float>(1,1) + Pi.at<float>(2,2);
				
				//!<check if wi < Tcov, the first check 
				if (w > Tcov)
					continue;
				
				
				//reproject 3D point 'Yi' to m neighborhood frames
				std::vector<float> weightList;
				std::vector<cv::Mat> YiList;
				bool flgReproject = true;
				for (int k = 1; k <= m; k++)
				{
					int currentFrame = framecnt - m + k;
					cv::Mat& rotation = rotationList.at(keyFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& translation = translationList.at(keyFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& disparity = disparityList.at(keyFrame);/////////////////////////////////////////////////////////currentFrame
					
					
					//'Yi' will be project on current frame's 'Ui' pixel
					cv::Mat Ui = cv::Mat::zeros(1, 3, CV_32F);
					Ui = K * (rotation * (Yi - translation).t() );
					Ui = Ui / Ui.at<float>(0,2);
					
					//Now we got the pixel's position, let's check the disparity
					float u = Ui.at<float>(0,0);
					float v = Ui.at<float>(0,1);
					
					
					//check if pixel out side the image
					if (u<0 || u>1242 || v <0 || v> 375 ||isnan(u) ||isnan(v))
					{
						flgReproject = false;
						break;
					}
					
					//get disparity
					float d = disparity.at<float>(v, u);
					
					//check if disparity is valid
					if ( !(d > 0 && d < param.disp_max))
					{
						flgReproject = false;
						break;
					}

					//!<check if disparity is low uncertainty
					//!<Use the same method with wi's check
					//!<since the Si is already set, we need only to refill the Ji
					
					Ji.at<float>(0,0) = baseline / d;
					Ji.at<float>(0,2) = - (j * baseline / (d*d) );
					Ji.at<float>(1,1) = baseline / d;
					Ji.at<float>(1,2) = - (i * baseline / (d*d) );
					Ji.at<float>(2,2) = - (focus * baseline / (d*d) );
					
					Pi = Ji * Si * Ji.t();
					float w = Pi.at<float>(0,0) + Pi.at<float>(1,1) + Pi.at<float>(2,2);//trace of Pi
					
					if (w > Tcov)
					{
						flgReproject = false;
						break;
					}
					weightList.push_back(w);//save point's weight(certainty) under current frame
					
					
					//!<recompute point's 3D coordinate with current frame's disparity
					//!<and save it to
					float z = focus * (baseline / d);
					float x = z * (j - u0) / focus;
					float y = z * (i - v0) / focus;
					cv::Mat Yi(1, 3, CV_32F);
					Yi.at<float>(0,0) = x;
					Yi.at<float>(0,1) = y;
					Yi.at<float>(0,2) = z;
					Yi = ( rotationList[keyFrame].inv() * Yi.t() + translationList[keyFrame].t() ).t();/////////////////////////////////////////////////////////currentFrame
					YiList.push_back(Yi);//save current point's position under current frame
				}
				
				
				if(flgReproject == false)
					continue;
				
				
				//!<check 3D difference between all reconstructed 3D points is within Tdist
				/*if (framecnt > 3 )
				{
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
				}*/
				
				//*******************************************
				//Photometric check
				//*******************************************
				//In the paper this is done with image's all 3 channals
				//Use the grey scale instead
				std::vector<float> NCCSList;
				std::vector<cv::Vec3b> colorList;
				float gp = 0;
				bool flgPhotometric = true;
				for (int k = 1; k <= m; k++)
				{
					int currentFrame = framecnt - m + k;
					cv::Mat curPixel;
					cv::Mat tarPixel;
				
				
					//calculate 2D position in current neighbor frame
					cv::Mat& rotation = rotationList.at(keyFrame);/////////////////////////////////////////////////////////currentFrame
					cv::Mat& translation = translationList.at(keyFrame);/////////////////////////////////////////////////////////currentFrame
					
					cv::Mat Ui(1, 3, CV_32F);
					Ui = K * (rotation * (Yi - translation).t() );
					Ui = Ui / Ui.at<float>(0,2);
					tarPixel = Ui;
					
					//save pixel value in target frame
					float u = tarPixel.at<float>(0, 0);
					float v = tarPixel.at<float>(0, 1);
					cv::Vec3b bgr = LRefectedList[keyFrame].at<cv::Vec3b>(v,u);/////////////////////////////////////////////////////////currentFrame
					colorList.push_back(bgr);
					if (u - patchSize/2 <0 || u + patchSize/2>1242 || v - patchSize/2 <0 || v + patchSize/2> 375 ||isnan(u) ||isnan(v))
					{
						flgPhotometric = false;
						break;
					}
					
					
					//calculate 2D position in key frame
					cv::Mat& _rotation = rotationList.at(keyFrame);
					cv::Mat& _translation = translationList.at(keyFrame);
					
					Ui = K * (_rotation * (Yi - _translation).t() );
					Ui = Ui / Ui.at<float>(0,2);
					curPixel = Ui;
					u = curPixel.at<float>(0, 0);
					v = curPixel.at<float>(0, 1);
					if (u - patchSize/2 <0 || u + patchSize/2>1242 || v - patchSize/2 <0 || v + patchSize/2> 375 ||isnan(u) ||isnan(v))
					{
						flgPhotometric = false;
						break;
					}
				
					float NCCScore = NCC(LNormalizedList.at(keyFrame), LNormalizedList.at(keyFrame), curPixel, tarPixel, patchSize);/////////////////////////////////////////////////////////currentFrame
					gp = gp + NCCScore;
					
				}
				if(flgPhotometric == false)
					continue;
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
		
		std::cout<<"Geometric & Photometric check finished."<<endl;
		std::cout<<currentFrameCloud.size()<<" points got."<<endl;
		
		//*******************************************
		//Remove outliers
		//*******************************************
		if (currentFrameCloud.size() != 0)
		{
			//for (int i = 0; i < currentFrameCloud.size(); i++)
			{
				//something can be done by PCL, lucky
				
				//!<Use radius removal filter
				pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
				// build the filter
				outrem.setInputCloud(currentFrameCloud.makeShared());
				outrem.setRadiusSearch(0.1);
				outrem.setMinNeighborsInRadius (2);
				// apply filter
				outrem.filter (currentFrameCloud);
				
				//!<Use voxel grid to down sample
				pcl::VoxelGrid<pcl::PointXYZRGB> sor;
				sor.setInputCloud (currentFrameCloud.makeShared());
				sor.setLeafSize (0.05f, 0.05f, 0.05f);
				sor.filter (currentFrameCloud);
			}
		}
		std::cout<<"outliers removed."<<endl;
		std::cout<<currentFrameCloud.size()<<" points remain."<<endl;
		
		//put this point to cloud
		for (int i = 0; i < currentFrameCloud.size(); i++)
		{
			globalCloud.push_back(currentFrameCloud.at(i));
		}
		
		
		
		//!< save each keyframe's reconstruction result to file
		char filename[512];
		sprintf( filename, "frame-%d.pcd", framecnt );
		pcl::io::savePCDFileASCII (filename, currentFrameCloud);
		//std::cout<<"PCD outputed."<<endl;
		
		std::cout<<globalCloud.size()<<" points total."<<endl;
	}
	
	
	//*******************************************
	//Voxel grid filtering
	//*******************************************
	
	//!<Use voxel grid to down sample
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZRGB> vox;
	vox.setInputCloud (globalCloud.makeShared());
	vox.setLeafSize (0.05f, 0.05f, 0.05f);
	vox.filter (globalCloud);
	
	std::cout<<"down sample finish."<<endl;
	
	pcl::io::savePCDFileASCII ("scene.pcd", globalCloud);
	std::cout<<"PCD outputed."<<endl;
	
    return 0;
}


