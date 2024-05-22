#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

#include "use-ikfom.hpp"
#include "esekfom.hpp"

/*
This header file mainly contains:
IMU data preprocessing: IMU initialization, IMU forward propagation, backward propagation to compensate for motion distortion
*/

#define MAX_INI_COUNT (10)  // Maximum iteration count
// Determine the chronological order of points (note that the curvature stores the timestamp)
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
  Eigen::Matrix<double, 12, 12> Q;    // Noise covariance matrix corresponding to Q in the paper's equation (8)
  void Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &pcl_un_);

  V3D cov_acc;             // Acceleration covariance
  V3D cov_gyr;             // Angular velocity covariance
  V3D cov_acc_scale;       // Initial acceleration covariance from external input
  V3D cov_gyr_scale;       // Initial angular velocity covariance from external input
  V3D cov_bias_gyr;        // Angular velocity bias covariance
  V3D cov_bias_acc;        // Acceleration bias covariance
  double first_lidar_time; // Time of the first point cloud in the current frame

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;        // Current frame point cloud without distortion
  sensor_msgs::ImuConstPtr last_imu_;     // Previous frame IMU
  vector<Pose6D> IMUpose;                 // Store IMU poses (used for backward propagation)
  M3D Lidar_R_wrt_IMU;                    // Rotation extrinsics from lidar to IMU
  V3D Lidar_T_wrt_IMU;                    // Translation extrinsics from lidar to IMU
  V3D mean_acc;                           // Mean acceleration, used to calculate variance
  V3D mean_gyr;                           // Mean angular velocity, used to calculate variance
  V3D angvel_last;                        // Previous frame angular velocity
  V3D acc_s_last;                         // Previous frame acceleration
  double start_timestamp_;                // Start timestamp
  double last_lidar_end_time_;            // End timestamp of the previous frame
  int init_iter_num = 1;                  // Initialization iteration count
  bool b_first_frame_ = true;             // Whether it is the first frame
  bool imu_need_init_ = true;             // Whether IMU needs initialization
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;                          // Initialization iteration count
  Q = process_noise_cov();                    // Call process_noise_cov from use-ikfom.hpp to initialize noise covariance
  cov_acc = V3D(0.1, 0.1, 0.1);               // Initialize acceleration covariance
  cov_gyr = V3D(0.1, 0.1, 0.1);               // Initialize angular velocity covariance
  cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001); // Initialize angular velocity bias covariance
  cov_bias_acc = V3D(0.0001, 0.0001, 0.0001); // Initialize acceleration bias covariance
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;                       // Initialize previous frame angular velocity
  Lidar_T_wrt_IMU = Zero3d;                   // Initialize lidar to IMU translation extrinsics
  Lidar_R_wrt_IMU = Eye3d;                    // Initialize lidar to IMU rotation extrinsics
  last_imu_.reset(new sensor_msgs::Imu());    // Initialize previous frame IMU
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()   // Reset parameters
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;                   // Whether IMU needs initialization
  start_timestamp_ = -1;                   // Start timestamp
  init_iter_num = 1;                       // Initialization iteration count
  IMUpose.clear();                         // Clear IMU poses
  last_imu_.reset(new sensor_msgs::Imu()); // Initialize previous frame IMU
  cur_pcl_un_.reset(new PointCloudXYZI()); // Initialize current frame point cloud without distortion
}

// Pass in external parameters
void ImuProcess::set_param(const V3D &transl, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)  
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
  cov_gyr_scale = gyr;
  cov_acc_scale = acc;
  cov_bias_gyr = gyr_bias;
  cov_bias_acc = acc_bias;
}

// IMU initialization: use the average value of the initial IMU frames to initialize the state x
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N)
{
  // MeasureGroup struct represents all data currently being processed, including the IMU queue and a frame of lidar point cloud, as well as the lidar start and end times
  // Initialize gravity, gyro bias, acc and gyro covariance, normalize the acceleration measurements to unit gravity
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_) // If it is the first frame of IMU
  {
    Reset();    // Reset IMU parameters
    N = 1;      // Set iteration count to 1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    // Initial IMU acceleration
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       // Initial IMU angular velocity
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;              // Set the first frame acceleration value as the initial mean
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;              // Set the first frame angular velocity value as the initial mean
    first_lidar_time = meas.lidar_beg_time;                   // Set the lidar start time corresponding to the current IMU frame as the initial time
  }

  for (const auto &imu : meas.imu)    // Calculate the mean and variance based on all IMU data
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc  += (cur_acc - mean_acc) / N;    // Update the mean based on the difference between the current frame and the mean
    mean_gyr  += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc)  / N;
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr)  / N / N * (N-1);

    N ++;
  }
  
  state_ikfom init_state = kf_state.get_x();        // Get the state x_ from esekfom.hpp
  init_state.grav = - mean_acc / mean_acc.norm() * G_m_s2;    // Obtain the average measured unit direction vector * gravity acceleration preset value
  
  init_state.bg  = mean_gyr;      // Set the angular velocity measurement as the gyro bias
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;      // Pass in the lidar and IMU extrinsics
  init_state.offset_R_L_I = Sophus::SO3d(Lidar_R_wrt_IMU);
  kf_state.change_x(init_state);      // Pass the initialized state to x_ in esekfom.hpp

  Matrix<double, 24, 24> init_P = MatrixXd::Identity(24,24);      // Get the covariance matrix P_ from esekfom.hpp
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = init_P(23,23) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

  // std::cout << "IMU init new -- init_state  " << init_state.pos  <<" " << init_state.bg <<" " << init_state.ba <<" " << init_state.grav << std::endl;
}

// Backward propagation
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI &pcl_out)
{
  /*** Add the last IMU at the end of the previous frame to the beginning of the current frame ***/
  auto v_imu = meas.imu;         // Extract the current frame's IMU queue
  v_imu.push_front(last_imu_);   // Add the last IMU at the end of the previous frame to the beginning of the current frame
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    // Get the time of the IMU at the end of the current frame
  const double &pcl_beg_time = meas.lidar_beg_time;      // Start and end timestamps of the point cloud
  const double &pcl_end_time = meas.lidar_end_time;
  
  // Reorder the point cloud based on the timestamp of each point in the cloud
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  // The curvature stores the timestamp (in preprocess.cpp)

  state_ikfom imu_state = kf_state.get_x();  // Get the posterior state estimated by KF from the last time as the initial state for this IMU prediction
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
  // Add the initial state to IMUpose, including time interval, previous frame acceleration, previous frame angular velocity, previous frame velocity, previous frame position, previous frame rotation matrix

  /*** Forward propagation ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu; // angvel_avr is the average angular velocity, acc_avr is the average acceleration, acc_imu is the IMU acceleration, vel_imu is the IMU velocity, pos_imu is the IMU position
  M3D R_imu;    // IMU rotation matrix used to eliminate motion distortion

  double dt = 0;

  input_ikfom in;
  // Traverse all IMU measurements estimated this time and integrate, discrete midpoint method forward propagation
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);        // Get the current frame's IMU data
    auto &&tail = *(it_imu + 1);    // Get the next frame's IMU data
    // Determine the chronological order: whether the timestamp of the next frame is earlier than the end timestamp of the previous frame; if not, continue directly
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),      // Midpoint integration
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr  = acc_avr * G_m_s2 / mean_acc.norm(); // Adjust the acceleration by the gravity value (divided by the initialized IMU magnitude * 9.8)

    // If the IMU start time is earlier than the last lidar end time (because the last IMU of the previous frame is inserted at the beginning of this frame, this will occur once)
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; // Propagate from the end time of the last lidar to the end of this IMU, calculating the time difference
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();     // Time interval between two IMU times
    }
    
    in.acc = acc_avr;     // The midpoint of the two frames of IMU is used as input in for forward propagation
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;         // Configure the covariance matrix
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;

    kf_state.predict(dt, Q, in);    // IMU forward propagation, each propagation time interval is dt

    imu_state = kf_state.get_x();   // Update the IMU state to the integrated state
    // Update the previous frame angular velocity = next frame angular velocity - bias  
    angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
    // Update the previous frame acceleration in the world coordinate system = R*(acceleration - bias) - g
    acc_s_last  = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();   

    // std::cout << "acc_s_last: " << acc_s_last.transpose() << std::endl;
    // std::cout << "imu_state.ba: " << imu_state.ba.transpose() << std::endl;
    // std::cout << "imu_state.grav: " << imu_state.grav.transpose() << std::endl;
    acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
    // std::cout << "--acc_s_last: " << acc_s_last.transpose() << std::endl<< std::endl;

    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;    // Time interval between the next IMU time and the start of this lidar
    IMUpose.push_back( set_pose6d( offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix() ) );
  }

  // Add the last IMU measurement
  dt = abs(pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  imu_state = kf_state.get_x();   
  last_imu_ = meas.imu.back();              // Save the last IMU measurement for the next frame
  last_lidar_end_time_ = pcl_end_time;      // Save the end time of the last lidar measurement for the next frame

   /*** Eliminate distortion for each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;

  // Traverse each IMU frame
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);   // Get the IMU rotation matrix of the previous frame
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);     // Get the IMU velocity of the previous frame
    pos_imu<<VEC_FROM_ARRAY(head->pos);     // Get the IMU position of the previous frame
    acc_imu<<VEC_FROM_ARRAY(tail->acc);     // Get the IMU acceleration of the next frame
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  // Get the IMU angular velocity of the next frame

    // The point cloud was previously sorted in ascending order by time, and IMUpose was also pushed in ascending order by time
    // Starting the loop from the end of IMUpose means starting from the maximum time, so we only need to check if the point cloud time > IMU head time, and not if the point cloud time < IMU tail
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;    // Time interval between the point and the start time of IMU

      /*    P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    */

      M3D R_i(R_imu * Sophus::SO3d::exp(angvel_avr * dt).matrix() );   // Rotation at the point it_pcl's time: IMU rotation matrix of the previous frame * exp(next frame angular velocity * dt)   
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);   // Position at the point's time (in lidar coordinate system)
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);   // From the point's position in the world to the end position of the lidar
      V3D P_compensate = imu_state.offset_R_L_I.matrix().transpose() * (imu_state.rot.matrix().transpose() * (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}


double T1,T2;
void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloudXYZI::Ptr &cur_pcl_un_)
{
  // T1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)   
  {
    // The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);  // If the initial few frames require IMU parameter initialization

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_); 

  // T2 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<T2 - T1<<endl;
}
