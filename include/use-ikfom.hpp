#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "common_lib.h"
#include "sophus/so3.hpp"

// This header file mainly contains: definitions of state variable x and input variable u, 
// as well as functions for related matrices in forward propagation

// 24-dimensional state variable x
struct state_ikfom
{
    Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);
    Sophus::SO3d rot = Sophus::SO3d(Eigen::Matrix3d::Identity());
    Sophus::SO3d offset_R_L_I = Sophus::SO3d(Eigen::Matrix3d::Identity());
    Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d bg = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d grav = Eigen::Vector3d(0, 0, -G_m_s2);
};

// Input u
struct input_ikfom
{
    Eigen::Vector3d acc = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d gyro = Eigen::Vector3d(0, 0, 0);
};

// Initialization of noise covariance Q (corresponding to Q in equation (8), used in IMU_Processing.hpp)
Eigen::Matrix<double, 12, 12> process_noise_cov()
{
    Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
    Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

    return Q;
}

// Corresponding to f in equation (2) and (3)
// Derivative of state variable x with respect to time
Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in)
{
    // Corresponding order: velocity (3), angular velocity (3), extrinsic T (3), extrinsic rotation R (3), 
    // acceleration (3), gyroscope bias (3), accelerometer bias (3), position (3). Not consistent with the order in the paper.
    Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
    Eigen::Vector3d omega = in.gyro - s.bg; // Input IMU angular velocity (i.e., actual measured value) - estimated bias (corresponding to the first row of the equation)
    Eigen::Vector3d a_inertial = s.rot.matrix() * (in.acc - s.ba); // Input IMU acceleration, first convert to world coordinate system (corresponding to the third row of the equation)
    // s.rot G_R_imu[i]
    // G_R_imu[i] * (acc[i] - ba[i])

    for (int i = 0; i < 3; i++)
    {
        res(i) = s.vel[i]; // Velocity (corresponding to the second row of the equation) => G_v_imu[i]
        res(i + 3) = omega[i]; // Angular velocity (corresponding to the first row of the equation) => angvel_imu[i] - bg_imu[i]
        res(i + 12) = a_inertial[i] + s.grav[i]; // Acceleration (corresponding to the third row of the equation) => G_R_imu[i] * (acc_imu[i] - ba_imu[i]) + G_g[i]
    }

    return res;
}

// Corresponding to Fx in equation (7), note that this matrix is not multiplied by dt, and does not include the identity matrix
Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in)
{
    Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
    cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity(); // Corresponding to the second row, third column of equation (7) - I
    Eigen::Vector3d acc_ = in.acc - s.ba; // Measured acceleration = a_m - bias

    cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3d::hat(acc_); // Corresponding to the third row, first column of equation (7)
    cov.block<3, 3>(12, 18) = -s.rot.matrix(); // Corresponding to the third row, fifth column of equation (7)

    cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // Corresponding to the third row, sixth column of equation (7) - I
    cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); // Corresponding to the first row, fourth column of equation (7) (simplified to -I)
    return cov;
}

// Corresponding to Fw in equation (7), note that this matrix is not multiplied by dt
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s, input_ikfom in)
{
    Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
    cov.block<3, 3>(12, 3) = -s.rot.matrix(); // Corresponding to the third row, second column of equation (7) -R
    cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity(); // Corresponding to the first row, first column of equation (7) -A(w dt) simplified to -I
    cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); // Corresponding to the fourth row, third column of equation (7) - I
    cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); // Corresponding to the fifth row, fourth column of equation (7) - I
    return cov;
}

#endif
