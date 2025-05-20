#ifndef ESEKFOM_EKF_HPP1
#define ESEKFOM_EKF_HPP1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "use-ikfom.hpp"
#include <ikd-Tree/ikd_Tree.h>

// This header file mainly contains: generalized addition and subtraction, forward propagation main function,
// calculation of feature point residuals and their Jacobians, ESKF main function

const double epsi = 0.001; // When iterating ESKF, if dx < epsi, consider it converged

namespace esekfom
{
	using namespace Eigen;

	PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));		  // Plane parameters corresponding to feature points in the map (plane unit normal vector and current point to plane distance)
	PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // Valid feature points
	PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // Normal vectors corresponding to valid feature points
	bool point_selected_surf[100000] = {1};							  // Determine whether it is a valid feature point

	struct dyn_share_datastruct
	{
		bool valid;												   // Whether the number of valid feature points meets the requirements
		bool converge;											   // Whether it has converged during iteration
		Eigen::Matrix<double, Eigen::Dynamic, 1> h;				   // Residual (z in equation (14))
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // Jacobian matrix H (H in equation (14))
	};

	class esekf
	{
	public:
		typedef Matrix<double, 24, 24> cov;				// 24x24 covariance matrix
		typedef Matrix<double, 24, 1> vectorized_state; // 24x1 vector

		esekf(){};
		~esekf(){};

		state_ikfom get_x()
		{
			return x_;
		}

		cov get_P()
		{
			return P_;
		}

		void change_x(state_ikfom &input_state)
		{
			x_ = input_state;
		}

		void change_P(cov &input_cov)
		{
			P_ = input_cov;
		}

		// Generalized addition (equation (4))
		state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_)
		{
			state_ikfom x_r;
			x_r.pos = x.pos + f_.block<3, 1>(0, 0);

			x_r.rot = x.rot * Sophus::SO3d::exp(f_.block<3, 1>(3, 0));
			x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3d::exp(f_.block<3, 1>(6, 0));

			x_r.offset_T_L_I = x.offset_T_L_I + f_.block<3, 1>(9, 0);
			x_r.vel = x.vel + f_.block<3, 1>(12, 0);
			x_r.bg = x.bg + f_.block<3, 1>(15, 0);
			x_r.ba = x.ba + f_.block<3, 1>(18, 0);
			x_r.grav = x.grav + f_.block<3, 1>(21, 0);

			return x_r;
		}

		// Forward propagation (equations (4-8))
		void predict(double &dt, Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in)
		{
			Eigen::Matrix<double, 24, 1> f_ = get_f(x_, i_in);	  // f in equation (3)
			Eigen::Matrix<double, 24, 24> f_x_ = df_dx(x_, i_in); // df/dx in equation (7)
			Eigen::Matrix<double, 24, 12> f_w_ = df_dw(x_, i_in); // df/dw in equation (7)

			x_ = boxplus(x_, f_ * dt); // Forward propagation (equation (4))

			f_x_ = Matrix<double, 24, 24>::Identity() + f_x_ * dt; // Previously, the Fx matrix did not include the identity matrix and was not multiplied by dt; this corrects it

			P_ = (f_x_)*P_ * (f_x_).transpose() + (dt * f_w_) * Q * (dt * f_w_).transpose(); // Propagate the covariance matrix, i.e., equation (8)
		}

		// Calculate residuals and H matrix for each feature point
		void h_share_model(dyn_share_datastruct &ekfom_data, // ekfom_data: data structure for storing residuals and Jacobians
						   PointCloudXYZI::Ptr &feats_down_body,// ikdtree : global map			
						   KD_TREE<PointType> &ikdtree, // feats_down_body : feature points in the current frame
						   vector<PointVector> &Nearest_Points, // Nearest_Points : nearest points in the global map
						   bool extrinsic_est)
		{
			int feats_down_size = feats_down_body->points.size();
			laserCloudOri->clear();
			corr_normvect->clear();

#ifdef MP_EN
			omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif

			for (int i = 0; i < feats_down_size; i++) // Iterate over all feature points
			{
				PointType &point_body = feats_down_body->points[i];
				PointType point_world;

				V3D p_body(point_body.x, point_body.y, point_body.z);
				// Convert points from Lidar coordinate system to IMU coordinate system, and then to the world coordinate system based on the estimated pose x from forward propagation
				V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
				point_world.x = p_global(0);
				point_world.y = p_global(1);
				point_world.z = p_global(2);
				point_world.intensity = point_body.intensity;

				vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
				auto &points_near = Nearest_Points[i]; // Nearest_Points[i] is printed and found to be a vector sorted by distance from point_world

				double ta = omp_get_wtime();
				if (ekfom_data.converge)
				{
					// Find the nearest neighbor plane points of point_world
					ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
					// Determine whether it is a valid matching point, similar to the loam series, 
					// requiring the number of map points nearest to the feature point > threshold, distance < threshold. Set to true if conditions are met
					point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
																																		: true;
				}
				if (!point_selected_surf[i])
					continue; // If the point does not meet the conditions, skip the following steps

				Matrix<float, 4, 1> pabcd;		// Plane point information
				point_selected_surf[i] = false; // Set the point to invalid to check if it meets the conditions
				// Fit the plane equation ax + by + cz + d = 0 and calculate the point-to-plane distance
				if (esti_plane(pabcd, points_near, 0.1f))
				{
					float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); // Current point-to-plane distance
					float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());												   // If the residual is greater than the empirical threshold, 
																														   //    consider the point valid. 
																														   // In short, the closer the lidar point to the origin, 
																														   // the stricter the requirement for the point-to-plane distance

					if (s > 0.9) // If the residual is greater than the threshold, consider the point valid
					{
						point_selected_surf[i] = true;
						normvec->points[i].x = pabcd(0); // Store the unit normal vector of the plane and the current point-to-plane distance
						normvec->points[i].y = pabcd(1);
						normvec->points[i].z = pabcd(2);
						normvec->points[i].intensity = pd2;
					}
				}
			}

			int effct_feat_num = 0; // Number of valid feature points
			for (int i = 0; i < feats_down_size; i++)
			{
				if (point_selected_surf[i]) // For points that meet the requirements
				{
					laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // Store these points into laserCloudOri
					corr_normvect->points[effct_feat_num] = normvec->points[i];			// Store the normal vectors and point-to-plane distances corresponding to these points
					effct_feat_num++;
				}
			}

			if (effct_feat_num < 1)
			{
				ekfom_data.valid = false;
				ROS_WARN("No Effective Points! \n");
				return;
			}

			// Calculate the Jacobian matrix H and the residual vector
			ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
			ekfom_data.h.resize(effct_feat_num);

			for (int i = 0; i < effct_feat_num; i++)
			{
				V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
				M3D point_crossmat;
				point_crossmat << SKEW_SYM_MATRX(point_);
				V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
				M3D point_I_crossmat;
				point_I_crossmat << SKEW_SYM_MATRX(point_I_);

				// Get the corresponding plane normal vector
				const PointType &norm_p = corr_normvect->points[i];
				V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

				// Calculate the Jacobian matrix H
				V3D C(x_.rot.matrix().transpose() * norm_vec);
				V3D A(point_I_crossmat * C);
				if (extrinsic_est)
				{
					V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
				}
				else
				{
					ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
				}

				// Residual: point-to-plane distance
				// member variable intensity holds point-to-plane distance
				ekfom_data.h(i) = -norm_p.intensity;
			}
		}

		// Generalized subtraction
		vectorized_state boxminus(state_ikfom x1, state_ikfom x2)
		{
			vectorized_state x_r = vectorized_state::Zero();

			x_r.block<3, 1>(0, 0) = x1.pos - x2.pos;

			x_r.block<3, 1>(3, 0) = Sophus::SO3d(x2.rot.matrix().transpose() * x1.rot.matrix()).log();
			x_r.block<3, 1>(6, 0) = Sophus::SO3d(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();

			x_r.block<3, 1>(9, 0) = x1.offset_T_L_I - x2.offset_T_L_I;
			x_r.block<3, 1>(12, 0) = x1.vel - x2.vel;
			x_r.block<3, 1>(15, 0) = x1.bg - x2.bg;
			x_r.block<3, 1>(18, 0) = x1.ba - x2.ba;
			x_r.block<3, 1>(21, 0) = x1.grav - x2.grav;

			return x_r;
		}

		// IESKF
		// Note that the ESKF is not used here, but the IESKF is used instead
		void update_iterated_dyn_share_modified(double R, PointCloudXYZI::Ptr &feats_down_body,
												KD_TREE<PointType> &ikdtree, vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est)
		{
			normvec->resize(int(feats_down_body->points.size()));

			dyn_share_datastruct dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;
			state_ikfom x_propagated = x_; // Here, x_ and P_ are the state variables and covariance matrix after forward propagation, 
										   // because the predict function is called first before this function
			cov P_propagated = P_;

			vectorized_state dx_new = vectorized_state::Zero(); // 24x1 vector

			// Jacobians(J, H) are wrt to ERROR state!
			for (int i = -1; i < maximum_iter; i++) // maximum_iter is the maximum number of iterations for Kalman filtering
			{
				dyn_share.valid = true;
				// Calculate the Jacobian H, which is the derivative of the point-to-plane residual H (called h_x in the code)
				// reisduals and Jacobians are stored in the dyn_share data structure
				h_share_model(dyn_share, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

				if (!dyn_share.valid)
				{
					continue;
				}

				vectorized_state dx;
				dx_new = boxminus(x_, x_propagated); // x^k - x^ in equation (18)

				// Since the H matrix is sparse, with only the first 12 columns having non-zero elements, and the last 12 columns being zero,
				// block matrix form is used to reduce computation
				auto H = dyn_share.h_x;												// m x 12 matrix
				Eigen::Matrix<double, 24, 24> HTH = Matrix<double, 24, 24>::Zero(); // Matrix H^T * H
				HTH.block<12, 12>(0, 0) = H.transpose() * H;

				auto K_front = (HTH / R + P_.inverse()).inverse();
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
				K = K_front.block<24, 12>(0, 0) * H.transpose() / R; // Kalman gain, where R is treated as a constant

				Eigen::Matrix<double, 24, 24> KH = Matrix<double, 24, 24>::Zero(); // Matrix K * H
				KH.block<24, 12>(0, 0) = K * H;
				Matrix<double, 24, 1> dx_ = K * dyn_share.h + (KH - Matrix<double, 24, 24>::Identity()) * dx_new; // Equation (18)
				// std::cout << "dx_: " << dx_.transpose() << std::endl;
				x_ = boxplus(x_, dx_); // Equation (18)

				dyn_share.converge = true;
				for (int j = 0; j < 24; j++)
				{
					if (std::fabs(dx_[j]) > epsi) // If dx > epsi, consider it not converged
					{
						dyn_share.converge = false;
						break;
					}
				}

				if (dyn_share.converge)
					t++;

				if (!t && i == maximum_iter - 2) // If it has not converged after 3 iterations, force it to true, and the h_share_model function will re-search for nearest points
				{
					dyn_share.converge = true;
				}

				if (t > 1 || i == maximum_iter - 1)
				{
					P_ = (Matrix<double, 24, 24>::Identity() - KH) * P_; // Equation (19)
					return;
				}
			}
		}

	private:
		state_ikfom x_;
		cov P_ = cov::Identity();
	};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP1
