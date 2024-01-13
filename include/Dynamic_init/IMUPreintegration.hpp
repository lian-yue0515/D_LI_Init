#ifndef IMUPREINTEGRATION_HPP
#define IMUPREINTEGRATION_HPP

#include <chrono>
#include "color.h"
#include "cstring"
#include "common_lib.h"
#include "sophus/so3.hpp"

using SO3 = Sophus::SO3d;  
typedef sensor_msgs::Imu::ConstPtr IMU; 
using namespace std;
class IMUPreintegration {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Options {
        Options() {}
        V3D init_bg_ = V3D::Zero();  
        V3D init_ba_ = V3D::Zero(); 
        double noise_gyro_ = 1e-2;       
        double noise_acce_ = 1e-1;      
    };

    IMUPreintegration(Options options = Options()){
        bg_ = options.init_bg_;
        ba_ = options.init_ba_;
        const float ng2 = options.noise_gyro_ * options.noise_gyro_;
        const float na2 = options.noise_acce_ * options.noise_acce_;
        noise_gyro_acce_.diagonal() << ng2, ng2, ng2, na2, na2, na2;
    }

    void Integrate(const IMU &imu, double dt){

        V3D gyr;
        gyr << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
        gyr = gyr - bg_;  
        V3D acc;
        acc << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
        acc = acc - ba_;  

        dp_ = dp_ + dv_ * dt + 0.5f * dR_.matrix() * acc * dt * dt;
        dv_ = dv_ + dR_ * acc * dt;


        Eigen::Matrix<double, 9, 9> A;
        A.setIdentity();
        Eigen::Matrix<double, 9, 6> B;
        B.setZero();

        M3D acc_hat = SO3::hat(acc);
        double dt2 = dt * dt;

        A.block<3, 3>(3, 0) = -dR_.matrix() * dt * acc_hat;
        A.block<3, 3>(6, 0) = -0.5f * dR_.matrix() * acc_hat * dt2;
        A.block<3, 3>(6, 3) = dt * M3D::Identity();

        B.block<3, 3>(3, 3) = dR_.matrix() * dt;
        B.block<3, 3>(6, 3) = 0.5f * dR_.matrix() * dt2;

        dP_dba_ = dP_dba_ + dV_dba_ * dt - 0.5f * dR_.matrix() * dt2;                      
        dP_dbg_ = dP_dbg_ + dV_dbg_ * dt - 0.5f * dR_.matrix() * dt2 * acc_hat * dR_dbg_;  
        dV_dba_ = dV_dba_ - dR_.matrix() * dt;                                             
        dV_dbg_ = dV_dbg_ - dR_.matrix() * dt * acc_hat * dR_dbg_;                       

        V3D omega = gyr * dt;         
        M3D rightJ = SO3::jr(omega);  
        SO3 deltaR = SO3::exp(omega);   
        dR_ = dR_ * deltaR;            

        A.block<3, 3>(0, 0) = deltaR.matrix().transpose();
        B.block<3, 3>(0, 0) = rightJ * dt;


        cov_ = A * cov_ * A.transpose() + B * noise_gyro_acce_ * B.transpose();


        dR_dbg_ = deltaR.matrix().transpose() * dR_dbg_ - rightJ * dt;  // (4.39a)

        dt_ += dt;
    }

    SO3 GetDeltaRotation(const V3D &bg){ return dR_ * SO3::exp(dR_dbg_ * (bg - bg_)); }
    V3D GetDeltaVelocity(const V3D &bg, const V3D &ba){ return dv_ + dV_dbg_ * (bg - bg_) + dV_dba_ * (ba - ba_); }
    V3D GetDeltaPosition(const V3D &bg, const V3D &ba){ return dp_ + dP_dbg_ * (bg - bg_) + dP_dba_ * (ba - ba_); }

   public:
    double dt_ = 0;                        
    M9D cov_ = M9D::Zero();              
    M6D noise_gyro_acce_ = M6D::Zero();  

    V3D bg_ = V3D::Zero();
    V3D ba_ = V3D::Zero();

    SO3 dR_;
    V3D dv_ = V3D::Zero();
    V3D dp_ = V3D::Zero();

    M3D dR_dbg_ = M3D::Zero();
    M3D dV_dbg_ = M3D::Zero();
    M3D dV_dba_ = M3D::Zero();
    M3D dP_dbg_ = M3D::Zero();
    M3D dP_dba_ = M3D::Zero();
};


#endif 
