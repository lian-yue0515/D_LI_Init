#ifndef DYNAMIC_INIT_H
#define DYNAMIC_INIT_H

#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <csignal>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <condition_variable>
#include <common_lib.h>
#include <algorithm>
#include <ceres/ceres.h>
#include <sys/time.h>
#include "matplotlibcpp.h"
#include <pcl/io/pcd_io.h>
#include "../integration_base.hpp"
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include "../integration_base.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl_conversions/pcl_conversions.h>


#define FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef pcl::PointCloud<pcl::FPFHSignature33> FPFHFeature;

struct CalibState {
    M3D R;
    V3D T;
    IntegrationBase *pre_integration;
    double timeStamp;

    CalibState(M3D R_, V3D T_, double t_) : R(R_), T(T_), timeStamp(t_){
    };
};


class Dynamic_init {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ofstream fout_LiDAR_meas, fout_IMU_meas;
    int data_accum_length;
    int lidar_frame_count;
    bool first_point, second_point;
    double mean_acc_norm = 9.8;
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> Undistortpoint;
    vector<GYR_> GYR_first;
    vector<GYR_> GYR_pose;
    vector<MeasureGroup> Initialized_data;
    IntegrationBase *tmp_pre_integration;
    V3D acc_0;
    V3D gyr_0;
    bool dynamic_init_fished = false;
    bool Data_processing_fished = false;
    deque<CalibState> system_state;


    Dynamic_init();

    ~Dynamic_init();


    void solve_Rot_bias_gyro();
    void LinearAlignment(StatesGroup icp_state, VectorXd &x);
    void LinearAlignment_withoutba(StatesGroup icp_state, VectorXd &x);
    void RefineGravity(StatesGroup icp_state, Vector3d &g, VectorXd &x);
    void RefineGravity_withoutba(StatesGroup icp_state, Vector3d &g, VectorXd &x);
    void Dynamic_Initialization(int &orig_odom_freq, int &cut_frame_num, double &timediff_imu_wrt_lidar,
                            const double &move_start_time);
    bool Data_processing(MeasureGroup& meas);
    bool Data_processing_lo(M3D rot, V3D t, double time, IntegrationBase *pre_integration);
    void Data_propagate();
    void clear();
    void print_initialization_result(V3D &bias_g, V3D &bias_a, V3D gravity, V3D V_0);

    inline V3D get_Grav_L0() {
        return Grav_L0;
    }
    inline V3D get_gyro_bias() {
        return gyro_bias;
    }
    inline V3D get_acc_bias() {
        return acc_bias;
    }
    inline V3D get_V_0() {
    return V_0;
    }
    inline double get_g() {
    return g_;
    }

private:
    /// Parameters needed to be calibrateds
    V3D Grav_L0;                  // Gravity vector in the initial Lidar frame L_0
    V3D gyro_bias;                // gyro bias
    V3D acc_bias;                 // acc bias
    V3D V_0;                           // initial velocity
    double g_;
};

#endif
