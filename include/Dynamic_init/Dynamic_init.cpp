#include "Dynamic_init.h"
#include <pcl/registration/icp.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include "IMUPreintegration.hpp"
double last_lidar_end_time_;   
V3D angvel_last;
sensor_msgs::ImuConstPtr last_imu_;
const bool time_(PointType &x, PointType &y) {return (x.curvature < y.curvature);};
Pose pose_cur{0,0,0,0,0,0};
Pose pose_cur_no{0,0,0,0,0,0};
Pose doICP(pcl::PointCloud<pcl::PointXYZINormal> cureKeyframeCloud, pcl::PointCloud<pcl::PointXYZINormal> targetKeyframeCloud)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr sourcePtr = cureKeyframeCloud.makeShared();
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr targetPtr = targetKeyframeCloud.makeShared();
    pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
    icp.setMaxCorrespondenceDistance(50);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    // Align pointclouds 
    icp.setInputSource(sourcePtr);
    icp.setInputTarget(targetPtr);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZINormal>());
    icp.align(*unused_result);
    float loopFitnessScoreThreshold = 0.8;
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold)
    {
        std::cout << "ICP odometry failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << std::endl;
    }
    else
    {
        std::cout << "ICP odometry passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << std::endl;
    }
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    return Pose{x, y, z, roll, pitch, yaw};

} 

Dynamic_init::Dynamic_init(){
    fout_LiDAR_meas.open(FILE_DIR("LiDAR_meas.txt"), ios::out);
    fout_IMU_meas.open(FILE_DIR("IMU_meas.txt"), ios::out);
    data_accum_length = 5;
    lidar_frame_count = 0;
    gyro_bias = Zero3d;
    acc_bias = Zero3d;
    Grav_L0 = Zero3d;
    V_0 = Zero3d;
    first_point = true;
    second_point = true;
}

Dynamic_init::~Dynamic_init() = default;

void Dynamic_init::set_IMU_state(const deque<CalibState> &IMU_states){
    IMU_state_group.assign(IMU_states.begin(), IMU_states.end() - 1);
}

void Dynamic_init::set_Lidar_state(const deque<CalibState> &Lidar_states) {
    Lidar_state_group.assign(Lidar_states.begin(), Lidar_states.end() - 1);
}

void Dynamic_init::push_ALL_IMU_CalibState(const sensor_msgs::Imu::ConstPtr &msg, const double &mean_acc_norm) {
    CalibState IMUstate;
    IMUstate.ang_vel = V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    IMUstate.linear_acc =
            V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) / mean_acc_norm *
            G_m_s2;
    IMUstate.timeStamp = msg->header.stamp.toSec();
    IMU_state_group_ALL.push_back(IMUstate);
}


void Dynamic_init::push_Lidar_CalibState(const M3D &rot, const V3D &omg, const V3D &linear_vel, const double &timestamp) {
    CalibState Lidarstate;
    Lidarstate.rot_end = rot;
    Lidarstate.ang_vel = omg;
    Lidarstate.linear_vel = linear_vel;
    Lidarstate.timeStamp = timestamp;
    Lidar_state_group.push_back(Lidarstate);
}

void Dynamic_init::clear() {
    // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
    cout << "\x1B[2J\x1B[H";
}

bool Dynamic_init::Data_processing(MeasureGroup& meas, StatesGroup icp_state)//, state_ikfom state
{
    Initialized_data.push_back(meas);
    if(first_point){
        first_point = false;
        second_point = true;
        lidar_frame_count++;
        last_imu_ = Initialized_data[0].imu.back();
        return 0;
        odom.push_back(pose_cur);
    }
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    auto pcl_current = *(meas.lidar);
    double pcl_beg_time, pcl_end_time;
    pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time ***/
    sort(pcl_current.points.begin(), pcl_current.points.end(), time_);
    pcl_end_time = pcl_beg_time + pcl_current.points.back().curvature / double(1000);
    double imu_end_time = v_imu.back()->header.stamp.toSec();

    const double &pcl_end_offset_time = pcl_current.points.back().curvature / double(1000);

    /*** speed calculation ***/
    auto pcl_last = *(Initialized_data[Initialized_data.size()-2].lidar);
    Eigen::Vector4f last_cen;					
    pcl::compute3DCentroid(pcl_last, last_cen);	
    Eigen::Vector4f current_cen;
    float timediff = pcl_end_time - pcl_beg_time;			
    pcl::compute3DCentroid(pcl_current, current_cen);	
    V3D displacement = V3D(current_cen[0] - last_cen[0], 
                        current_cen[1] - last_cen[1], 
                    current_cen[2] - last_cen[2]);
    V3D vel_cen = - displacement/timediff;      //Velocity direction opposite to numerical calculation
    cout<<"velocity for "<<lidar_frame_count<<" frame: "<<vel_cen<<endl;
    if(second_point)  //Motion distortion removal for first
    {
        second_point = false;
        auto v_imu_ = Initialized_data[0].imu;
        double dt_ = 0;
        M3D R_imu_(icp_state.rot_end);
        V3D angvel_avr_;
        double imu_end_time_ = v_imu_.back()->header.stamp.toSec();
        double pcl_beg_time_, pcl_end_time_;
        pcl_beg_time_ = Initialized_data[0].lidar_beg_time;
        /*** sort point clouds by offset time ***/
        sort(pcl_last.points.begin(), pcl_last.points.end(), time_);
        pcl_end_time_ = pcl_beg_time_ + pcl_last.points.back().curvature / double(1000);
        const double &pcl_end_offset_time_ = pcl_current.points.back().curvature / double(1000);
        GYR_first.clear();
        GYR_first.push_back(imu_accumulative(0.0, angvel_avr_, R_imu_));
        /*** forward propagation at each imu point ***/
        for (auto it_imu = v_imu_.end(); it_imu < (v_imu_.begin() + 1); it_imu--)
        {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu - 1);
            if (tail->header.stamp.toSec() < last_lidar_end_time_)
                continue;
            angvel_avr_ << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
            if(head->header.stamp.toSec() < last_lidar_end_time_)
                dt_ = tail->header.stamp.toSec() - last_lidar_end_time_;
            else
                dt_ = tail->header.stamp.toSec() - head->header.stamp.toSec();
            R_imu_ = R_imu_*Exp(angvel_avr_, dt_);
            /* save the poses at each IMU gyr */
            angvel_avr_ = angvel_avr_;    //Initial default bias is zero
            double &&offs_t = tail->header.stamp.toSec() - pcl_end_time_;
            GYR_first.push_back(imu_accumulative(offs_t, angvel_avr_, R_imu_));
        }
        last_lidar_end_time_ = pcl_end_time;
        last_imu_ = Initialized_data[0].imu.back();

        //for first:
        /*** undistort each lidar point (backward propagation) ***/
        auto it_pcl = pcl_last.points.end() - 1; //a single point in k-th frame
        for (auto it_kp = GYR_first.begin() + 1 ; it_kp != GYR_first.end(); it_kp++)
        {
            double dt, dt_j;
            auto head = it_kp + 1;
            M3D R; 
            V3D ANGVEL;
            R = (head->rot).inverse();
            ANGVEL << VEC_FROM_ARRAY(head->angvel);
            ANGVEL = -ANGVEL;
            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
                /* Transform to the 'scan-end' IMU frame（I_k frame)*/
                M3D R_i(R * Exp(ANGVEL, dt));
                V3D p_in(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D P_compensate = icp_state.offset_R_L_I.transpose() * (R_imu_.transpose() * (R_i * (icp_state.offset_R_L_I * p_in + icp_state.offset_T_L_I) - icp_state.pos_end) - icp_state.offset_T_L_I);

                dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
                V3D p_jk;
                p_jk = - (icp_state.rot_end).transpose() * vel_cen * dt_j;
                P_compensate = P_compensate + p_jk;
                
                /// save Undistorted points
                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);
                if (it_pcl == pcl_last.points.begin()) break;
            }
        }
        Undistortpoint.push_back(pcl_last.makeShared());

    }

    V3D angvel_avr, vel_imu(icp_state.vel_end), pos_imu(icp_state.pos_end);
    M3D R_imu(icp_state.rot_end);
    /*** Initialize IMU pose ***/
    GYR_pose.clear();
    GYR_pose.push_back(imu_accumulative(0.0, angvel_last, R_imu));
    /*** forward propagation at each imu point ***/
    double dt = 0;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header.stamp.toSec() < last_lidar_end_time_)
            continue;

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        if(head->header.stamp.toSec() < last_lidar_end_time_)
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        else
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        R_imu = R_imu*Exp(angvel_avr, dt);
        angvel_last = angvel_avr;
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        GYR_pose.push_back(imu_accumulative(offs_t, angvel_last, R_imu));
    }
    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    icp_state.vel_end = vel_cen;
    icp_state.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    icp_state.pos_end = pos_imu + vel_cen*(pcl_end_time - pcl_beg_time);

    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;
    double dt_j = 0.0;
    /*** undistort each lidar point (backward propagation) ***/
    //for next:
    auto it_pcl = pcl_current.points.end() - 1; //a single point in k-th frame
    for (auto it_kp = GYR_pose.end() - 1; it_kp != GYR_pose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        R_imu = head->rot;
        angvel_avr << VEC_FROM_ARRAY(head->angvel);
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
            /* Transform to the 'scan-end' IMU frame（I_k frame)*/
            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D p_in(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D P_compensate = icp_state.offset_R_L_I.transpose() * (icp_state.rot_end.transpose() * \
                (R_i * (icp_state.offset_R_L_I * p_in + icp_state.offset_T_L_I) - icp_state.pos_end) - \
                icp_state.offset_T_L_I);
            
            dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
            V3D p_jk;
            p_jk = - icp_state.rot_end.transpose() * icp_state.vel_end * dt_j;
            P_compensate = P_compensate + p_jk;
            
            /// save Undistorted points
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
            if (it_pcl == pcl_current.points.begin()) break;
        }
    }
    Undistortpoint.push_back(pcl_current.makeShared());
    icpodom.push_back(doICP(*Undistortpoint.back(), *Undistortpoint[Undistortpoint.size()-2]));
    pose_cur = pose_cur.addPoses(pose_cur, icpodom.back());
    odom.push_back(pose_cur);

    icpodom_no.push_back(doICP(*Initialized_data.back().lidar, *Initialized_data[Initialized_data.size()-2].lidar));
    pose_cur_no = pose_cur_no.addPoses(pose_cur_no, icpodom_no.back());
    odom_no.push_back(pose_cur_no);
    // icp_state.rot_end = pose_cur.poseto_rotation();
    // icp_state.pos_end = pose_cur.poseto_position();
    if (lidar_frame_count < data_accum_length)
    {
        lidar_frame_count++;
        return false;
    }
    return true; 
}

void Dynamic_init::Data_propagate(){
    //Calculate icp odometer as well as IMU preintegration
    
    return;
    
}

void Dynamic_init::Dynamic_Initialization(int &orig_odom_freq, int &cut_frame_num, double &timediff_imu_wrt_lidar,
                                const double &move_start_time) {

    TimeConsuming time("Batch optimization");


 
    printf(BOLDBLUE"============================================================ \n\n" RESET);
    // print_initialization_result(time_L_I, Rot_Lidar_wrt_IMU, Trans_Lidar_wrt_IMU, gyro_bias, acc_bias, Grav_L0);
    printf(BOLDBLUE"============================================================ \n\n" RESET);
    printf(BOLDCYAN "[Initialization] Lidar IMU initialization done.\n");
    printf("" RESET);
}
/*
void Dynamic_init::solve_Rotation_only() {
    double R_LI_quat[4];
    R_LI_quat[0] = 1;
    R_LI_quat[1] = 0;
    R_LI_quat[2] = 0;
    R_LI_quat[3] = 0;

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_rot;
    problem_rot.AddParameterBlock(R_LI_quat, 4, quatParam);


    for (int i = 0; i < IMU_state_group.size(); i++) {
        M3D Lidar_angvel_skew;
        Lidar_angvel_skew << SKEW_SYM_MATRX(Lidar_state_group[i].ang_vel);
        problem_rot.AddResidualBlock(Angular_Vel_Cost_only_Rot::Create(IMU_state_group[i].ang_vel,
                                                                    Lidar_state_group[i].ang_vel),
                                                                    nullptr,
                                                                    R_LI_quat);

    }
    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_rot, &summary_quat);
    Eigen::Quaterniond q_LI(R_LI_quat[0], R_LI_quat[1], R_LI_quat[2], R_LI_quat[3]);
    Rot_Lidar_wrt_IMU = q_LI.matrix();
}

void Dynamic_init::solve_Rot_bias_gyro(double &timediff_imu_wrt_lidar) {
    Eigen::Quaterniond quat(Rot_Lidar_wrt_IMU);
    double R_LI_quat[4];
    R_LI_quat[0] = quat.w();
    R_LI_quat[1] = quat.x();
    R_LI_quat[2] = quat.y();
    R_LI_quat[3] = quat.z();

    double bias_g[3]; //Initial value of gyro bias
    bias_g[0] = 0;
    bias_g[1] = 0;
    bias_g[2] = 0;

    double time_lag2 = 0; //Second time lag (IMU wtr Lidar)

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_ang_vel;

    problem_ang_vel.AddParameterBlock(R_LI_quat, 4, quatParam);
    problem_ang_vel.AddParameterBlock(bias_g, 3);

    for (int i = 0; i < IMU_state_group.size(); i++) {
        double deltaT = Lidar_state_group[i].timeStamp - IMU_state_group[i].timeStamp;
        problem_ang_vel.AddResidualBlock(Angular_Vel_Cost::Create(IMU_state_group[i].ang_vel,
                                                                IMU_state_group[i].ang_acc,
                                                                Lidar_state_group[i].ang_vel,
                                                                deltaT),
                                                                nullptr,
                                                                R_LI_quat,
                                                                bias_g,
                                                                &time_lag2);
    }


    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_ang_vel, &summary_quat);

    Eigen::Quaterniond q_LI(R_LI_quat[0], R_LI_quat[1], R_LI_quat[2], R_LI_quat[3]);
    Rot_Lidar_wrt_IMU = q_LI.matrix();
    V3D euler_angle = RotMtoEuler(q_LI.matrix());
    gyro_bias = V3D(bias_g[0], bias_g[1], bias_g[2]);

    time_lag_2 = time_lag2;
    time_delay_IMU_wtr_Lidar = time_lag_1 + time_lag_2;
    cout << "Total time delay (IMU wtr Lidar): " << time_delay_IMU_wtr_Lidar + timediff_imu_wrt_lidar << " s" << endl;
    cout << "Using LIO: SUBTRACT this value from IMU timestamp" << endl
         << "           or ADD this value to LiDAR timestamp." << endl <<endl;

    //The second temporal compensation
    IMU_time_compensate(get_lag_time_2(), false);

    for (int i = 0; i < Lidar_state_group.size(); i++) {
        fout_after_rot << setprecision(12) << (Rot_Lidar_wrt_IMU * Lidar_state_group[i].ang_vel + gyro_bias).transpose()
                       << " " << Lidar_state_group[i].timeStamp << endl;
    }

}

void Dynamic_init::solve_trans_biasacc_grav() {
    M3D Rot_Init = Eye3d;
    Rot_Init.diagonal() = V3D(1, 1, 1);
    Eigen::Quaterniond quat(Rot_Init);
    double R_GL0_quat[4];
    R_GL0_quat[0] = quat.w();
    R_GL0_quat[1] = quat.x();
    R_GL0_quat[2] = quat.y();
    R_GL0_quat[3] = quat.z();

    double bias_aL[3]; //Initial value of acc bias
    bias_aL[0] = 0;
    bias_aL[1] = 0;
    bias_aL[2] = 0;

    double Trans_IL[3]; //Initial value of Translation of IL (IMU with respect to Lidar)
    Trans_IL[0] = 0.0;
    Trans_IL[1] = 0.0;
    Trans_IL[2] = 0.0;

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_acc;

    problem_acc.AddParameterBlock(R_GL0_quat, 4, quatParam);
    problem_acc.AddParameterBlock(bias_aL, 3);
    problem_acc.AddParameterBlock(Trans_IL, 3);

    //Jacobian of acc_bias, gravity, Translation
    int Jaco_size = 3 * Lidar_state_group.size();
    MatrixXd Jacobian(Jaco_size, 9);
    Jacobian.setZero();

    //Jacobian of Translation
    MatrixXd Jaco_Trans(Jaco_size, 3);
    Jaco_Trans.setZero();

    for (int i = 0; i < IMU_state_group.size(); i++) {
        problem_acc.AddResidualBlock(Linear_acc_Cost::Create(Lidar_state_group[i],
                                                    Rot_Lidar_wrt_IMU,
                                                    IMU_state_group[i].linear_acc),
                                                    nullptr,
                                                    R_GL0_quat,
                                                    bias_aL,
                                                    Trans_IL);

        Jacobian.block<3, 3>(3 * i, 0) = -Lidar_state_group[i].rot_end;
        Jacobian.block<3, 3>(3 * i, 3) << SKEW_SYM_MATRX(STD_GRAV);
        M3D omg_skew, angacc_skew;
        omg_skew << SKEW_SYM_MATRX(Lidar_state_group[i].ang_vel);
        angacc_skew << SKEW_SYM_MATRX(Lidar_state_group[i].ang_acc);
        M3D Jaco_trans_i = omg_skew * omg_skew + angacc_skew;
        Jaco_Trans.block<3, 3>(3 * i, 0) = Jaco_trans_i;
        Jacobian.block<3, 3>(3 * i, 6) = Jaco_trans_i;
    }

    for (int index = 0; index < 3; ++index) {
        problem_acc.SetParameterUpperBound(bias_aL, index, 0.01);
        problem_acc.SetParameterLowerBound(bias_aL, index, -0.01);
    }

    ceres::Solver::Options options_acc;
    ceres::Solver::Summary summary_acc;
    ceres::Solve(options_acc, &problem_acc, &summary_acc);


    Eigen::Quaterniond q_GL0(R_GL0_quat[0], R_GL0_quat[1], R_GL0_quat[2], R_GL0_quat[3]);
    Rot_Grav_wrt_Init_Lidar = q_GL0.matrix();
    Grav_L0 = Rot_Grav_wrt_Init_Lidar * STD_GRAV;

    V3D bias_a_Lidar(bias_aL[0], bias_aL[1], bias_aL[2]);
    acc_bias = Rot_Lidar_wrt_IMU * bias_a_Lidar;

    V3D Trans_IL_vec(Trans_IL[0], Trans_IL[1], Trans_IL[2]);
    Trans_Lidar_wrt_IMU = -Rot_Lidar_wrt_IMU * Trans_IL_vec;

    for (int i = 0; i < IMU_state_group.size(); i++) {
        V3D acc_I = Lidar_state_group[i].rot_end * Rot_Lidar_wrt_IMU.transpose() * IMU_state_group[i].linear_acc -
                    Lidar_state_group[i].rot_end * bias_a_Lidar;
        V3D acc_L = Lidar_state_group[i].linear_acc +
                    Lidar_state_group[i].rot_end * Jaco_Trans.block<3, 3>(3 * i, 0) * Trans_IL_vec - Grav_L0;
        fout_acc_cost << setprecision(10) << acc_I.transpose() << " " << acc_L.transpose() << " "
                    << IMU_state_group[i].timeStamp << " " << Lidar_state_group[i].timeStamp << endl;
    }

    M3D Hessian_Trans = Jaco_Trans.transpose() * Jaco_Trans;
    EigenSolver<M3D> es_trans(Hessian_Trans);
    M3D EigenValue_mat_trans = es_trans.pseudoEigenvalueMatrix();
    M3D EigenVec_mat_trans = es_trans.pseudoEigenvectors();

}

void Dynamic_init::normalize_acc(deque<CalibState> &signal_in) {
    V3D mean_acc(0, 0, 0);

    for (int i = 1; i < 10; i++) {
        mean_acc += (signal_in[i].linear_acc - mean_acc) / i;
    }

    for (int i = 0; i < signal_in.size(); i++) {
        signal_in[i].linear_acc = signal_in[i].linear_acc / mean_acc.norm() * G_m_s2;
    }
}
*/
void Dynamic_init::print_initialization_result(V3D &bias_g, V3D &bias_a, V3D gravity, V3D V_0){
    cout.setf(ios::fixed);
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Bias of Gyroscope        = " << bias_g.transpose() << " rad/s" << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Bias of Accelerometer    = " << bias_a.transpose() << " m/s^2" << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "Gravity in World Frame   = " << gravity.transpose() << " m/s^2" << endl << endl;
    printf(BOLDCYAN "[Init Result] " RESET);
    cout << "V_0  = " << V_0.transpose() << " m/s" << endl << endl;
}

void Dynamic_init::plot_result() {
    vector<vector<double>> IMU_omg(3), IMU_acc(3), IMU_ang_acc(3), Lidar_omg(3), Lidar_acc(3), Lidar_ang_acc(3);
    for (auto it_IMU_state = IMU_state_group.begin(); it_IMU_state != IMU_state_group.end() - 1; it_IMU_state++) {
        for (int i = 0; i < 3; i++) {
            IMU_omg[i].push_back(it_IMU_state->ang_vel[i]);
            IMU_acc[i].push_back(it_IMU_state->linear_acc[i]);
            IMU_ang_acc[i].push_back(it_IMU_state->ang_acc[i]);
        }
    }
    for (auto it_Lidar_state = Lidar_state_group.begin();
         it_Lidar_state != Lidar_state_group.end() - 1; it_Lidar_state++) {
        for (int i = 0; i < 3; i++) {
            Lidar_omg[i].push_back(it_Lidar_state->ang_vel[i]);
            Lidar_acc[i].push_back(it_Lidar_state->linear_acc[i]);
            Lidar_ang_acc[i].push_back(it_Lidar_state->ang_acc[i]);
        }
    }

    plt::figure(1);
    plt::subplot(2, 3, 1);
    plt::named_plot("IMU omg x", IMU_omg[0]);
    plt::named_plot("IMU omg y", IMU_omg[1]);
    plt::named_plot("IMU omg z", IMU_omg[2]);
    plt::legend();
    plt::grid(true);

    plt::subplot(2, 3, 2);
    plt::named_plot("IMU acc x", IMU_acc[0]);
    plt::named_plot("IMU acc y", IMU_acc[1]);
    plt::named_plot("IMU acc z", IMU_acc[2]);
    plt::legend();
    plt::grid(true);

    plt::subplot(2, 3, 3);
    plt::named_plot("IMU ang acc x", IMU_ang_acc[0]);
    plt::named_plot("IMU ang acc y", IMU_ang_acc[1]);
    plt::named_plot("IMU ang acc z", IMU_ang_acc[2]);
    plt::legend();
    plt::grid(true);

    plt::subplot(2, 3, 4);
    plt::named_plot("Lidar omg x", Lidar_omg[0]);
    plt::named_plot("Lidar omg y", Lidar_omg[1]);
    plt::named_plot("Lidar omg z", Lidar_omg[2]);
    plt::legend();
    plt::grid(true);

    plt::subplot(2, 3, 5);
    plt::named_plot("Lidar acc x", Lidar_acc[0]);
    plt::named_plot("Lidar acc y", Lidar_acc[1]);
    plt::named_plot("Lidar acc z", Lidar_acc[2]);
    plt::legend();
    plt::grid(true);

    plt::subplot(2, 3, 6);
    plt::named_plot("Lidar ang acc x", Lidar_ang_acc[0]);
    plt::named_plot("Lidar ang acc y", Lidar_ang_acc[1]);
    plt::named_plot("Lidar ang acc z", Lidar_ang_acc[2]);
    plt::legend();
    plt::grid(true);

    plt::show();
    plt::pause(0);
    plt::close();
}