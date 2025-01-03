#include "Dynamic_init.h"

double last_lidar_end_time_ = 0;   
V3D angvel_last;
sensor_msgs::ImuConstPtr last_imu_;
const bool time_(PointType &x, PointType &y) {return (x.curvature < y.curvature);};
Pose pose_cur_no{0,0,0,0,0,0};
Pose pose_cur_Paremi{0,0,0,0,0,0};
Pose icp_result;
std::ostringstream oss;

void voxel_filter(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<PointType>::Ptr &cloud_filtered,
                float leaf_size = 1.0f) {
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.setInputCloud(cloud);
    voxel.filter(*cloud_filtered);
}

Pose trans2pose(const M3D rot, const V3D tran) {
    Eigen::Affine3f transformation_matrix = Eigen::Affine3f::Identity();
    transformation_matrix.linear() = rot.cast<float>();
    transformation_matrix.translation() = tran.cast<float>();

    float tx, ty, tz, troll, tpitch, tyaw;
    pcl::getTranslationAndEulerAngles(transformation_matrix, tx, ty, tz, troll, tpitch, tyaw);
    Pose pose;
    pose.x = tx;
    pose.y = ty;
    pose.z = tz;
    pose.roll = troll;
    pose.pitch = tpitch;
    pose.yaw = tyaw;
    return pose;
}
pcl::PointCloud<PointType>::Ptr Pointscloud_trans(const pcl::PointCloud<PointType>::Ptr cloudIn, const Pose &tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

    int numberOfCores = 16;
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}
Pose doNDT(pcl::PointCloud<PointType> cureKeyframeCloud, pcl::PointCloud<PointType> targetKeyframeCloud) {
    pcl::PointCloud<PointType>::Ptr sourcePtr (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr targetPtr (new pcl::PointCloud<PointType>);
    sourcePtr = cureKeyframeCloud.makeShared();
    targetPtr = targetKeyframeCloud.makeShared();
    pcl::NormalDistributionsTransform<PointType, PointType> ndt;
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1);
    ndt.setMaximumIterations(50);

    ndt.setInputSource(sourcePtr);
    ndt.setInputTarget(targetPtr);
    pcl::PointCloud<PointType>::Ptr unused_result (new pcl::PointCloud<PointType>);
    ndt.align (*unused_result, Eigen::Matrix4f::Identity());
    float loopFitnessScoreThreshold = 0.8;
    if (ndt.hasConverged() == false || ndt.getFitnessScore() > loopFitnessScoreThreshold)
    {
        std::cout << "NDT odometry failed (" << ndt.getFitnessScore() << " > " << loopFitnessScoreThreshold << std::endl;
    }
    else
    {
        std::cout << "NDT odometry passed (" << ndt.getFitnessScore() << " < " << loopFitnessScoreThreshold << std::endl;
    }
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = ndt.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    return Pose{x, y, z, roll, pitch, yaw};
}

Pose doICP(pcl::PointCloud<PointType>::Ptr cureKeyframeCloud, pcl::PointCloud<PointType>::Ptr targetKeyframeCloud)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr sourcePtr = cureKeyframeCloud;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr targetPtr = targetKeyframeCloud;
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
    icp.setMaxCorrespondenceDistance(10);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(1);
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

Pose ParemidICP(pcl::PointCloud<PointType>::Ptr cureKeyframeCloud, pcl::PointCloud<PointType>::Ptr targetKeyframeCloud)
{
    pcl::PointCloud<PointType>::Ptr sourcePtr_05 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr targetPtr_05 (new pcl::PointCloud<PointType>);
    voxel_filter(cureKeyframeCloud, sourcePtr_05, 1);
    voxel_filter(targetKeyframeCloud, targetPtr_05, 1);
    Pose pose_05 = doICP(sourcePtr_05, targetPtr_05);


    pcl::PointCloud<PointType>::Ptr sourcePtr_01 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr targetPtr_01 (new pcl::PointCloud<PointType>);
    sourcePtr_01 = Pointscloud_trans(cureKeyframeCloud, pose_05);
    voxel_filter(sourcePtr_01, sourcePtr_01, 0.5);
    voxel_filter(targetKeyframeCloud, targetPtr_01, 0.5);
    Pose pose_01 = pose_05.addPoses(pose_05, doICP(sourcePtr_01, targetPtr_01));

    pcl::PointCloud<PointType>::Ptr sourcePtr_00 (new pcl::PointCloud<PointType>);
    sourcePtr_00 = Pointscloud_trans(cureKeyframeCloud, pose_01);
    pcl::PointCloud<PointType>::Ptr sourcePtr_005 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr targetPtr_005 (new pcl::PointCloud<PointType>);
    voxel_filter(sourcePtr_00, sourcePtr_005, 0.1);
    voxel_filter(targetKeyframeCloud, targetPtr_005, 0.1);
    Pose pose = pose_01.addPoses(pose_01, doICP(sourcePtr_005, targetPtr_005));
    return pose;
}

Dynamic_init::Dynamic_init(){
    fout_LiDAR_meas.open(FILE_DIR("LiDAR_meas.txt"), ios::out);
    fout_IMU_meas.open(FILE_DIR("IMU_meas.txt"), ios::out);
    data_accum_length = 10;
    lidar_frame_count = 0;
    gyro_bias = Zero3d;
    acc_bias = Zero3d;
    Grav_L0 = Zero3d;
    V_0 = Zero3d;
    first_point = true;
    second_point = true;
}

Dynamic_init::~Dynamic_init() = default;


void Dynamic_init::clear() {
    // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
    cout << "\x1B[2J\x1B[H";
}

bool Dynamic_init::Data_processing(MeasureGroup& meas, StatesGroup &icp_state, std::map<double, Pose> &poseMap, bool usetrue)
{
    std::chrono::steady_clock::time_point Data_begin = std::chrono::steady_clock::now();
    Initialized_data.push_back(meas);
    if(first_point){
        first_point = false;
        second_point = true;
        lidar_frame_count++;
        last_imu_ = meas.imu.back();
        acc_0.x() = meas.imu.back()->linear_acceleration.x;
        acc_0.y() = meas.imu.back()->linear_acceleration.y;
        acc_0.z() = meas.imu.back()->linear_acceleration.z;
        gyr_0.x() = meas.imu.back()->angular_velocity.x;
        gyr_0.y() = meas.imu.back()->angular_velocity.y;
        gyr_0.z() = meas.imu.back()->angular_velocity.z;
        odom.push_back(pose_cur);
        odom_no.push_back(pose_cur);
        odom_Paremi.push_back(pose_cur);
        if(usetrue){
            oss.str("");
            oss << std::fixed << std::setprecision(6) << meas.lidar_beg_time;
            double normalizedDesiredTimestamp = std::stod(oss.str());
            Pose pose_true = {poseMap.find(normalizedDesiredTimestamp)->second.x,
                                poseMap.find(normalizedDesiredTimestamp)->second.y,
                                poseMap.find(normalizedDesiredTimestamp)->second.z,
                                poseMap.find(normalizedDesiredTimestamp)->second.roll,
                                poseMap.find(normalizedDesiredTimestamp)->second.pitch,
                                poseMap.find(normalizedDesiredTimestamp)->second.yaw};
            Trueicpodom.push_back(pose_true);
            cout<<"time:"<<normalizedDesiredTimestamp<<" "<<poseMap.find(normalizedDesiredTimestamp)->second.x<<endl;
            icp_result = pose_true;
        }else{
            icp_result = pose_cur;
        }
        CalibState calibState_first(icp_result.poseto_rotation(), icp_result.poseto_position(), meas.lidar_end_time);
        system_state.push_back(calibState_first);
        return 0;
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
    if(second_point)  //Motion distortion removal for first
    {
        second_point = false;
        auto v_imu_ = Initialized_data[0].imu;
        double dt_ = 0;
        M3D R_imu_(icp_state.rot_end);
        V3D angvel_avr_, pos_imu_(icp_state.pos_end);
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
        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
        {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);
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
        double note = pcl_end_time_ > imu_end_time_ ? 1.0 : -1.0;
        dt_ = note * (pcl_end_time_ - imu_end_time_);
        M3D r = R_imu_ * Exp(V3D(note * angvel_avr_), dt_);
        V3D p = pos_imu_ + vel_cen * (pcl_end_time_ - pcl_beg_time_);
        last_lidar_end_time_ = pcl_end_time_;
        last_imu_ = Initialized_data[0].imu.back();

        //for first:
        /*** undistort each lidar point (backward propagation) ***/
        auto it_pcl = pcl_last.points.end() - 1; //a single point in k-th frame
        for (auto it_kp = GYR_first.end() - 1 ; it_kp != GYR_first.begin(); it_kp--)
        {
            double dt, dt_j;
            auto head = it_kp - 1;
            M3D R; 
            V3D ANGVEL;
            R = (head->rot);
            ANGVEL << VEC_FROM_ARRAY(head->angvel);
            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
                /* Transform to the 'scan-end' IMU frame（I_k frame)*/
                M3D R_i(R * Exp(ANGVEL, dt));
                V3D p_in(it_pcl->x, it_pcl->y, it_pcl->z);
                M3D R_jk = icp_state.offset_R_L_I.transpose() * icp_state.rot_end.transpose() * r.transpose() * R_i * icp_state.offset_R_L_I;
                dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
                V3D p_jk;
                p_jk = - (icp_state.rot_end).transpose() * vel_cen * dt_j;
                V3D P_compensate = R_jk * p_in + p_jk;
                
                /// save Undistorted points
                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);
                if (it_pcl == pcl_last.points.begin()) break;
            }
        }
        Undistortpoint.push_back(pcl_last.makeShared());

    }

    V3D angvel_avr, pos_imu(icp_state.pos_end);
    M3D R_imu(icp_state.rot_end);
    Pose pose_initial = trans2pose(icp_state.rot_end, icp_state.pos_end);
    /*** Initialize IMU pose ***/
    GYR_pose.clear();
    GYR_pose.push_back(imu_accumulative(0.0, angvel_last, R_imu));
    /*** forward propagation at each imu point ***/
    double dt = 0;
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, get_acc_bias(), get_gyro_bias()};
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
        V3D linear_acceleration_;
        V3D angular_velocity_;
        linear_acceleration_.x() = tail->linear_acceleration.x;
        linear_acceleration_.y() = tail->linear_acceleration.y;
        linear_acceleration_.z() = tail->linear_acceleration.z;
        angular_velocity_.x() = tail->angular_velocity.x;
        angular_velocity_.y() = tail->angular_velocity.y;
        angular_velocity_.z() = tail->angular_velocity.z;
        if(tail->header.stamp.toSec() > pcl_end_time){
            dt = pcl_end_time - head->header.stamp.toSec();
            double dt_ = tail->header.stamp.toSec() - pcl_end_time;
            double w1 = dt / (dt + dt_);
            double w2 = dt_ / (dt + dt_);
            linear_acceleration_.x() = w1 * tail->linear_acceleration.x + w2 * tail->linear_acceleration.x;
            linear_acceleration_.y() = w1 * tail->linear_acceleration.y + w2 * tail->linear_acceleration.y;
            linear_acceleration_.z() = w1 * tail->linear_acceleration.z + w2 * tail->linear_acceleration.z;
            angular_velocity_.x() = w1 * tail->angular_velocity.x + w2 * tail->angular_velocity.x;
            angular_velocity_.y() = w1 * tail->angular_velocity.y + w2 * tail->angular_velocity.y;
            angular_velocity_.z() = w1 * tail->angular_velocity.z + w2 * tail->angular_velocity.z;
            acc_0 = linear_acceleration_;
            gyr_0 = angular_velocity_;
        }
        tmp_pre_integration->push_back(dt, linear_acceleration_ / mean_acc_norm * G_m_s2, angular_velocity_);
        angvel_last = angvel_avr;
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        GYR_pose.push_back(imu_accumulative(offs_t, angvel_last, R_imu));
    }
    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    icp_state.vel_end = vel_cen;
    icp_state.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    icp_state.pos_end = pos_imu + vel_cen * (pcl_end_time - pcl_beg_time);
    Pose pose_prediction = trans2pose(icp_state.rot_end, icp_state.pos_end);
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
            M3D R_jk = icp_state.offset_R_L_I.transpose() * icp_state.rot_end.transpose() * R_i * icp_state.offset_R_L_I;
            
            dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
            V3D p_jk;
            p_jk = - icp_state.rot_end.transpose() * icp_state.vel_end * dt_j;
            V3D P_compensate = R_jk * p_in + p_jk;
            
            /// save Undistorted points
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
            if (it_pcl == pcl_current.points.begin()) break;
        }
    }

    Undistortpoint.push_back(pcl_current.makeShared());
    Pose pose_diff = pose_prediction.diffpose(pose_initial);
    pcl::PointCloud<PointType>::Ptr Pointscloud_near = Pointscloud_trans(Undistortpoint.back(), pose_diff);
    cout<<endl<<"--------------------------"<<endl;
    std::chrono::steady_clock::time_point Icp_begin = std::chrono::steady_clock::now();
    Pose icp_trans = pose_diff.addPoses(pose_diff, ParemidICP(Pointscloud_near, Undistortpoint[Undistortpoint.size()-2]));
    std::chrono::steady_clock::time_point Icp_end = std::chrono::steady_clock::now();
    double icp_time = chrono::duration_cast<chrono::duration<double> >(Icp_end - Icp_begin).count();
    cout<<"icp need time : "<< icp_time <<endl;
    pose_cur = pose_cur.addPoses(pose_cur, icp_trans);
    odom.push_back(pose_cur);
    if(usetrue){
        oss.str("");
        oss << std::fixed << std::setprecision(6) << meas.lidar_beg_time;
        double normalizedDesiredTimestamp = std::stod(oss.str());
        Pose pose_true = {poseMap.find(normalizedDesiredTimestamp)->second.x,
                            poseMap.find(normalizedDesiredTimestamp)->second.y,
                            poseMap.find(normalizedDesiredTimestamp)->second.z,
                            poseMap.find(normalizedDesiredTimestamp)->second.roll,
                            poseMap.find(normalizedDesiredTimestamp)->second.pitch,
                            poseMap.find(normalizedDesiredTimestamp)->second.yaw};
        Trueicpodom.push_back(pose_true);
        cout<<"time:"<<normalizedDesiredTimestamp<<" "<<poseMap.find(normalizedDesiredTimestamp)->second.x<<endl;
        icp_result = pose_true;
    }else{
        icp_result = pose_cur;
    }
    icp_result.addtrans_left(icp_state.offset_R_L_I, icp_state.offset_T_L_I);
    CalibState calibState(icp_result.poseto_rotation(), icp_result.poseto_position(), pcl_end_time);
    calibState.pre_integration = tmp_pre_integration;
    system_state.push_back(calibState);
    // cout<<"------------------ParemidICP-----------------------"<<endl;
    // pose_cur_Paremi = pose_cur_Paremi.addPoses(pose_cur_Paremi, ParemidICP(Initialized_data.back().lidar, Initialized_data[Initialized_data.size()-2].lidar));
    // odom_Paremi.push_back(pose_cur_Paremi);
    // cout<<"------------------ICP-----------------------"<<endl;
    // pose_cur_no = pose_cur_no.addPoses(pose_cur_no, doICP(Initialized_data.back().lidar, Initialized_data[Initialized_data.size()-2].lidar));
    // odom_no.push_back(pose_cur_no);
    // icp_result = pose_cur_Paremi;
    // icp_result.addtrans_left(icp_state.offset_R_L_I, icp_state.offset_T_L_I);
    // CalibState calibState(icp_result.poseto_rotation(), icp_result.poseto_position(), pcl_end_time);
    // calibState.pre_integration = tmp_pre_integration;
    // system_state.push_back(calibState);
    std::chrono::steady_clock::time_point Data_end = std::chrono::steady_clock::now();
    double data_time = chrono::duration_cast<chrono::duration<double> >(Data_end - Data_begin).count();
    cout<<"data_processing need time : "<< data_time <<endl;
    icp_state.rot_end = pose_cur.poseto_rotation();
    icp_state.pos_end = pose_cur.poseto_position();
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

void Dynamic_init::solve_Rot_bias_gyro() {
    M3D A;
    V3D b;
    V3D delta_bg;
    A.setZero();
    b.setZero();
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++)
    {
        auto frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->R.transpose() * frame_j->R);
        tmp_A = frame_j->pre_integration->jacobian.template block<3, 3>(3, 12);
        tmp_b = 2 * (frame_j->pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    gyro_bias += delta_bg;

    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end( ); frame_i++)
    {
        auto frame_j = next(frame_i);
        frame_j->pre_integration->repropagate(Vector3d::Zero(), gyro_bias);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void Dynamic_init::RefineGravity(StatesGroup icp_state, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = system_state.size();
    int n_state = all_frame_count * 3 + 2 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();


    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
        {
            auto frame_j = next(frame_i);

            MatrixXd  tmp_A(6, 11);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 3>(0, 8) = -frame_j->pre_integration->jacobian.template block<3, 3>(0, 9);     
            tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p- icp_state.offset_T_L_I\
                        - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I)\
                        - frame_i->R.transpose() * dt * dt / 2 * g0;
            //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
            tmp_A.block<3, 2>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 3>(3, 8) = -frame_j->pre_integration->jacobian.template block<3, 3>(6, 9);
            tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v\
                        - frame_i->R.transpose() * dt * Matrix3d::Identity() * g0;
            //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            cov_inv.setIdentity();
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<5, 5>() += r_A.bottomRightCorner<5, 5>();
            b.tail<5>() += r_b.tail<5>();

            A.block<6, 5>(i * 3, n_state - 5) += r_A.topRightCorner<6, 5>();
            A.block<5, 6>(n_state - 5, i * 3) += r_A.bottomLeftCorner<5, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 5);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

void Dynamic_init::LinearAlignment(StatesGroup icp_state, VectorXd &x){
    int all_pose_count = system_state.size();
    int n_state = all_pose_count * 3 + 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    int i = 0;
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
    {
        auto frame_j = next(frame_i);

        MatrixXd  tmp_A(6, 12);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 9) = -frame_j->pre_integration->jacobian.template block<3, 3>(0, 9);     
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p - icp_state.offset_T_L_I\
                    - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I);
        
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
        tmp_A.block<3, 3>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 9) = -frame_j->pre_integration->jacobian.template block<3, 3>(6, 9);
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<6, 6>() += r_A.bottomRightCorner<6, 6>();
        b.tail<6>() += r_b.tail<6>();

        A.block<6, 6>(i * 3, n_state - 6) += r_A.topRightCorner<6, 6>();
        A.block<6, 6>(n_state - 6, i * 3) += r_A.bottomLeftCorner<6, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    std::ofstream outFile("/home/myx/fighting/dynamic_init_lidar_inertial/src/LiDAR_DYNAMIC_INIT/Log/A_ba.txt");
    outFile<<A;
    outFile.close();
    Vector3d g = x.segment<3>(n_state - 6);
    auto ba = x.segment<3>(n_state - 3);
    auto v_0 = x.segment<3>(0);
    cout<<"size: "<<x.size()<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" ba     " <<  ba.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0.transpose());
    RefineGravity(icp_state, g, x);
    auto ba_ = x.segment<3>(n_state - 4);
    cout<<"size: "<<x.size()<<endl;
    auto v_0_ = x.segment<3>(0);
    cout<<"----------------------------------------------------"<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" ba     " <<  ba.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0_.transpose());
}


void Dynamic_init::RefineGravity_withoutba(StatesGroup icp_state, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = system_state.size();
    int n_state = all_frame_count * 3 + 2;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();


    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
        {
            auto frame_j = next(frame_i);

            MatrixXd  tmp_A(6, 8);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;   
            tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p- icp_state.offset_T_L_I\
                        - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I)\
                        - frame_i->R.transpose() * dt * dt / 2 * g0;
            //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
            tmp_A.block<3, 2>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v\
                        - frame_i->R.transpose() * dt * Matrix3d::Identity() * g0;
            //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            cov_inv.setIdentity();
            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
            b.tail<2>() += r_b.tail<2>();

            A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
            A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 2);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

void Dynamic_init::LinearAlignment_withoutba(StatesGroup icp_state, VectorXd &x){
    int all_pose_count = system_state.size();
    int n_state = all_pose_count * 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    int i = 0;
    for (auto frame_i = system_state.begin(); next(frame_i) != system_state.end(); frame_i++, i++)
    {
        auto frame_j = next(frame_i);

        MatrixXd  tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->pre_integration->sum_dt;
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->delta_p - icp_state.offset_T_L_I\
                    - frame_i->R.transpose() * (frame_j->T - frame_i->T - frame_j->R * icp_state.offset_T_L_I);
        // cout << "delta_p   " << frame_j->pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->R.transpose() * frame_j->R;
        tmp_A.block<3, 3>(3, 6) = frame_i->R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->delta_v;
        // cout << "delta_v   " << frame_j->pre_integration->delta_v.transpose() << endl;
        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    Vector3d g = x.segment<3>(n_state - 3);
    auto v_0 = x.segment<3>(0);
    std::ofstream outFile("/home/myx/fighting/dynamic_init_lidar_inertial/src/LiDAR_DYNAMIC_INIT/Log/A.txt");
    outFile<<A;
    outFile.close();
    // cout<<"size: "<<x.size()<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0.transpose());
    g_ = g.norm();
    RefineGravity_withoutba(icp_state, g, x);
    auto v_0_ = x.segment<3>(0);
    cout<<"----------------------------------------------------"<<endl;
    ROS_WARN_STREAM(" result g     " << g.norm() << " " << g.transpose());
    ROS_WARN_STREAM(" v_0     " <<  v_0_.transpose());
    Grav_L0 = g;
    V_0 = v_0_;
}
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
