#include "Dynamic_init.h"

Dynamic_init::Dynamic_init(){
    fout_LiDAR_meas.open(FILE_DIR("LiDAR_meas.txt"), ios::out);
    fout_IMU_meas.open(FILE_DIR("IMU_meas.txt"), ios::out);
    data_accum_length = 4;
    lidar_frame_count = 0;
    gyro_bias = Zero3d;
    acc_bias = Zero3d;
    Grav_L0 = Zero3d;
    V_0 = Zero3d;
    first_deDistortLidar = false;
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

bool Dynamic_init::Data_processing(MeasureGroup& meas)
{
    Initialized_data.push_back(meas);
    if( lidar_frame_count == 0 )
    {
        lidar_frame_count++;
        return false;
    }
    // Step 1: De-distort lidar data
    // deDistortLidar(meas);
    
    // Step 2: Use ICP for trajectory estimation and save the odomentry.
    // estimateTrajectoryICP(meas.lidar);
    
    // Step 3: Perform IMU pre-integration and estimate the odomentry.
    // preintegrateIMU(meas.imu);
    
    // Step 4: Check the count of received lidar data frames.
    lidar_frame_count++;
    if (lidar_frame_count <= data_accum_length)
    {
        // If less than or equal to 4 frames, exit the function.
        return false;
    }
    return true; 
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