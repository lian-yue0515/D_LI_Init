#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <lidar_dynamic_init/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <color.h>
#include <scope_timer.hpp>
#include <pcl/common/transforms.h>

using namespace std;
using namespace Eigen;

#define USE_IKFOM

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)         // Gravaty const in GuangDong/China
#define DIM_STATE (18)        // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)       // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV   (1)
#define NUM_MATCH_POINTS    (5)
#define MAX_MEAS_DIM        (10000)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef lidar_dynamic_init::Pose6D Pose6D;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
using M9D = Eigen::Matrix<double, 9, 9>;
using M6D = Eigen::Matrix<double, 6, 6>;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

const M3D Eye3d(M3D::Identity());
const M3F Eye3f(M3F::Identity());
const V3D Zero3d(0, 0, 0);
const V3D E3d(1, 1, 1);
const V3F Zero3f(0, 0, 0);


extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern Eigen::Vector3d G;

class Utility
{
public:
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }
};
struct GYR_{
    GYR_(){
        rot.Identity();
        angvel = Zero3d;
        offset_time = 0;
    };
    V3D angvel;
    M3D rot;
    double offset_time;
};
struct MeasureGroup     // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
};

struct StatesGroup
{
    StatesGroup() {
		this->rot_end = M3D::Identity();
		this->pos_end = Zero3d;
        this->offset_R_L_I = M3D::Identity();
        this->offset_T_L_I = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g  = Zero3d;
        this->bias_a  = Zero3d;
        this->gravity = Zero3d;
	};
    void set_extrinsic(const V3D &transl, const M3D &rot)
    {
        this->offset_T_L_I = transl;
        this->offset_R_L_I = rot;
    }
    StatesGroup(const StatesGroup& b) {
		this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->offset_R_L_I = b.offset_R_L_I;
        this->offset_T_L_I = b.offset_T_L_I;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
	};

    StatesGroup& operator=(const StatesGroup& b)
	{
        this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->offset_R_L_I = b.offset_R_L_I;
        this->offset_T_L_I = b.offset_T_L_I;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        return *this;
	};

    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        StatesGroup a;
		a.rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		a.pos_end = this->pos_end + state_add.block<3,1>(3,0);
        a.offset_R_L_I = this->offset_R_L_I *  Exp(state_add(6,0), state_add(7,0), state_add(8,0));
        a.offset_T_L_I = this->offset_T_L_I + state_add.block<3,1>(9,0);
        a.vel_end = this->vel_end + state_add.block<3,1>(12,0);
        a.bias_g  = this->bias_g  + state_add.block<3,1>(15,0);
        a.bias_a  = this->bias_a  + state_add.block<3,1>(18,0);
        a.gravity = this->gravity + state_add.block<3,1>(21,0);
		return a;
	};

    StatesGroup& operator+=(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        this->rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		this->pos_end += state_add.block<3,1>(3,0);
        this->offset_R_L_I = this->offset_R_L_I * Exp(state_add(6,0), state_add(7,0), state_add(8,0));
        this->offset_T_L_I += state_add.block<3,1>(9,0);
        this->vel_end += state_add.block<3,1>(12,0);
        this->bias_g  += state_add.block<3,1>(15,0);
        this->bias_a  += state_add.block<3,1>(18,0);
        this->gravity += state_add.block<3,1>(21,0);
		return *this;
	};

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b)
	{
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3,1>(0,0)  = Log(rotd);
        a.block<3,1>(3,0)  = this->pos_end - b.pos_end;
        M3D offsetd(b.offset_R_L_I.transpose() * this->offset_R_L_I);
        a.block<3,1>(6,0) = Log(offsetd);
        a.block<3,1>(9,0) = this->offset_T_L_I - b.offset_T_L_I;
        a.block<3,1>(12,0)  = this->vel_end - b.vel_end;
        a.block<3,1>(15,0)  = this->bias_g  - b.bias_g;
        a.block<3,1>(18,0) = this->bias_a  - b.bias_a;
        a.block<3,1>(21,0) = this->gravity - b.gravity;
		return a;
	};

    void resetpose()
    {
        this->rot_end = M3D::Identity();
		this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

	M3D rot_end;      // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;      // the estimated position at the end lidar point (world frame)
    M3D offset_R_L_I; // Rotation from Lidar frame L to IMU frame I
    V3D offset_T_L_I; // Translation from Lidar frame L to IMU frame I
    V3D vel_end;      // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;       // gyroscope bias
    V3D bias_a;       // accelerator bias
    V3D gravity;      // the estimated gravity acceleration
};

template<typename T>
T rad2deg(T radians)
{
  return radians * 180.0 / PI_M;
}

template<typename T>
T deg2rad(T degrees)
{
  return degrees * PI_M / 180.0;
}

template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}

template<typename T>
auto imu_accumulative(const double t, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 3> &R)
{
    GYR_ rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.angvel[i] = g(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot(i, j) = R(i,j);
    }
    return move(rot_kp);
}


/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template<typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);
    
    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}


struct Pose {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    V3D poseto_position()
    {
        V3D position(x, y, z);
        return position;
    }
    M3D poseto_rotation()
    {
        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
        
        Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
        
        M3D rotationMatrix = q.matrix();
        return rotationMatrix;
    }
    
    Pose diffpose(Pose _p2)
    {
        Eigen::Affine3f SE3_p1 = pcl::getTransformation(x, y, z, roll, pitch, yaw);
        Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
        Eigen::Matrix4f SE3_delta0 = SE3_p2.matrix().inverse() * SE3_p1.matrix();
        Eigen::Affine3f SE3_delta;
        SE3_delta.matrix() = SE3_delta0;
        float dx, dy, dz, droll, dpitch, dyaw;
        pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);

        return Pose{dx, dy, dz, droll, dpitch, dyaw};
    }
    Pose addPoses(const Pose& pose1,const Pose& pose2) {
        Pose poseOut;
        Eigen::Affine3f posein_a = pcl::getTransformation(pose1.x, pose1.y, pose1.z, pose1.roll, pose1.pitch, pose1.yaw);
        Eigen::Affine3f poseout_a = pcl::getTransformation(pose2.x, pose2.y, pose2.z, pose2.roll, pose2.pitch, pose2.yaw);;
        Eigen::Affine3f Out_a = posein_a * poseout_a;
        float tx, ty, tz, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(Out_a, tx, ty, tz, roll, pitch, yaw);
        poseOut.x = tx;
        poseOut.y = ty;
        poseOut.z = tz;
        poseOut.roll = roll;
        poseOut.pitch = pitch;
        poseOut.yaw = yaw;
        return poseOut;
    }
    void addtrans_left(const M3D rot, const V3D tran) {
        Eigen::Affine3f transformation_matrix = Eigen::Affine3f::Identity();
        transformation_matrix.linear() = rot.cast<float>();
        transformation_matrix.translation() = tran.cast<float>();

        Eigen::Affine3f pose = pcl::getTransformation(x, y, z, roll, pitch, yaw);
        Eigen::Affine3f Out_a = transformation_matrix * pose;
        float tx, ty, tz, troll, tpitch, tyaw;
        pcl::getTranslationAndEulerAngles(Out_a, tx, ty, tz, troll, tpitch, tyaw);
        x = tx;
        y = ty;
        z = tz;
        roll = troll;
        pitch = tpitch;
        yaw = tyaw;
    }
    void addtrans_right(const M3D rot, const V3D tran) {
        Eigen::Affine3f transformation_matrix = Eigen::Affine3f::Identity();
        transformation_matrix.linear() = rot.cast<float>();
        transformation_matrix.translation() = tran.cast<float>();

        Eigen::Affine3f pose = pcl::getTransformation(x, y, z, roll, pitch, yaw);
        Eigen::Affine3f Out_a =  pose * transformation_matrix;
        float tx, ty, tz, troll, tpitch, tyaw;
        pcl::getTranslationAndEulerAngles(Out_a, tx, ty, tz, troll, tpitch, tyaw);
        x = tx;
        y = ty;
        z = tz;
        roll = troll;
        pitch = tpitch;
        yaw = tyaw;
    }
};


#endif