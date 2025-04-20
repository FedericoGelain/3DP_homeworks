#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  //this class receives in input the correspondence source <-> target point
  PointDistance(Eigen::Vector3d source, Eigen::Vector3d target) : source(source), target(target) {}

  //, will apply the transformation
  //(parameters to optimize) to the source pointand compute as residual the distance between the transformed
  //source point and the target one
  template <typename T>
  bool operator()(const T* const transformation, T* residuals) const { 
    T source_point[3]; 
    T transformed_source_point[3];

    source_point[0] = T(source[0]);
    source_point[1] = T(source[1]);
    source_point[2] = T(source[2]);

    //apply the rotation to the source point
    ceres::AngleAxisRotatePoint(transformation, source_point, transformed_source_point);

    //add the corresponding translation
    transformed_source_point[0] += transformation[3];
    transformed_source_point[1] += transformation[4];
    transformed_source_point[2] += transformation[5];

    //store in the residual the distance between the transformed source point and the target point
    residuals[0] = T(target[0]) - transformed_source_point[0];
    residuals[1] = T(target[1]) - transformed_source_point[1];
    residuals[2] = T(target[2]) - transformed_source_point[2];

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d source, const Eigen::Vector3d target) {
    return new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(new PointDistance(source, target));
  }

  Eigen::Vector3d source;
  Eigen::Vector3d target;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //if the mode provided is different from the two available, abort the registration
  if(mode != "svd" && mode != "lm") {
    std::cout << "choose the registration mode between \"svd\" and \"lm\" " << std::endl;
    return ;
  }
  
  //this initialization ensures no convergence at the first iteration
  double prev_rmse = std::numeric_limits<double>::min();
  double curr_rmse = std::numeric_limits<double>::max();

  int iter = 0;

  //continue the main loop until one of those two conditions happens:
  //-the difference between the current and previous RMSE is smaller than the relative RMSE, so convergence has been reached
  //-the maximum number of iterations has been reached
  while (iter < max_iteration && std::abs(curr_rmse - prev_rmse) >= relative_rmse) {
    //for each point in the source point cloud, find the corresponding closest point in the target point cloud
    //(storing the corresponding indices) and the overall rmse between the two point clouds
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> data = find_closest_point(threshold);

    //update the rmse values accordingly
    prev_rmse = curr_rmse;
    curr_rmse = std::get<2>(data);

    std::cout << "Iteration " << iter << ", RMSE = " << curr_rmse << std::endl;

    //retrieve the current transformation computed with either svd or lm
    Eigen::Matrix4d curr_transf;

    //retrieve the current transformation based on the mode chosen
    if (mode == "svd")
      curr_transf = get_svd_icp_transformation(std::get<0>(data), std::get<1>(data));
    else if (mode == "lm")
      curr_transf = get_lm_icp_registration(std::get<0>(data), std::get<1>(data));

    //and apply it to the source_for_icp point cloud
    source_for_icp_.Transform(curr_transf);

    //retrieve the transformation computed in the previous iteration
    Eigen::Matrix4d prev_transf = get_transformation();

    //and compute the full transformation to apply to the source point cloud to match source_for_icp
    Eigen::Matrix4d full_transf = Eigen::Matrix4d::Identity();

    full_transf = curr_transf * prev_transf;

    //update transformation_
    set_transformation(full_transf);

    iter++;
  }

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

  //create the KD tree from the target point cloud
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  
  int num_source_points  = source_for_icp_.points_.size();

  //utility vectors when searching the nearest neighbours for each source point
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_for_icp_.points_[i];

    //search the closest neighbour to the current source point in the target KD tree,
    //returning its corresponding index and distance from it
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

    //if the distance between the two points is lower than the threshold, it means that they are inliers
    if(dist2[0] < threshold) {
      //so register their indices
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);

      //and consider their distance for the rmse computation
      rmse = rmse * i/(i+1) + dist2[0]/(i+1);
    }
  }

  rmse = sqrt(rmse);

  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  //select from both point clouds only the correspondences (points whose indices are stored in the vectors provided as parameters)
  std::vector<Eigen::Vector3d> correspondences_source;
  std::vector<Eigen::Vector3d> correspondences_target;
  
  for(int i = 0; i < source_indices.size(); i++) {
    correspondences_source.push_back(source_for_icp_.points_[source_indices[i]]);
    correspondences_target.push_back(target_.points_[target_indices[i]]);
  }

  //compute the source point cloud centroid
  Eigen::Vector3d source_centroid;

  for(size_t i = 0; i < correspondences_source.size(); i++) {
    source_centroid += correspondences_source[i];
  }

  source_centroid /= correspondences_source.size();
  
  //compute the target point cloud centroid
  Eigen::Vector3d target_centroid;

  for(size_t i = 0; i < correspondences_target.size(); i++) {
    target_centroid += correspondences_target[i];
  }

  target_centroid /= correspondences_target.size();

  Eigen::Matrix3d W;

  //for all the points in the two point clouds, ordered based on find_closest_point(),
  //subtract the corresponding centroid and compute the matrix W
  for(int i = 0; i < source_indices.size(); i++) {
    Eigen::Vector3d source_shifted_point;
    Eigen::Vector3d target_shifted_point;

    source_shifted_point = correspondences_source[i] - source_centroid;
    target_shifted_point = correspondences_target[i] - target_centroid;
    
    W += target_shifted_point * source_shifted_point.transpose();
  }

  //decompose W using SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

  //compute R, handling the reflection case
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity(3,3);
  
  R = svd.matrixU() * svd.matrixV().transpose();

  if(R.determinant() == -1) {
    //define the diagonal matrix to change the sign of the last column of U
    Eigen::DiagonalMatrix<double, 3> T (1,1,-1);

    R = svd.matrixU() * T * svd.matrixV().transpose();
  }

  //compute the translation vector using R
  Eigen::Vector3d t = target_centroid - R * source_centroid;

  //set the obtained rotation matrix and translation vector in the transformation matrix
  transformation.block(0,0,3,3) = R;
  transformation.block(0,3,3,1) = t;

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  ceres::Problem problem;
  ceres::Solver::Summary summary;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();
  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    //create the cost function for the current pair of points
    ceres::CostFunction *cost_function = PointDistance::Create(source_for_icp_.points_[source_indices[i]], target_.points_[target_indices[i]]);

    //and add a residual block inside the problem
    problem.AddResidualBlock(cost_function, nullptr, &transformation_arr[0]);
  }

  ceres::Solve(options, &problem, &summary);

  //retrieve the optimized parameters, stored in transformation_arr
  Eigen::Matrix3d R;

  //use the Euler angles to compute the corresponding rotation matrix
  R = Eigen::AngleAxisd(transformation_arr[0], Eigen::Vector3d::UnitX()) * 
      Eigen::AngleAxisd(transformation_arr[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(transformation_arr[2], Eigen::Vector3d::UnitZ());

  Eigen::Vector3d t (transformation_arr[3], transformation_arr[4], transformation_arr[5]);

  //set the obtained rotation matrix and translation vector in the transformation matrix
  transformation.block(0,0,3,3) = R;
  transformation.block(0,3,3,1) = t;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


