#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "sgm.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#define NUM_DIRS 3
#define PATHS_PER_SCAN 8

using namespace std;
using namespace cv;
using namespace Eigen;
static char hamLut[256][256];
static int directions[NUM_DIRS] = {0, -1, 1};

//compute values for hamming lookup table
void compute_hamming_lut()
{
  for (uchar i = 0; i < 255; i++)
  {
    for (uchar j = 0; j < 255; j++)
    {
      uchar census_xor = i^j;
      uchar dist=0;
      while(census_xor)
      {
        ++dist;
        census_xor &= census_xor-1;
      }
      
      hamLut[i][j] = dist;
    }
  }
}

namespace sgm 
{
  SGM::SGM(unsigned int disparity_range, unsigned int p1, unsigned int p2, float conf_thresh, unsigned int window_height, unsigned window_width):
  disparity_range_(disparity_range), p1_(p1), p2_(p2), conf_thresh_(conf_thresh), window_height_(window_height), window_width_(window_width)
  {
    compute_hamming_lut();
  }

  // set images and initialize all the desired values
  void SGM::set(const  cv::Mat &left_img, const  cv::Mat &right_img, const  cv::Mat &right_mono)
  {
    views_[0] = left_img;
    views_[1] = right_img;
    mono_ = right_mono;


    height_ = left_img.rows;
    width_ = right_img.cols;
    pw_.north = window_height_/2;
    pw_.south = height_ - window_height_/2;
    pw_.west = window_width_/2;
    pw_.east = width_ - window_height_/2;
    init_paths();
    cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    inv_confidence_.resize(height_, vector<float>(width_));
    aggr_cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    path_cost_.resize(PATHS_PER_SCAN, ul_array3D(height_, ul_array2D(width_, ul_array(disparity_range_)))
    );
  }

  //initialize path directions
  void SGM::init_paths()
  {
    for(int i = 0; i < NUM_DIRS; ++i)
    {
      for(int j = 0; j < NUM_DIRS; ++j)
      {
        // skip degenerate path
        if (i==0 && j==0)
          continue;
        paths_.push_back({directions[i], directions[j]});
      }
    }
  }

  //compute costs and fill volume cost cost_
  void SGM::calculate_cost_hamming()
  {
    uchar census_left, census_right, shift_count;
    cv::Mat_<uchar> census_img[2];
    cv::Mat_<uchar> census_mono[2];
    cout << "\nApplying Census Transform" <<endl;
    
    for( int view = 0; view < 2; view++)
    {
      census_img[view] = cv::Mat_<uchar>::zeros(height_,width_);
      census_mono[view] = cv::Mat_<uchar>::zeros(height_,width_);

      for (int r = 1; r < height_ - 1; r++)
      {
        uchar *p_center = views_[view].ptr<uchar>(r),
              *p_census = census_img[view].ptr<uchar>(r);
        p_center += 1;
        p_census += 1;

        for(int c = 1; c < width_ - 1; c++, p_center++, p_census++)
        {
          uchar p_census_val = 0, m_census_val = 0, shift_count = 0;
          for (int wr = r - 1; wr <= r + 1; wr++)
          {
            for (int wc = c - 1; wc <= c + 1; wc++)
            {

              if( shift_count != 4 )//skip the center pixel
              {
                p_census_val <<= 1;
                m_census_val <<= 1;
                if(views_[view].at<uchar>(wr,wc) < *p_center ) //compare pixel values in the neighborhood
                  p_census_val = p_census_val | 0x1;

              }
              shift_count ++;
            }
          }
          *p_census = p_census_val;
        }
      }
    }

    cout <<"\nFinding Hamming Distance" <<endl;
    
    for(int r = window_height_/2 + 1; r < height_ - window_height_/2 - 1; r++)
    {
      for(int c = window_width_/2 + 1; c < width_ - window_width_/2 - 1; c++)
      {
        for(int d=0; d<disparity_range_; d++)
        {
          long cost = 0;
          for(int wr = r - window_height_/2; wr <= r + window_height_/2; wr++)
          {
            uchar *p_left = census_img[0].ptr<uchar>(wr),
                  *p_right = census_img[1].ptr<uchar>(wr);


            int wc = c - window_width_/2;
            p_left += wc;
            p_right += wc + d;



            const uchar out_val = census_img[1].at<uchar>(wr, width_ - window_width_/2 - 1);


            for(; wc <= c + window_width_/2; wc++, p_left++, p_right++)
            {
              uchar census_left, census_right, m_census_left, m_census_right;
              census_left = *p_left;
              if (c+d < width_ - window_width_/2)
              {
                census_right= *p_right;

              }

              else
              {
                census_right= out_val;
              }


              cost += ((hamLut[census_left][census_right]));
            }
          }
          cost_[r][c][d]=cost;
        }
      }
    }
  }

  void SGM::compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path)
  {
    unsigned long prev_cost, best_prev_cost, no_penalty_cost, penalty_cost, 
                  small_penalty_cost, big_penalty_cost;

    //////////////////////////// Code to be completed (1/4) /////////////////////////////////
    // Complete the compute_path_cost() function that, given: 
    // i) a single pixel p defined by its coordinates cur_x and cur_y; 
    // ii) a path with index cur_path (cur_path=0,1,..., PATHS_PER_SCAN - 1, a path for 
    //     each direction), and;
    // iii) the direction increments direction_x and direction_y associated with the path 
    //      with index cur_path (that are the dx,dy increments to move along the path 
    //      direction, both can be -1, 0, or 1), 
    // should compute the path cost for p for all the possible disparities d from 0 to 
    // disparity_range_ (excluded, already defined). The output should be stored in the 
    // tensor (already allocated) path_cost_[cur_path][cur_y][cur_x][d], for all possible d.
    /////////////////////////////////////////////////////////////////////////////////////////

    // if the processed pixel is the first:
    if(cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
    {
      //it means that there is no previous pixel in the path
      for(int d = 0; d < disparity_range_; ++d)
      {
        //so simply set the path cost to the E_data term
        path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d];
      }
    }

    else
    {
      //otherwise, we need also to compute the smoothness term

      //coordinates of the previous pixel in the path
      int prev_x = cur_x - direction_x;
      int prev_y = cur_y - direction_y;

      //first min_Delta E(q, Delta) is computed, which will be used also for the big penalty term
      best_prev_cost = path_cost_[cur_path][prev_y][prev_x][0];

      for(int d = 1; d < disparity_range_; ++d) {
        //check the cost of the previous pixel in the path for the current disparity
        prev_cost = path_cost_[cur_path][prev_y][prev_x][d];
        
        //if it's smaller than the current best one, update it
        if(prev_cost < best_prev_cost) {
          best_prev_cost = prev_cost;
        }
      }

      //third term: min_Delta E(q, Delta) + c2
      big_penalty_cost = best_prev_cost + p2_;

      for(int d = 0; d < disparity_range_; ++d) {
        //first term: E(q, f_q)
        no_penalty_cost = path_cost_[cur_path][prev_y][prev_x][d];

        //second term: E(q, f_q) + c1
        if(d == 0) {
          //there's only the d + 1 term
          small_penalty_cost = path_cost_[cur_path][prev_y][prev_x][d + 1] + p1_;  
        }
        else {
          if (d == disparity_range_ - 1) {
            //there's only the d - 1 term
            small_penalty_cost = path_cost_[cur_path][prev_y][prev_x][d - 1] + p1_;
          }
          else {
            //default case, both d - 1 and d + 1 available. We can already pick the smallest between the two
            small_penalty_cost = std::min(path_cost_[cur_path][prev_y][prev_x][d - 1], path_cost_[cur_path][prev_y][prev_x][d + 1]) + p1_;
          }
        }

        //the E_smooth term is the minimum value across the three penalty terms
        penalty_cost = std::min(std::min(no_penalty_cost, small_penalty_cost), big_penalty_cost);

        //finally, the path cost is equal to the E_data term + E_smooth term - minimum cost found for the previous pixel
        path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d] + penalty_cost - best_prev_cost;
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////
  }

  
  void SGM::aggregation()
  {
    
    //for all defined paths
    for(int cur_path = 0; cur_path < PATHS_PER_SCAN; ++cur_path)
    {

      //////////////////////////// Code to be completed (2/4) /////////////////////////////////
      // Initialize the variables start_x, start_y, end_x, end_y, step_x, step_y with the 
      // right values, after that uncomment the code below
      /////////////////////////////////////////////////////////////////////////////////////////

      int dir_x = paths_[cur_path].direction_x;
      int dir_y = paths_[cur_path].direction_y;
      
      int start_x, start_y, end_x, end_y, step_x, step_y;

      //x direction
      //moving left to right (this is also the default initialization for dir_x = 0)
      if(dir_x == 1 || dir_x == 0) {
        start_x = pw_.west;
        end_x = pw_.east;
        step_x = 1;
      }
      else if(dir_x == -1) { //dir_x = -1, so moving right to left
        start_x = pw_.east;
        end_x = pw_.west;
        step_x = -1;        
      }

      //y direction
      //moving top to bottom (this is also the default initialization for dir_y = 0)
      if(dir_y == 1 || dir_y == 0) {
        start_y = pw_.north;
        end_y = pw_.south;
        step_y = 1;
      }
      else if(dir_y == -1) { //dir_y = -1, so moving bottom to top
        start_y = pw_.south;
        end_y = pw_.north;
        step_y = -1;        
      }
      
      for(int y = start_y; y != end_y ; y+=step_y)
      {
        for(int x = start_x; x != end_x ; x+=step_x)
        {
          compute_path_cost(dir_y, dir_x, y, x, cur_path);
        }
      }
      
      /////////////////////////////////////////////////////////////////////////////////////////
    }
    
    float alpha = (PATHS_PER_SCAN - 1) / static_cast<float>(PATHS_PER_SCAN);
    //aggregate the costs
    for (int row = 0; row < height_; ++row)
    {
      for (int col = 0; col < width_; ++col)
      {
        for(int path = 0; path < PATHS_PER_SCAN; path++)
        {
          unsigned long min_on_path = path_cost_[path][row][col][0];
          int disp =  0;

          for(int d = 0; d<disparity_range_; d++)
          {
            aggr_cost_[row][col][d] += path_cost_[path][row][col][d];
            if (path_cost_[path][row][col][d]<min_on_path)
              {
                min_on_path = path_cost_[path][row][col][d];
                disp = d;
              }

          }
          inv_confidence_[row][col] += (min_on_path - alpha * cost_[row][col][disp]);

        }
      }
    }

  }


  void SGM::compute_disparity()
  {
      calculate_cost_hamming();
      aggregation();
      disp_ = Mat(Size(width_, height_), CV_8UC1, Scalar::all(0));
      int n_valid = 0;
      //vector to save the unscaled disparities from the right-to-left initial guess
      std::vector<uchar> d_mono;      
      //vector to save the good confidence disparities computed with sgm
      std::vector<uchar> d_sgm;

      for (int row = 0; row < height_; ++row)
      {
          for (int col = 0; col < width_; ++col)
          {
              unsigned long smallest_cost = aggr_cost_[row][col][0];
              int smallest_disparity = 0;
              for(int d=disparity_range_-1; d>=0; --d)
              {

                  if(aggr_cost_[row][col][d]<smallest_cost)
                  {
                      smallest_cost = aggr_cost_[row][col][d];
                      smallest_disparity = d; 

                  }
              }
              inv_confidence_[row][col] = smallest_cost - inv_confidence_[row][col];

              // If the following condition is true, the disparity at position (row, col) has a good confidence
              if (inv_confidence_[row][col] > 0 && inv_confidence_[row][col] <conf_thresh_)
              {
                //////////////////////////// Code to be completed (3/4) /////////////////////////////////
                // Since the disparity at position (row, col) has a good confidence, it can be added 
                // togheter with the corresponding unscaled disparity from the right-to-left initial 
                // guess mono_.at<uchar>(row, col) to the pool of disparity pairs that will be used 
                // to estimate the unknown scale factor.    
                /////////////////////////////////////////////////////////////////////////////////////////

                //save the sgm disparity at position (row,col), scaled in the [0,255] range
                d_sgm.push_back(smallest_disparity * 255.0 / disparity_range_);

                //and the corresponding first guess disparity at the same position
                d_mono.push_back(mono_.at<uchar>(row, col));

                //and increase the number of valid disparties found
                n_valid++;
                
                /////////////////////////////////////////////////////////////////////////////////////////
              }

              disp_.at<uchar>(row, col) = smallest_disparity*255.0/disparity_range_;

          }
      }

      //////////////////////////// Code to be completed (4/4) /////////////////////////////////
      // Using all the disparity pairs accumulated in the previous step, 
      // estimate the unknown scaling factor and scale the initial guess disparities 
      // accordingly. Finally,  and use them to improve/replace the low-confidence SGM 
      // disparities.
      /////////////////////////////////////////////////////////////////////////////////////////

      //define the nx2 matrix A = [d_mono 1], obtained by stacking the nx1 vector d_mono and the nx1 vector of ones horizontally
      Eigen::MatrixXf A(n_valid,2);
      
      for(int i = 0; i < n_valid; i++) {
        A(i, 0) = d_mono[i];
        A(i, 1) = 1;
      }
      
      //define the vector b_sgm as a nx1 matrix (to simplify the computation of the scaling factors)
      Eigen::MatrixXf b(n_valid, 1);

      for(int i = 0; i < n_valid; i++) {
        b(i, 0) = d_sgm[i];
      }

      //x is the 2x1 vector that will contain the scaling factors h and k stacked vertically
      Eigen::VectorXf x(2);

      //x = (A^T * A)^-1 * A^T * b
      x = (A.transpose() * A).inverse() * A.transpose() * b;

      //retreive the scaling factors
      float h = x(0);
      float k = x(1);

      for (int row = 0; row < height_; ++row)
      {
          for (int col = 0; col < width_; ++col)
          {
              //finally, find all the disparities computed with sgm that yield a low confidence
              //and replace them using the SCALED initial guess disparities at the same position
              if (!(inv_confidence_[row][col] > 0 && inv_confidence_[row][col] < conf_thresh_))
              {
                disp_.at<uchar>(row, col) = h * mono_.at<uchar>(row, col) + k;
              }
          }
      }

      /////////////////////////////////////////////////////////////////////////////////////////

  }

  float SGM::compute_mse(const cv::Mat &gt)
  {
    cv::Mat1f container[2];
    cv::normalize(gt, container[0], 0, 85, cv::NORM_MINMAX);
    cv::normalize(disp_, container[1], 0, disparity_range_, cv::NORM_MINMAX);

    cv::Mat1f  mask = min(gt, 1);
    cv::multiply(container[1], mask, container[1], 1);
    float error = 0;
    for (int y=0; y<height_; ++y)
    {
      for (int x=0; x<width_; ++x)
      {
        float diff = container[0](y,x) - container[1](y,x);
        error+=(diff*diff);
      }
    }
    error = error/(width_*height_);
    return error;
  }

  void SGM::save_disparity(char* out_file_name)
  {
    imwrite(out_file_name, disp_);
    return;
  }
  

}

