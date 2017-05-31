#include <iostream>
#include <memory>
#include <functional>	// ref()
#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <limits>


//OPENGL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "linmath.h"

// #include "ImGuiUtils.h"

#include "stereo_test/ShaderProgram.h"
#include "stereo_test/Gui.h"
// #include "baseline/Renderer.h"
// #include "baseline/Scene.h"
// #include "baseline/Simulation.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/core/ocl.hpp"
#include "opencv2/ximgproc.hpp"
// #include "opencv2/edge_filter.hpp"

#include <stereo_test/extendedfast.h>
#include "elas.h"
#include "image.h"


#define OPENCV_THREAD_COUNT 4
#define DOWNSAMPLE 2
#define MAXIMUM_DISPARITY 30


//first implementation was done using
// REAL-TIME LOCAL STEREO MATCHING USING GUIDED IMAGE FILTERING


cv::Mat mat2gray(const cv::Mat& src)
{
    cv::Mat dst;
    cv::normalize(src, dst, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    return dst;
}



static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error %d: %s\n", error, description);
}


void downsample (cv::Mat& img, cv::Mat& dst, float factor){
  cv::Size size(img.cols/factor,img.rows/factor);
  cv::resize(img,dst,size);
}


void compute_cost_volume(std::vector<cv::Mat>& cost_volume , cv::Mat img_left, cv::Mat img_right, float alpha, float truncation_color, float truncation_gradient  ){
  // downsample
  // downsample(img_left,img_left,DOWNSAMPLE);
  // downsample(img_right,img_right,DOWNSAMPLE);

  cv::Mat sobel_x_left,sobel_x_right;
  cv::Mat img_left_gray, img_right_gray;
  cv::cvtColor(img_left,img_left_gray, CV_BGR2GRAY);
  cv::cvtColor(img_right,img_right_gray, CV_BGR2GRAY);
  cv::Sobel(img_left_gray, sobel_x_left, img_left_gray.depth(), 1, 0, 3);
  cv::Sobel(img_right_gray, sobel_x_right, img_right_gray.depth(), 1, 0, 3);



  //TODO not completiely optimal since you don't need to check for all the values in the right img, only the ones to the left of pixel i,j
  for (size_t i = 0; i < img_left.rows; i++) {
    for (size_t j = 0; j < img_left.cols; j++) {

      // cv::Vec3b intensity_left = img_left.at<cv::Vec3b>(i,j);

//       uchar b = frame.data[frame.channels()*(frame.cols*y + x) + 0];
// uchar g = frame.data[frame.channels()*(frame.cols*y + x) + 1];
// uchar r = frame.data[frame.channels()*(frame.cols*y + x) + 2];

      for (size_t d = 0; d < MAXIMUM_DISPARITY ; d++) {

        if ((j-d)<0 ){
          continue; //TODO not exactly the most effiecient way since you have an if inside a loop
        }

        // std::cout << "d is " << d << '\n';
        //Compute cost between pixel at (i,j) in the left image and pixel at (i,j-d) in the right image

        float color_diff=0;
        cv::Vec3b intensity_left = img_left.at<cv::Vec3b>(i,j);
        cv::Vec3b intensity_right = img_right.at<cv::Vec3b>(i,j-d);
        for (size_t c_idx = 0; c_idx < img_left.channels(); c_idx++) {
          color_diff+=std::abs( intensity_left.val[c_idx]- intensity_right[c_idx] );
        }


        float grad_diff=0;
        grad_diff = std::abs(sobel_x_left.at<uchar>(i,j) - sobel_x_right.at<uchar>(i,j-d));


        float cost=0;
        // float alpha=0.5;
        // float truncation_color=100;
        // float truncation_gradient=150;
        cost=alpha* std::min (color_diff, truncation_color) + (1-alpha)* std::min(grad_diff, truncation_gradient);
        // cost=color_diff;

        // std::cout << "cost is " << cost << '\n';

        cost_volume[d].at<float>(i,j)=cost;


      }
    }
  }


}



void winner_take_all(std::vector<cv::Mat> cost_volume, cv::Mat& disparity_map){

  disparity_map= cv::Mat (cost_volume[0].rows,cost_volume[0].cols, CV_32S, cv::Scalar(0) );

  for (size_t i = 0; i < cost_volume[0].rows; i++) {
    for (size_t j = 0; j < cost_volume[0].cols; j++) {
      float min_cost=std::numeric_limits<float>::max();
      int disparity=0;

      for (size_t d = 0; d < cost_volume.size(); d++) {
        if (cost_volume[d].at<float>(i,j)<min_cost){
          min_cost=cost_volume[d].at<float>(i,j);
          disparity=d;
        }
      }

      // std::cout << "dispairty is " << disparity << '\n';
      disparity_map.at<int>(i,j)=disparity;
      // disparity_map.at<int>(i,j)=disparity;



    }
  }


}


int main(int, char**){


    // Setup window
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui OpenGL3 example", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);


    //Objects
    // std::shared_ptr<Scene> scene (new Scene);
    // std::shared_ptr<Renderer> renderer (new Renderer);
    // std::shared_ptr<Simulation> sim (new Simulation);
    // renderer->init(scene);
    // sim->init(scene,renderer);


    //Imgui
    ImGui_ImplGlfwGL3_Init(window, true);
    std::unique_ptr<Gui> gui (new Gui());
    gui->init_fonts(); //needs to be initialized inside the context



    //Shaders
    // std::shared_ptr<ShaderProgram> shader_program (new ShaderProgram);
    // shader_program->initFromFiles("shaders/vert_shader.glsl", "shaders/frag_shader.glsl");
    // shader_program->addUniform("MVP");
    // shader_program->addAttribute("vCol");
    // shader_program->addAttribute("vPos");





    //OpenCV
    cv::setNumThreads(OPENCV_THREAD_COUNT);
    // std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/Aloe_half/view1.png";
    // std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/Aloe_half/view5.png";
    // std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/Aloe/view1.png";
    // std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/Aloe/view5.png";

    std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/cones_quarter/im2.png";
    std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/cones_quarter/im6.png";

    // std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/teddy_quarter/im2.png";
    // std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/teddy_quarter/im6.png";
    //
    // std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/reindeer_half/view1.png";
    // std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/reindeer_half/view5.png";

    // std::string img_left_path="/media/alex/Data/Master/SHK/Data/middelbury/tsukuba/scene1.row3.col2.ppm";
    // std::string img_left_right="/media/alex/Data/Master/SHK/Data/middelbury/tsukuba/scene1.row3.col5.ppm";

    cv::Mat img_left_rgb=cv::imread(img_left_path);
    cv::Mat img_right_rgb=cv::imread(img_left_right);
    cv::Mat img_left_gray;
    cv::Mat img_right_gray;
    cv::cvtColor(img_left_rgb,img_left_gray, CV_BGR2GRAY);
    cv::cvtColor(img_right_rgb,img_right_gray, CV_BGR2GRAY);
    cv::Mat img_left_kp;
    cv::Mat img_right_kp;
    std::vector<cv::KeyPoint> kp_left;
    std::vector<cv::KeyPoint> kp_right;

    int smoothing_algorithm_type=2;
    const char* smoothing_algorithms[] = {
      "Domain transform",
      "Guided filter",
      "Fast global smoother" };


    //FAST PARAMS
    float fast_tresh=10;
    int fast_type=2;
    bool fast_nonmaxsupress=true;
    float exfast_adaptivity=1.0f;
    const char* desc[] =
    {
          "cv::FastFeatureDetector::TYPE_7_12",
          "cv::FastFeatureDetector::TYPE_5_8",
          "cv::FastFeatureDetector::TYPE_9_16",
    };


    //BLOB params
    cv::SimpleBlobDetector::Params params_blob;
    std::vector<cv::KeyPoint> kp_blob;

    //AKAZE


    //SOBEL
    cv::Mat sobelx;


    //Domain transform
    cv::Mat smoothed;
    float dt_sigma_spacial=10;
    float dt_sigma_color=30;
    int dt_mode=0;
    int dt_iters=3;
    const char* dt_mode_list[] =
    {
          "DTF_NC",
          "DTF_RF",
          "DTF_IC",
    };


    //Guided filter
    int gf_radius=3;
    float gf_eps=10.0f;


    //Fast global smoother ---FASTEST
    float fast_smoother_lambda=100.0f;
    float fast_smoother_color=5.0f;
    float fast_smoother_lambda_attenuation=0.25f;
    int fast_smoother_iters=2;



    //My Stereo things
    std::vector<cv::Mat> cost_volume;
    cv::Mat disparity_map;
    downsample (img_left_gray,img_left_gray,DOWNSAMPLE);
    downsample (img_right_gray,img_right_gray,DOWNSAMPLE);
    downsample (img_left_rgb,img_left_rgb,DOWNSAMPLE);
    downsample (img_right_rgb,img_right_rgb,DOWNSAMPLE);
    //alocate volume
    cost_volume.clear();
    for (size_t i = 0; i < MAXIMUM_DISPARITY; i++) {
      cost_volume.push_back(cv::Mat(img_left_gray.rows, img_left_gray.cols, CV_32F, cv::Scalar(255))  );
    }
    float alpha=0.5;
    float truncation_color=100;
    float truncation_gradient=150;
    ImVec2 transfer_func[10];
    transfer_func[0].x=-1;
    float time_avg=0;
    int times_count=1;


    //ELAS params
    Elas::parameters elas_param;



    while(true){
      glfwPollEvents();
      ImGui_ImplGlfwGL3_NewFrame();
      int win_width, win_height;
      glfwGetFramebufferSize(window, &win_width, &win_height);
      glViewport(0, 0, win_width, win_height);
      glClear(GL_COLOR_BUFFER_BIT);

      gui->update();
      // sim->update();
      // renderer->draw();


      ImGui::ListBox("Smoothing algorithm", &smoothing_algorithm_type, smoothing_algorithms, IM_ARRAYSIZE(smoothing_algorithms));


      //GUI FAST
      if (ImGui::TreeNode("FAST")){
        ImGui::Combo("FAST TYPE", &fast_type, desc, IM_ARRAYSIZE(desc));
        ImGui::TreePop();
      }


      //GUI FAST
      if (ImGui::TreeNode("FAST")){
        ImGui::Checkbox("fast_nonmaxsupress", &fast_nonmaxsupress);
        ImGui::SliderFloat("fast_thresh", &fast_tresh, 0.0f, 50.0f);
        ImGui::Combo("FAST TYPE", &fast_type, desc, IM_ARRAYSIZE(desc));
        ImGui::SliderFloat("esfast_adaptivity", &exfast_adaptivity, 0.0f, 2.0f);
        ImGui::TreePop();
      }

      //GUI BLOB
      if (ImGui::TreeNode("BLOB")){
        ImGui::SliderFloat("minDistBetweenBlobs", &params_blob.minDistBetweenBlobs, 0.0f, 50.0f);
        ImGui::Checkbox("filterByArea", &params_blob.filterByArea);
        ImGui::SliderFloat("params.minArea", &params_blob.minArea, 0.0f, 500.0f);
        ImGui::SliderFloat("params.maxArea", &params_blob.maxArea, 0.0f, 500.0f);
        ImGui::TreePop();
      }

      //GUI DOMAIN TRANSFORM
      if (ImGui::TreeNode("DOMAIN TRANSFORM")){
        ImGui::SliderFloat("dt_sigma_spacial", &dt_sigma_spacial, 0.0f, 50.0f);
        ImGui::SliderFloat("dt_sigma_color", &dt_sigma_color, 0.0f, 255.0f);
        ImGui::Combo("dt_mode", &dt_mode, dt_mode_list, IM_ARRAYSIZE(dt_mode_list));
        ImGui::SliderInt("dt_iters", &dt_iters, 0, 10);
        ImGui::TreePop();
      }

      //GUI DOMAIN TRANSFORM
      if (ImGui::TreeNode("GUIDED FILTER ")){
        ImGui::SliderInt("gf_radius", &gf_radius, 0, 50);
        ImGui::SliderFloat("gf_eps", &gf_eps, 0.0f, 2550.0f);
        ImGui::TreePop();
      }

      //GUI FAST GLOBAL SMOOTHER
      if (ImGui::TreeNode("FAST GLOBAL SMOOTHER ")){
        ImGui::SliderFloat("fast_smoother_lambda", &fast_smoother_lambda, 0.0f, 200.0f);
        ImGui::SliderFloat("fast_smoother_color", &fast_smoother_color, 0.0f, 255.0f);
        ImGui::SliderFloat("fast_smoother_lambda_attenuation", &fast_smoother_lambda_attenuation, 0.0f, 1.0f);
        ImGui::SliderInt("fast_smoother_iters", &fast_smoother_iters, 1, 10);
        ImGui::TreePop();
      }

      //GUI MYSTEREO
      if (ImGui::TreeNode("STEREO ")){
        ImGui::SliderFloat("alpha", &alpha, 0.0f, 1.0f);
        ImGui::SliderFloat("truncation_color", &truncation_color, 0.0f, 255.0f);
        ImGui::SliderFloat("truncation_gradient", &truncation_gradient, 0.0f, 255.0f);
        ImGui::TreePop();
      }

      //GUI ELAS
      if (ImGui::TreeNode("ELAS ")){
        ImGui::SliderInt("disp_min", &elas_param.disp_min, 0, 255);
        ImGui::SliderInt("disp_max", &elas_param.disp_max, 0, 255);
        ImGui::SliderFloat("support_threshold", &elas_param.support_threshold, 0.0f, 1.0f);
        ImGui::SliderInt("support_texture", &elas_param.support_texture, 0, 100);
        ImGui::SliderInt("candidate_stepsize", &elas_param.candidate_stepsize, 0, 10);
        ImGui::SliderInt("incon_window_size", &elas_param.incon_window_size, 0, 10);
        ImGui::SliderInt("incon_threshold", &elas_param.incon_threshold, 0, 10);
        ImGui::SliderInt("incon_min_support", &elas_param.incon_min_support, 0, 10);
        ImGui::Checkbox("add_corners", &elas_param.add_corners);
        ImGui::SliderInt("grid_size", &elas_param.grid_size, 0, 100);
        ImGui::SliderFloat("beta", &elas_param.beta, 0.0f, 1.0f);
        ImGui::SliderFloat("gamma", &elas_param.gamma, 0.0f, 10.0f);
        ImGui::SliderFloat("sigma", &elas_param.sigma, 0.0f, 10.0f);
        ImGui::SliderFloat("sradius", &elas_param.sradius, 0.0f, 10.0f);
        ImGui::SliderInt("match_texture", &elas_param.match_texture, 0, 10);
        ImGui::SliderInt("lr_threshold", &elas_param.lr_threshold, 0, 10);
        ImGui::SliderFloat("speckle_sim_threshold", &elas_param.speckle_sim_threshold, 0.0f, 5.0f);
        ImGui::SliderInt("speckle_size", &elas_param.speckle_size, 0, 1000);
        ImGui::SliderInt("ipol_gap_width", &elas_param.ipol_gap_width, 0, 5000);
        ImGui::Checkbox("filter_median", &elas_param.filter_median);
        ImGui::Checkbox("filter_adaptive_mean", &elas_param.filter_adaptive_mean);
        ImGui::Checkbox("postprocess_only_left", &elas_param.postprocess_only_left);
        ImGui::Checkbox("subsampling", &elas_param.subsampling);
        ImGui::TreePop();
      }








      // cv::Ptr<cv::SimpleBlobDetector> blob_detector = cv::SimpleBlobDetector::create(params_blob);
      // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
      // cv::Ptr<cv::ORB> orb = cv::ORB::create();
      cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(fast_tresh,fast_nonmaxsupress,fast_type);
      cv::Ptr<cv::ximgproc::DTFilter> dt_filter = cv::ximgproc::createDTFilter(img_left_gray,dt_sigma_spacial,dt_sigma_color, dt_mode,dt_iters);
      cv::Ptr<cv::ximgproc::GuidedFilter> gf_filter = cv::ximgproc::createGuidedFilter(img_left_gray,gf_radius,gf_eps);
      cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> fast_smoother_filter = cv::ximgproc::createFastGlobalSmootherFilter(img_left_rgb,fast_smoother_lambda,fast_smoother_color,fast_smoother_lambda_attenuation,fast_smoother_iters);
      cv::Ptr<cv::FeatureDetector> leftFeatureDetector = new sparsestereo::ExtendedFAST(fast_nonmaxsupress, fast_tresh, exfast_adaptivity, false, 2);
      Elas elas(elas_param);


      std::chrono::steady_clock::time_point begin_detect = std::chrono::steady_clock::now();

      //FAST
      // fast->detect(img_left_gray, kp_left);

      //EXFAST (has issued in opencv 3)
      // leftFeatureDetector->detect(img_left_gray, kp_left);


      //BLOB
      // blob_detector->detect( img_left_gray, kp_left);


      //AKAZE
      // akaze->detectAndCompute(img_left_gray, cv::noArray(), kp_left, cv::noArray());


      //ORB
      // orb->detectAndCompute(img_left_gray, cv::noArray(), kp_left, cv::noArray());

      //SOBEL
      // cv::Sobel(img_left_gray, sobelx, img_left_gray.depth(), 1, 0, 3);
      // cv::threshold( sobelx, sobelx, 120, 255,0 );


      //DOMAIN TRANSFORM
      // if (smoothing_algorithm_type==0) dt_filter->filter(img_left_gray,smoothed,img_left_gray.depth());
      //
      //
      // //GUIDED FILTER
      // if (smoothing_algorithm_type==1) gf_filter->filter(img_left_gray,smoothed,img_left_gray.depth());
      //
      // //Fast global smoother
      // if (smoothing_algorithm_type==2) fast_smoother_filter->filter 	(img_left_gray,smoothed);
      //
      //
      //

      // my STEREO STUFF
      compute_cost_volume(cost_volume, img_left_rgb,img_right_rgb, alpha, truncation_color, truncation_gradient);


      // for (size_t i = 0; i < cost_volume.size(); i++) {
      //   cv::imshow("cost_unfiltered", mat2gray(cost_volume[i]));
      //   cv::waitKey(0);
      // }

      //Filter cost volume
      for (size_t d = 0; d < cost_volume.size(); d++) {
        if (smoothing_algorithm_type==0) dt_filter->filter(cost_volume[d],cost_volume[d],cost_volume[d].depth());
        if (smoothing_algorithm_type==1) gf_filter->filter(cost_volume[d],cost_volume[d],cost_volume[d].depth());
        if (smoothing_algorithm_type==2) {
          //   cv::Rect roi= cv::Rect(d,0,img_left_gray.cols-d, img_left_gray.rows );
          //   cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> fast_smoother_filter = cv::ximgproc::createFastGlobalSmootherFilter(img_left_gray(roi),fast_smoother_lambda,fast_smoother_color,fast_smoother_lambda_attenuation,fast_smoother_iters);
          // fast_smoother_filter->filter 	(cost_volume[d](roi),cost_volume[d](roi));
          fast_smoother_filter->filter 	(cost_volume[d],cost_volume[d]);
        }
      }

      // for (size_t i = 0; i < cost_volume.size(); i++) {
      //   cv::imshow("cost_filtered", mat2gray(cost_volume[i]));
      //   cv::waitKey(0);
      // }


      winner_take_all(cost_volume,disparity_map);


      //Elas
        // // allocate memory for disparity images
        // // get image width and height
        // int32_t width  = img_left_gray.cols;
        // int32_t height = img_left_gray.rows;
        //
        // // allocate memory for disparity images
        // const int32_t dims[3] = {width,height,width}; // bytes per line = width
        // float* D1_data = (float*)malloc(width*height*sizeof(float));
        // float* D2_data = (float*)malloc(width*height*sizeof(float));
        //
        // elas.process(img_left_gray.data,img_right_gray.data,D1_data,D2_data,dims);
        //
        // // find maximum disparity for scaling output disparity images to [0..255]
        // float disp_max = 0;
        // for (int32_t i=0; i<width*height; i++) {
        //   if (D1_data[i]>disp_max) disp_max = D1_data[i];
        //   if (D2_data[i]>disp_max) disp_max = D2_data[i];
        // }
        //
        // // copy float to uchar
        // image<uchar> *D1 = new image<uchar>(width,height);
        // image<uchar> *D2 = new image<uchar>(width,height);
        // disparity_map= cv::Mat (img_left_gray.rows,img_left_gray.cols, CV_8U, cv::Scalar(0) );
        // for (int32_t i=0; i<width*height; i++) {
        //   disparity_map.data[i] = (uint8_t)std::max(255.0*D1_data[i]/disp_max,0.0);
        // }
        //
        // delete D1;
        // delete D2;
        // free(D1_data);
        // free(D2_data);





      std::chrono::steady_clock::time_point end_detect= std::chrono::steady_clock::now();
      // std::cout << "TIME: " << "detect and compute" << " = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_detect - begin_detect).count() <<std::endl;
      // std::string time= std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end_detect - begin_detect).count());
      // std::cout << "time is " << time << '\n';
      float time_f= std::chrono::duration_cast<std::chrono::nanoseconds>(end_detect - begin_detect).count() /1e6 ;
      time_avg = time_avg + (time_f - time_avg)/times_count;
      times_count++;
      ImGui::Text("Time for detection (%.3f ms): ", time_f );
      ImGui::Text("Time for detection (%.3f ms): ", time_avg );
      ImGui::Text("Number of keypoints left img: %ld: ", kp_left.size() );


      // disparity_map=mat2gray(disparity_map);
      double min, max;
      cv::minMaxLoc(disparity_map, &min, &max);
      // std::cout << "disparity map min max is " << min << " " << max << '\n';
      // for (size_t i = 0; i < disparity_map.rows; i++) {
      //   for (size_t j = 0; j < disparity_map.cols; j++) {
      //     if (disparity_map.at<int>(i,j) > (60/DOWNSAMPLE)  ){
      //       disparity_map.at<int>(i,j)=0;
      //     }
      //   }
      // }

      //GUI transfer function
      if (ImGui::Curve("Das editor", ImVec2(400, 200), 10, transfer_func))
      {
      }
      for (size_t i = 0; i < disparity_map.rows; i++) {
        for (size_t j = 0; j < disparity_map.cols; j++) {
          // disparity_map.at<uchar>(i,j)  = ImGui::CurveValue( disparity_map.at<uchar>(i,j)/255.0 , 10, transfer_func)*255;
          // disparity_map.at<int>(i,j)  = ImGui::CurveValue( disparity_map.at<int>(i,j)/(float)max , 10, transfer_func)*max;

        }
      }

      cv::minMaxLoc(disparity_map, &min, &max);
      std::cout << "disparity map min max is " << min << " " << max << '\n';





      //DRAW STUFF
      cv::drawKeypoints(img_left_rgb, kp_left, img_left_kp, cv::Scalar(0.0,250.0,0.0), cv::DrawMatchesFlags::DEFAULT );
      if (!kp_left.empty()){
        cv::imshow("img_left_kp", img_left_kp);
      }

      // cv::imshow("sobel", mat2gray(sobelx));
      // cv::imshow("sobel", sobelx);

      cv::imshow("disparity", mat2gray(disparity_map));
      // downsample_in_place(smoothed,2);
      // downsample(smoothed, smoothed,2);
      // cv::imshow("img_left_kp", mat2gray(smoothed));
      // cv::imshow("img_left_kp", mat2gray(smoothed));
      cv::waitKey(1);




      ImGui::Render();
      glfwSwapBuffers(window);
    }



    // Cleanup
    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();

    return 0;
}
