#include"trt.h"
#include<getopt.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<numeric>
#include<algorithm>
#include"config.h"
#include"postprocess.h"
#include"postprocess_gpu.h"
#include<chrono>



int main(int argc, char* argv[]){

    int opt=0,option_index = 0;
    std::string engine = "test16.trt";
    std::string inputs = "test.jpg";
    std::string output = "result";
    int show = 0;
    int save = 0;
    static struct option opts[]=
    {
        {"engine-model",required_argument,nullptr,'e'},
        {"inputs",required_argument,nullptr,'i'},
        {"output",required_argument,nullptr,'o'},
        {"show",no_argument,nullptr,'v'},
        {"save",no_argument,nullptr,'s'}
        {0,0,0,0}
    };

    while((opt=getopt_long_only(argc,argv,"e:i:o:s:",opts,&option_index))!=-1)
    {
        switch (opt)
        {
        case 'e':engine = std::string(optarg);break;
        case 'i':inputs = std::string(optarg);break;
        case 'o':output = std::string(optarg);break;
        case 'v':show = 1;break;        
        case 's':save = 1;break;
        
        default:
            break;
        }
    }
    Tn::onnx2tensorrt net(engine);

    float*seg_preds_cpu=NULL;
    float*cate_labels_cpu=NULL;
    float*cate_scores_cpu=NULL;
    int ind_size;

    if(inputs.find("jpg") != std::string::npos || inputs.find("png") != std::string::npos){

        std::cout<<"read image file "<<inputs<<std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat inputimage = cv::imread(inputs);
        cv::Mat rgb;
        cv::cvtColor(inputimage,rgb,cv::COLOR_BGR2RGB);
        cv::Mat outputimage;
        cv::Size resizeoutput = cv::Size(outputsize[0],outputsize[1]);
        cv::resize(inputimage,outputimage,resizeoutput);
        auto end_time = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"process spend time "<<total<<" ms"<<std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        ind_size = net.infer_gpupost(rgb,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu);

        end_time = std::chrono::high_resolution_clock::now();
        total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

        start_time = std::chrono::high_resolution_clock::now();

        std::string output_name = output+".jpg";
        if(show){
            cv::namedWindow("output image",1);
            vis_seg(outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output_name,save,true);
            cv::destroyAllWindows();
        }
        else
        {
            vis_seg(outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output_name,save,false);
        }
        
        free(seg_preds_cpu);
        free(cate_scores_cpu);
        free(cate_labels_cpu);

        end_time = std::chrono::high_resolution_clock::now();
        total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
        std::cout<<"vis spend time "<<total<<" ms"<<std::endl;

    }
    else if (inputs.find("mp4") != std::string::npos || inputs.find("avi") != std::string::npos)
    {   
        std::cout<<"read video from "<<inputs<<std::endl;
        std::string output_name = output+".avi";
        cv::VideoCapture cap;
        cv::VideoWriter writer(output_name,cv::VideoWriter::fourcc('X','V','I','D'),30,cv::Size(outputsize[0],outputsize[1]));
        cap.open(inputs);

        if(!cap.isOpened()){
            std::cout<<"Error: video stream can't be opened!"<<std::endl;
            return 1;
        }

        if(show){
            cv::namedWindow("output",1);
        }
        
        cv::Mat input_image;
        while (cap.read(input_image))
        {
            
            if(!input_image.data){
                continue;
            }
            auto start_time = std::chrono::high_resolution_clock::now();
            cv::Mat rgb;
            cv::cvtColor(input_image,rgb,cv::COLOR_BGR2RGB);
            cv::Mat outputimage;
            cv::Size resizeoutput = cv::Size(outputsize[0],outputsize[1]);
            cv::resize(input_image,outputimage,resizeoutput);
            auto end_time = std::chrono::high_resolution_clock::now();
            float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"process spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            ind_size = net.infer_gpupost(rgb,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu);

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            if(show){
                vis_seg(writer,outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output,save,true);
            }
            else
            {
                vis_seg(writer,outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output,save,false);
            }
            
            if (seg_preds_cpu!=NULL){
                free(seg_preds_cpu);
                seg_preds_cpu=NULL;
            }
            if (cate_scores_cpu!=NULL){
                free(cate_scores_cpu);
                cate_scores_cpu=NULL;
            }
            if (cate_labels_cpu!=NULL){
                free(cate_labels_cpu);
                cate_labels_cpu=NULL;
            }

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"vis spend time "<<total<<" ms"<<std::endl;
        }
        cap.release();
        
    }
    else if (inputs.find("txt") != std::string::npos )
    {
        std::cout<<"read image list "<<inputs<<std::endl;
        std::ifstream inputImageNameList(inputs);
        std::list<std::string> fileNames;

        if(!inputImageNameList.is_open()){
            std::cout<<"can not read image list "<<inputs<<std::endl;
            return 1;
        }
        std::string strLine;
        while (std::getline(inputImageNameList,strLine))
        {
            fileNames.push_back(strLine);
        }
        inputImageNameList.close();

        for(auto it=fileNames.begin();it!=fileNames.end();++it){

            auto start_time = std::chrono::high_resolution_clock::now();
            cv::Mat inputimage = cv::imread(*it);
            cv::Mat rgb;
            cv::cvtColor(inputimage,rgb,cv::COLOR_BGR2RGB);
            cv::Mat outputimage;
            cv::Size resizeoutput = cv::Size(outputsize[0],outputsize[1]);
            cv::resize(inputimage,outputimage,resizeoutput);
            auto end_time = std::chrono::high_resolution_clock::now();
            float total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"process spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            ind_size = net.infer_gpupost(rgb,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu);

            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"infer spend time "<<total<<" ms"<<std::endl;

            start_time = std::chrono::high_resolution_clock::now();

            std::string output_name = output+".jpg";
            if(show){
                cv::namedWindow("output image",1);
                vis_seg(outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output_name,save,true);
                cv::destroyAllWindows();
            }
            else
            {
                vis_seg(outputimage,seg_preds_cpu,cate_labels_cpu,cate_scores_cpu,ind_size,output_name,save,false);
            }
        
            if (seg_preds_cpu!=NULL){
                free(seg_preds_cpu);
                seg_preds_cpu=NULL;
            }
            if (cate_scores_cpu!=NULL){
                free(cate_scores_cpu);
                cate_scores_cpu=NULL;
            }
            if (cate_labels_cpu!=NULL){
                free(cate_labels_cpu);
                cate_labels_cpu=NULL;
            }
            end_time = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float,std::milli>(end_time-start_time).count();
            std::cout<<"vis spend time "<<total<<" ms"<<std::endl;

        }

    }
    
    else
    {
        std::cout<<"do not support this format,please use file end with .mp4/.avi/.jpg/.png"<<std::endl;
    }

    std::cout<<"success all"<< std::endl;

    return 0;

}

