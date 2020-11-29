#include<vector>
#include<string>
#include<opencv2/opencv.hpp>
#include"trt.h"
#include<getopt.h>


using namespace std;

vector<float> prepareImage(cv::Mat& img){
    using namespace cv;
    int w=1344;
    int h=800;
    auto scaleSize = cv::Size(h,w);
    Mat resized;
    cv::resize(img,resized,scaleSize,0,0,INTER_CUBIC);
    
    Mat rgb;
    cvtColor(resized,rgb,COLOR_BGR2RGB);

    Mat imgFloat;
    resized.convertTo(imgFloat,CV_32FC3);
    //HWC2CHW

    vector<Mat> input_channels(3);
    cv::split(imgFloat,input_channels);
    vector<float> result(w*h*3);



    auto data =result.data();
    int channelLength = h*w;
    for(int i=0;i<3;++i){
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }
    
    return result;

}

int main(int argc, char *argv[]){ //ARGV[1]
    int opt=0,option_index=0;
    static struct option opts[]={
        {"input-onnx",required_argument,nullptr,'i'},
        {"output-engine",required_argument,nullptr,'o'},
        {"mode",required_argument,nullptr,'m'},
        {0,0,0,0}
    };

    std::string onnx = "test.onnx";
    std::string engine = "testFp16.engine";
    Tn::RUN_MODE mode = Tn::RUN_MODE::FLOAT16;
    while ((opt = getopt_long_only(argc,argv,"i:o:m:",opts,&option_index))!=-1){
        switch (opt)
        {
        case 'i':onnx=std::string(optarg);
            break;
        case 'o':engine=std::string(optarg);
            break;    
        case 'm':{int a=atoi(optarg);
            switch (a){
            case 0:mode=Tn::RUN_MODE::FLOAT32;
                break;
            case 1:mode=Tn::RUN_MODE::FLOAT16;
                break;
            default:
                break;
            };    
            break;}
        default:
            break;
        }
    }

    std::cout<<"input-onnx "<<onnx<<std::endl
            <<"output-engine "<<engine<<std::endl;
    
    Tn::onnx2tensorrt net(onnx,1,mode);
    net.saveEngine(engine);
    std::cout<<"save "<<engine<<std::endl;

    //net.~onnx2tensorrt();
    return 0;

}