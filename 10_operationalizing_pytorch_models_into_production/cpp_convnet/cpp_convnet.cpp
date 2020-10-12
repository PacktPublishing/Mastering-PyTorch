#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    Mat img = imread(argv[2], IMREAD_GRAYSCALE);
    resize(img, img, Size(28, 28));
    auto input_ = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
    auto input = input_.permute({2,0,1}).unsqueeze_(0).reshape({1, 1, img.rows, img.cols}).toType(c10::kFloat).div(255);
    input = (input - 0.1302) / 0.3069;
    auto module = torch::jit::load(argv[1]);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output_ = module.forward(inputs).toTensor();
    cout << output_ << '\n';
    auto output = output_.argmax(1);
    cout << output << '\n';
    return 0;
}