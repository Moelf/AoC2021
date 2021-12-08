#include <iostream>
#include <fstream>
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <string>

int main(int argc, char** argv) {
    std::string fname = "../../input1";
    std::ifstream in_file;
    in_file.open(fname);

    xt::xarray<double> data = xt::load_csv<double>(in_file);
    xt::xarray<double> depth_diff = xt::diff(data, 1, 0);

    auto val = xt::sum(depth_diff > 0);
    std::cout << "Part 1:" << val << std::endl;

    int result = 0;
    int this_val = 0;
    for (size_t i = 0; i < data.size() - 3; i++) {
        this_val = data(i+3, 0) > data(i, 0) ? 1 : 0;
        result += this_val;
    }

    std::cout << "Part 2:" << result << std::endl;

    return 0;
}