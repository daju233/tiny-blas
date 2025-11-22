#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>  // for thrust::plus, maximum, etc.
#include <thrust/random.h>
#include <iostream>
#include <limits>
#include <numeric>  // for std::accumulate (host reference)

// 自定义二元函数对象：用于计算乘积
struct multiply {
    __host__ __device__
    float operator()(const float& a, const float& b) const {
        return a * b;
    }
};

int main() {
    const int N = 10;

    // 1. 在主机生成数据（例如：[1.0, 2.0, ..., 10.0]）
    thrust::host_vector<float> h_data(N);
    std::iota(h_data.begin(), h_data.end(), 1.0f); // 填充 1,2,...,10

    // 或者用随机数（取消注释以下两行即可）：
    // thrust::default_random_engine rng;
    // thrust::uniform_real_distribution<float> dist(1.0f, 5.0f);
    // for (int i = 0; i < N; ++i) h_data[i] = dist(rng);

    std::cout << "Input data: ";
    for (float x : h_data) std::cout << x << " ";
    std::cout << "\n\n";

    // 2. 拷贝到 GPU
    thrust::device_vector<float> d_data = h_data;

    // 3. 各种 reduce 操作

    // (a) 求和：∑x_i
    float sum = thrust::reduce(d_data.begin(), d_data.end(), 0.0f, thrust::plus<float>());
    // 等价简写（默认就是 plus）：
    // float sum = thrust::reduce(d_data.begin(), d_data.end());

    // (b) 最大值
    float max_val = thrust::reduce(d_data.begin(), d_data.end(),
                                   -std::numeric_limits<float>::infinity(),
                                   thrust::maximum<float>());

    // (c) 最小值
    float min_val = thrust::reduce(d_data.begin(), d_data.end(),
                                   std::numeric_limits<float>::infinity(),
                                   thrust::minimum<float>());

    // (d) 乘积：∏x_i （注意：初始值必须是 1.0f）
    float product = thrust::reduce(d_data.begin(), d_data.end(), 1.0f, multiply{});

    // 4. 打印结果
    std::cout << "Sum      : " << sum << "\n";
    std::cout << "Max      : " << max_val << "\n";
    std::cout << "Min      : " << min_val << "\n";
    std::cout << "Product  : " << product << "\n";

    // 5. 验证（可选）：用 CPU 计算对比
    float cpu_sum = std::accumulate(h_data.begin(), h_data.end(), 0.0f);
    std::cout << "\nCPU sum check: " << (std::abs(sum - cpu_sum) < 1e-5 ? "OK" : "ERROR") << "\n";

    return 0;
}