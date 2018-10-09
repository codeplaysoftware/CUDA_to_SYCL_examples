#include <SYCL/sycl.hpp>
int main() {
  // array size
  auto array_size = 256;
  // We initialised the A and B as an input vector
  std::vector<float> A(array_size, 1.0f);
  std::vector<float> B(array_size, 1.0f);
  // The output vector does not need to be initialized
  std::vector<float> C(array_size);
  { // beginning of SYCL objects' scope
    // constructing a SYCL queue for CVengine OpenCL device where automatically
    // build the underlying context and command_queue for the chosen device.
    auto sycl_queue = cl::sycl::queue;
    // input SYCL buffer A
    auto A_buff =
        cl::sycl::buffer<float>(A.data(), cl::sycl::range<1>(array_size));
    // input SYCL buffer B
    auto B_buff =
        cl::sycl::buffer<float>(B.data(), cl::sycl::range<1>(array_size));
    // output SYCL buffer C
    auto C_buff =
        cl::sycl::buffer<float>(C.data(), cl::sycl::range<1>(array_size));
    // getting the total number of compute untis
    auto num_groups =
        sycl_queue.get_device()
            .get_info<cl::sycl::info::device::max_compute_units>();
    // getting the maximum work group size per thread
    auto work_group_size =
        sycl_queue.get_device()
            .get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    auto total_threads = num_groups * work_group_size;
    // submitting the SYCL kernel to the cvengine SYCL queue.
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      // getting read access over the sycl buffer A inside the device kernel
      auto A_acc = A_buff.get_access<cl::sycl::access::mode::read>(cgh);
      // getting read access over the sycl buffer B inside the device kernel
      auto B_acc = B_buff.get_access<cl::sycl::access::mode::read>(cgh);
      // getting write access over the sycl buffer C inside the device kernel
      auto C_acc = C_buff.get_access<cl::sycl::access::mode::write>(cgh);
      // constructing the kernel
      cgh.parallel_for<class vec_add>(
          cl::sycl::range<1>{total_threads}, [=](cl::sycl::item<1> itemId) {
            auto id = itemId.get_id(0);
            for (auto i = id; i < C_acc.get_count(); i += itemId.get_range()[0])
              C_acc[i] = A_acc[i] + B_acc[i];
          });
    });
  } // end of SYCL objects' scope
  return 0;
}
