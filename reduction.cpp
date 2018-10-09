#include <SYCL/sycl.hpp>
#include <iostream>
#include <sstream>

template <int reduction_class>
class reduce_t;

// Naive algorithm using tree based architecture, and using even threads
// to calculate the result. The algorithm time is log(n), and we use n threads.
// we use modulo operator to distinguish the even threads.
template <>
class reduce_t<0> {
 public:
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    // in_size is equivalent of total number of thread
    index_t global_id = item_id.get_global(0);
    index_t local_id = item_id.get_local(0);
    local_acc[local_id] = (global_id < in_size) ? in_acc[global_id] : 0;
    for (index_t i = 1; i < item_id.get_local_range(0); i *= 2) {
      // wait for all thread to put the data in the local memory
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      if (local_id % (2 * i) == 0)
        local_acc[local_id] += local_acc[local_id + i];
    }
    if (item_id.get_local(0) == 0) {
      out_acc[item_id.get_group(0)] = local_acc[0];
    }
  }
};

// using consecutive thread to calculate the result instead of even threads.
template <>
class reduce_t<1> {
 public:
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    // in_size is equivalent of total number of thread
    index_t global_id = item_id.get_global(0);
    index_t local_id = item_id.get_local(0);
    local_acc[local_id] = (global_id < in_size) ? in_acc[global_id] : 0;
    for (index_t i = 1; i < item_id.get_local_range(0); i *= 2) {
      // wait for all thread to put the data in the local memory
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      // replacing odd threads with contiguous threads
      auto id = local_id * 2 * i;
      if (id < item_id.get_local_range(0)) local_acc[id] += local_acc[id + i];
    }
    if (item_id.get_local(0) == 0) {
      out_acc[item_id.get_group(0)] = local_acc[0];
    }
  }
};

// using consecutive thread to calculate the result instead of even threads.
template <>
class reduce_t<2> {
 public:
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    // in_size is equivalent of total number of thread
    index_t global_id = item_id.get_global(0);
    index_t local_id = item_id.get_local(0);
    local_acc[local_id] = (global_id < in_size) ? in_acc[global_id] : 0;
    for (index_t i = item_id.get_local_range(0) / 2; i > 0; i >>= 1) {
      // wait for all thread to put the data in the local memory
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      // replacing odd threads with contiguous threads
      if (local_id < i) local_acc[local_id] += local_acc[local_id + i];
    }
    if (item_id.get_local(0) == 0) {
      out_acc[item_id.get_group(0)] = local_acc[0];
    }
  }
};

// using consecutive thread to calculate the result instead of even threads.
template <>
class reduce_t<3> {
 public:
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    using output_type = typename write_accessor_t::value_type;
    // in_size is equivalent of total number of thread
    index_t global_id = item_id.get_global(0);
    index_t local_id = item_id.get_local(0);
    output_type private_sum = output_type(0);
    // per thread reduction
    for (int i = global_id; i < in_size; i += item_id.get_global_range(0)) {
      private_sum += ((i < in_size) ? in_acc[i] : output_type(0));
    }
    local_acc[local_id] = private_sum;
    for (index_t i = item_id.get_local_range(0) / 2; i > 0; i >>= 1) {
      // wait for all thread to put the data in the local memory
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      // replacing odd threads with contiguous threads
      if (local_id < i) local_acc[local_id] += local_acc[local_id + i];
    }
    if (item_id.get_local(0) == 0) {
      out_acc[item_id.get_group(0)] = local_acc[0];
    }
  }
};

// with static value for local size to allow compiler to unroll the parallel for
// loop
template <>
class reduce_t<4> {
 public:
  template <int local_size, typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    using output_type = typename write_accessor_t::value_type;
    // in_size is equivalent of total number of thread
    index_t global_id = item_id.get_global(0);
    index_t local_id = item_id.get_local(0);
    output_type private_sum = output_type(0);
    // per thread reduction
    for (int i = global_id; i < in_size; i += item_id.get_global_range(0)) {
      private_sum += ((i < in_size) ? in_acc[i] : output_type(0));
    }
    local_acc[local_id] = private_sum;
    // reduction for loop
    for (index_t i = local_size / 2; i > 0; i >>= 1) {
      // wait for all thread to put the data in the local memory
      item_id.barrier(cl::sycl::access::fence_space::local_space);
      // replacing odd threads with contiguous threads
      if (local_id < i) local_acc[local_id] += local_acc[local_id + i];
    }
    if (item_id.get_local(0) == 0) {
      out_acc[item_id.get_group(0)] = local_acc[0];
    }
  }
};

template <int reduction_class, int local_size>
struct reduction_factory {
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    reduce_t<reduction_class>::template reduce<local_size>(
        in_size, in_acc, local_acc, out_acc, item_id);
  }
};

template <int reduction_class>
struct reduction_factory<reduction_class, -1> {
  template <typename index_t, typename read_accessor_t,
            typename local_accessor_t, typename write_accessor_t>
  static void inline reduce(const index_t &in_size,
                            const read_accessor_t &in_acc,
                            local_accessor_t &local_acc,
                            write_accessor_t &out_acc,
                            cl::sycl::nd_item<1> &item_id) {
    reduce_t<reduction_class>::reduce(in_size, in_acc, local_acc, out_acc,
                                      item_id);
  }
};

template <int reduction_class, int local_size, typename index_t,
          typename read_accessor_t, typename local_accessor_t,
          typename write_accessor_t>
class reduction_t {
 private:
  const read_accessor_t in_acc;
  local_accessor_t local_acc;
  write_accessor_t out_acc;
  const index_t in_size;

 public:
  reduction_t(const read_accessor_t in_acc_, local_accessor_t local_acc_,
              write_accessor_t out_acc_, const index_t in_size_)
      : in_acc(in_acc_),
        local_acc(local_acc_),
        out_acc(out_acc_),
        in_size(in_size_) {}
  // kernel code
  void inline operator()(cl::sycl::nd_item<1> item_id) {
    reduction_factory<reduction_class,
                      ((reduction_class > 3) ? local_size
                                             : -1)>::reduce(in_size, in_acc,
                                                            local_acc, out_acc,
                                                            item_id);
  }
};

//#define static_reduction_class 4;
template <typename index_t, typename data_t>
cl::sycl::buffer<data_t> inline get_out_buffer(
    const index_t num_group, cl::sycl::buffer<data_t> out_buffer) {
  return (num_group > 1)
             ? cl::sycl::buffer<data_t>(cl::sycl::range<1>{size_t(num_group)})
             : out_buffer;
}
// to make global size multiple of local size
template <typename index_t>
inline index_t round_up(const index_t x, const index_t y) {
  return ((x + y - 1) / y) * y;
}
// launching multiple kernel where the partial result is bigger than work group
// load
template <int work_group_load, int k_factor, int reduction_class,
          typename index_t, typename data_t>
void reduction(index_t in_size, cl::sycl::queue &q,
               cl::sycl::buffer<data_t> in_buff,
               cl::sycl::buffer<data_t> out_buffer) {
  using read_accessor_t =
      cl::sycl::accessor<data_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  using write_accessor_t =
      cl::sycl::accessor<data_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;

  using local_accessor_t =
      cl::sycl::accessor<data_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;

  const constexpr index_t local_size = work_group_load / k_factor;
  const index_t global_size = round_up(in_size / k_factor, local_size);
  index_t num_group = global_size / local_size;

  bool condition = (num_group > work_group_load) ? true : false;
  auto temp_buff = get_out_buffer(num_group, out_buffer);

  // submitting the SYCL kernel to the cvengine SYCL queue.
  q.submit([&](cl::sycl::handler &cgh) {
    // getting read access over the sycl buffer A inside the device kernel
    auto in_acc =
        in_buff.template get_access<cl::sycl::access::mode::read>(cgh);
    // getting write access over the sycl buffer C inside the device kernel

    auto out_acc =
        temp_buff.template get_access<cl::sycl::access::mode::write>(cgh);

    auto local_acc = local_accessor_t(local_size, cgh);

    // constructing the kernel
    cgh.parallel_for(
        cl::sycl::nd_range<1>{cl::sycl::range<1>{size_t(global_size)},
                              cl::sycl::range<1>{size_t(local_size)}},
        reduction_t<reduction_class, local_size, index_t, read_accessor_t,
                    local_accessor_t, write_accessor_t>(in_acc, local_acc,
                                                        out_acc, in_size));
  });
  if (condition) {
    // launching a new kernel and passing tem_buff as an input
    reduction<work_group_load, k_factor, reduction_class>(
        num_group, q, temp_buff, out_buffer);
  } else if (num_group > 1) {
    // The temp_buff size is smaller than the work_group_load
    auto host_out_acc =
        out_buffer.template get_access<cl::sycl::access::mode::write>();
    auto host_in_acc =
        temp_buff.template get_access<cl::sycl::access::mode::read>();
    // reduce the remaining on the host
    for (index_t i = 0; i < num_group; i++) {
      host_out_acc[0] += host_in_acc[i];
    }
  }
}

int main(int argc, char *argv[]) {
  using data_t = double;
  using index_t = int;
  index_t in_size;
  static constexpr index_t work_group_load = 256;
  static constexpr index_t reduction_class = 4;
  const constexpr index_t k_factor = (reduction_class > 2) ? 2 : 1;
  std::istringstream ss(argv[1]);
  if (!(ss >> in_size))
    std::cerr << "Invalid input size " << argv[1]
              << ". Please insert the correct input size " << '\n';
  // auto global_size = round_up(in_size / k_factor, local_size);
  // We initialised the A and B as an input vector
  std::vector<data_t> input(in_size, data_t(1));
  // The output vector does not need to be initialized
  std::vector<data_t> output(1);
  {  // beginning of SYCL objects' scope
    // constructing a SYCL queue for CVengine OpenCL device where automatically
    // build the underlying context and command_queue for the chosen device.
    auto q = cl::sycl::queue(
        (cl::sycl::default_selector()), [&](cl::sycl::exception_list l) {
          bool error = false;
          for (auto e : l) {
            try {
              std::rethrow_exception(e);
            } catch (const cl::sycl::exception &e) {
              auto clError = e.get_cl_code();
              std::cout << e.what() << "CLERRORCODE : " << clError << std::endl;
              error = true;
            }
          }
          if (error) {
            throw std::runtime_error("SYCL errors detected");
          }
        });
    // input SYCL buffer A
    auto in_buff =
        cl::sycl::buffer<data_t>(input.data(), cl::sycl::range<1>(in_size));
    // output SYCL buffer C
    auto out_buff = cl::sycl::buffer<data_t>(output.data(), 1);

    // call reduction function
    reduction<work_group_load, k_factor, reduction_class>(in_size, q, in_buff,
                                                          out_buff);
  }  // end of SYCL objects' scope
  auto reference = 0;
  for (int i = 0; i < in_size; i++) {
    reference += input[i];
  }
  if (output[0] != reference) {
    std::cout << "The result is wrong. expected : " << reference
              << " vs  calculated: " << output[0] << "\n";
    return 1;
  } else {
    std::cout << "The result is correct."
              << "\n";
  }
  return 0;
}
