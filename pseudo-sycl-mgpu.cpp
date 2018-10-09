...;

// getting the list of all supported sycl platforms
auto platfrom_list = cl::sycl::platform::get_platforms();
// getting the list of devices from the platform
auto device_list = platform.get_devices();
// looping over platforms
for (const auto &platform : platfrom_list) {
  // looping over devices
  for (const auto &device : device_list) {
    auto queue = cl::sycl::queue(device);
    // submitting a kernel to a the sycl queue
    queue.submit([&](cl::sycl::handler &cgh) {
      ....
          // sycl kernel
          cgh.parallel_for(....);
    });
  }
}
...;
