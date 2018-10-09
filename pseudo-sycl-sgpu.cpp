...;

// constructing the quue for an specefic device
auto my_queue = cl::sycl::queue(device_selector);

// submitting a kernel to a the sycl queue
my_queue.submit([&](cl::sycl::handler &cgh) {
	....
  // sycl kernel 1
  cgh.parallel_for(....);
});

my_queue.submit([&](cl::sycl::handler &cgh) {
	....
  // sycl kernel 2
  cgh.parallel_for(....);
});

my_queue.submit([&](cl::sycl::handler &cgh) {
	....
  // sycl kernel 3
  cgh.parallel_for(....);
});

...;
