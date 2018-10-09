....
int num_gpu;
// finding available number of NVIDIA GPU devices
cudaGetDeviceCount(&num_gpu);

//looping over number of devices and dispatching a kernel per device.
for (int i = 0; i < ngpus; i++) {
	// selecting the current device
	cudaSetDevice(i);
	// executing a my_kernel on the selected device
	my_kernel<<<num_blocks, block_size>>>(...);
	// transfering data between the host and the selected device
	cudaMemcpy(...);
}
....
