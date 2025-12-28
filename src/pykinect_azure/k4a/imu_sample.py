import numpy as np

import _k4a


class ImuSample:
	def __init__(self, imu_sample_struct: _k4a.k4a_imu_sample_t):
		self.temp = float(imu_sample_struct.temperature)
		self.acc_time = int(imu_sample_struct.acc_timestamp_usec)
		self.gyro_time = int(imu_sample_struct.gyro_timestamp_usec)
		self.acc = np.array(imu_sample_struct.acc_sample.v, dtype=np.float32)
		self.gyro = np.array(
			imu_sample_struct.gyro_sample.v, dtype=np.float32)
