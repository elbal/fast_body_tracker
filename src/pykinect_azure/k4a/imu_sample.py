import numpy as np

from ._k4a_types import k4a_imu_sample_t


class ImuSample:
    def __init__(self, imu_sample_struct: k4a_imu_sample_t):
        self.temp = float(imu_sample_struct.temperature)
        self.acc_time = int(imu_sample_struct.acc_timestamp_usec)
        self.gyro_time = int(imu_sample_struct.gyro_timestamp_usec)
        self.acc = np.array(imu_sample_struct.acc_sample, dtype=np.float32)
        self.gyro = np.array(
            imu_sample_struct.gyro_sample, dtype=np.float32)
