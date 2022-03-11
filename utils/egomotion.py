import numpy as np
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import os


class Egomotion:

    def __init__(self, odom_csv):

        assert os.path.exists(odom_csv), f'{odom_csv} does not exist'
        self.timestamps = np.genfromtxt(odom_csv, dtype=np.int64, usecols=0, delimiter=',', skip_header=True)
        # (N , 2) [yaw_rate, v_ego]
        self.rates = np.genfromtxt(odom_csv, dtype=np.float64, usecols=(1, 2), delimiter=',', skip_header=True)
        self.egomotion_rates = scipy.interpolate.interp1d(self.timestamps, self.rates, axis=0, bounds_error=True)

    def get_yaw_rate(self, timestamp):
        if timestamp < self.timestamps[0]:
            print("Timestamp not in LUT -> Returning 0 yaw rate")
            return 0.0
        elif timestamp > self.timestamps[-1]:
            print("Timestamp not in LUT -> Returning 0 yaw rate")
            return 0.0
        else:
            return self.egomotion_rates(timestamp)[0]

    def get_velocity(self, timestamp):
        if timestamp < self.timestamps[0]:
            print("Timestamp not in LUT -> Returning 0 velocity")
            return 0.0
        elif timestamp > self.timestamps[-1]:
            print("Timestamp not in LUT -> Returning 0 velocity")
            return 0.0
        else:
            return self.egomotion_rates(timestamp)[1]

    def integrate_odometry(self, tmax_idx=-1):

        # Recalculate LUT with relative time in seconds
        time = np.float64((self.timestamps - self.timestamps[0]) / 1e9)
        assert time.shape[0] == self.rates.shape[0]
        egomotion_rates = scipy.interpolate.interp1d(time, self.rates, axis=0, bounds_error=True)

        def motion_model(t, z):
            """
            z_dot = [v*cos(yaw), v*sin(yaw), yaw_dot]
            """

            rates = egomotion_rates(t)
            yaw_rate_ego, v_ego = rates[0], rates[1]
            z_dot = np.asarray([v_ego * np.cos(z[2]),
                                v_ego * np.sin(z[2]),
                                yaw_rate_ego])
            return z_dot

        sol = scipy.integrate.solve_ivp(motion_model, [time[0], time[tmax_idx]], [0, 0, 0], max_step=0.1)
        return sol.t, sol.y.T, egomotion_rates(sol.t)

    def plot_odometry(self):
        t, z, rates = self.integrate_odometry()
        fig = plt.figure()

        ax_yaw = fig.add_subplot(321)
        ax_yaw.plot(t, rates[:, 0])
        ax_yaw.set_xlabel('t [s]')
        ax_yaw.set_ylabel('yaw rate [rad/s]')

        ax_v = fig.add_subplot(322)
        ax_v.plot(t, rates[:, 1])
        ax_yaw.set_xlabel('t [s]')
        ax_v.set_ylabel('v ego [m/s]')

        ax_x = fig.add_subplot(323)
        ax_x.plot(t, z[:, 0])
        ax_yaw.set_xlabel('t [s]')
        ax_x.set_ylabel('x [m]')

        ax_y = fig.add_subplot(324)
        ax_y.plot(t, z[:, 1])
        ax_yaw.set_xlabel('t [s]')
        ax_y.set_ylabel('y [m]')

        ax_traj = fig.add_subplot(3, 2, (5, 6))
        ax_traj.plot(z[:, 0], z[:, 1])
        ax_traj.set_ylabel('y [m]')
        ax_traj.set_xlabel('x [m]')
        idx_arrow = np.linspace(0, t.shape[0] - 2, 10, dtype=np.int32)
        arrow_dir = 1*np.diff(z, axis=0)
        ax_traj.quiver(z[idx_arrow, 0], z[idx_arrow, 1], arrow_dir[idx_arrow, 0], arrow_dir[idx_arrow, 1], units='dots', angles='xy', color='blue')

        plt.show()


if __name__ == "__main__":
    path = '/lhome/dscheub/ObjectDetection/data/external/ImmendingenSprayTestsv3/2021-07-26_19-15-15/odometry.csv'
    ego = Egomotion(path)
    t, z, rates = ego.integrate_odometry()
    print(z.shape)
    print(rates.shape)
    ego.plot_odometry()
    print(ego.get_velocity(ego.timestamps[70] + 45))
    print(ego.get_yaw_rate(ego.timestamps[70] + 45))
