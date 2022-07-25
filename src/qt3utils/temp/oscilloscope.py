import time
import argparse

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nidaqmx

import qt3utils.nidaq

parser = argparse.ArgumentParser(description='Oscilloscope for the SPCM.')
parser.add_argument('--daq',default = 'Dev1', type=str,
                    help='NI DAQ Device Name')
parser.add_argument('--clock-terminal', default = None, type=str,
                    help='Clock Terminal. If None (default) uses internal NI DAQ clock')
parser.add_argument('--clock-rate', default = 10000, type=int,
                    help='In Hz. Only used when using internal clock')
parser.add_argument('--data-samples', default = 100, type=int,
                    help='Number of data points to acquire per daq request.')
#this code should be moved to an 'executable' directory and
#entry points in setup.py should be defined (perhaps: qt3utils-scope) for easy launch

#these should become command-line arguments
#Configure the NI DAQ
device_name = 'Dev1'
read_write_timeout = 10
clock_terminal = None
edge_input_terminal = 'PFI12'
edge_input_counter = 'ctr2'
clock_rate = 10000 #Hz
N_data_samples_to_acquire = 100
daq_time = N_data_samples_to_acquire / clock_rate


class Scope:
    def __init__(self, ax, maxt=1000, dt=1):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = []
        self.ydata = []
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        if len(self.tdata) > 0:
            lastt = self.tdata[-1]
            if lastt > self.tdata[0] + self.maxt:  # reset the arrays
                self.tdata = [self.tdata[-1]]
                self.ydata = [self.ydata[-1]]
                self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
                self.ax.figure.canvas.draw()

        if len(self.tdata) > 0:
            t = self.tdata[-1] + self.dt
        else:
            t = self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.ax.set_ylim(np.min(self.ydata)*.95, 1.1*np.max(self.ydata))

        self.line.set_data(self.tdata, self.ydata )
        return self.line,


def read_daq_buffer(counter_task, detector_reader,  N_samples,  read_write_timeout=10):

    data_buffer = np.zeros(N_samples)
    counter_task.start()

    samples_read = detector_reader.read_many_sample_double(
                            data_buffer,
                            number_of_samples_per_channel=N_samples,
                            timeout=read_write_timeout)

    counter_task.stop()
    return data_buffer, samples_read


def run():
    print('configuring tasks')

    nidaq_config = qt3utils.nidaq.EdgeCounter(device_name)
    nidaq_config.reset_daq()

    if clock_terminal is None:
        nidaq_config.configure_di_clock(clock_rate = clock_rate)

        nidaq_config.configure_counter_period_measure(
            source_terminal = edge_input_terminal,
            N_samples_to_acquire_or_buffer_size = N_data_samples_to_acquire,
            clock_terminal = nidaq_config.clock_task_config['clock_terminal'],
            trigger_terminal = None,
            sampling_mode = nidaqmx.constants.AcquisitionType.FINITE)
    else:
        nidaq_config.configure_counter_period_measure(
            source_terminal = edge_input_terminal,
            N_samples_to_acquire_or_buffer_size = N_data_samples_to_acquire,
            clock_terminal = args.clock_terminal,
            trigger_terminal = None,
            sampling_mode = nidaqmx.constants.AcquisitionType.FINITE)

    nidaq_config.create_counter_reader()


    fig, ax = plt.subplots()
    scope = Scope(ax)

    # pass a generator in "emitter" to produce data for the update func
    print('starting clock')
    nidaq_config.clock_task.start()

    def emitter():
        """return counts per second """
        while True:
            #data_sample = run_once(edge_detector_task, edge_detector_reader)
            data_sample, samples_read = read_daq_buffer(nidaq_config.counter_task, nidaq_config.counter_reader, N_data_samples_to_acquire)
            yield data_sample.sum()/(samples_read / clock_rate)


    ani = animation.FuncAnimation(fig, scope.update, emitter, interval=20,
                                  blit=True)

    plt.show()


    #clean up
    print('cleaning up')
    nidaq_config.clock_task.stop()
    nidaq_config.clock_task.close()
    nidaq_config.counter_task.stop()
    nidaq_config.counter_task.close()

if __name__ == '__main__':
    run()
