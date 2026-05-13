import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

class Logger():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []

        self.log['original_foot', 'lfoot', 'pos'] = []
        self.log['original_foot', 'rfoot', 'pos'] = []
        self.video_writer = None

    def log_data(self, desired, current):
        for item in desired.keys():
            for level in desired[item].keys():
                self.log['desired', item, level].append(desired[item][level])
                self.log['current', item, level].append(current[item][level])

    def initialize_plot(self, frequency=1, save_video=False):
        self.frequency = frequency
        self.save_video = save_video
        self.plot_info = [
            {'axis': 0, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue' , 'style': '-' },
            {'axis': 0, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue' , 'style': '--'},
            {'axis': 0, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 0, 'color': 'green', 'style': '-' },
            {'axis': 0, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 0, 'color': 'green', 'style': '--'},
            {'axis': 1, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue' , 'style': '-' },
            {'axis': 1, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue' , 'style': '--'},
            {'axis': 1, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 1, 'color': 'green', 'style': '-' },
            {'axis': 1, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 1, 'color': 'green', 'style': '--'},
            {'axis': 2, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue' , 'style': '-' },
            {'axis': 2, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue' , 'style': '--'},
            {'axis': 2, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 2, 'color': 'green', 'style': '-' },
            {'axis': 2, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 2, 'color': 'green', 'style': '--'},
            
        ]

        plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
        self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))

        self.lines = {}
        for item in self.plot_info:
            key = item['batch'], item['item'], item['level'], item['dim']
            self.lines[key], = self.ax[item['axis']].plot([], [], color=item['color'], linestyle=item['style'])

        # second window
        self.fig_xy, self.ax_xy = plt.subplots(figsize=(7, 7))
        self.fig_xy.canvas.manager.set_window_title('Piano XY (Top-Down)')
        self.line_com_xy, = self.ax_xy.plot([], [], color='red', label='CoM', linewidth=1.5)
        self.line_zmp_xy, = self.ax_xy.plot([], [], color='black', label='ZMP', linewidth=0.8)
        self.line_com_orig_xy, = self.ax_xy.plot([], [], color='blue', linestyle='--', linewidth=1, alpha=0.6, label='CoM Originale')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.set_xlabel('X [m]')
        self.ax_xy.set_ylabel('Y [m]')
        
        custom_lines = [Line2D([0], [0], color='red', lw=1.5),
                        Line2D([0], [0], color='black', lw=0.8),
                        Line2D([0], [0], color='gray', lw=1.2),
                        Line2D([0], [0], color='blue', lw=1, linestyle='--')]
        self.ax_xy.legend(custom_lines, ['CoM', 'ZMP', 'Passi Attuali', 'Passi Originali'])
        self.ax_xy.grid(True, alpha=0.3)
        
        plt.ion()
        plt.show()

        if self.save_video:
            self.fig_xy.canvas.draw()
            
            rgba_buffer = self.fig_xy.canvas.buffer_rgba()
            img_shape = np.asarray(rgba_buffer).shape 
            
            self.video_size = (img_shape[1], img_shape[0]) 

            self.video_writer = cv2.VideoWriter(
                'piano_xy_tracking.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                10,
                self.video_size
            )

    def update_plot(self, time):
        if time % self.frequency != 0:
            return

        for item in self.plot_info:
            trajectory_key = item['batch'], item['item'], item['level']
            trajectory = np.array(self.log[trajectory_key]).T[item['dim']]
            line_key = item['batch'], item['item'], item['level'], item['dim']
            self.lines[line_key].set_data(np.arange(len(trajectory)), trajectory)

        # set limits
        for i in range(len(self.ax)):
            self.ax[i].relim()
            self.ax[i].autoscale_view()
        
        # update second window
        com_data = np.array(self.log['current', 'com', 'pos']).T
        zmp_data = np.array(self.log['current', 'zmp', 'pos']).T
        if com_data.size > 0:
            self.line_com_xy.set_data(com_data[0], com_data[1])
        if zmp_data.size > 0:
            self.line_zmp_xy.set_data(zmp_data[0], zmp_data[1])
        
        
        # update CoM line
        if ('desired', 'com_pure', 'pos') in self.log:
            com_pure_data = np.array(self.log['desired', 'com_pure', 'pos']).T
            if com_pure_data.size > 0:
                self.line_com_orig_xy.set_data(com_pure_data[0], com_pure_data[1])
        

        self.ax_xy.relim()
        self.ax_xy.autoscale_view()
            
        # redraw the plot
        self.fig.canvas.draw()
        self.fig_xy.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig_xy.canvas.flush_events()

        if self.video_writer is not None:
            self.fig_xy.canvas.draw()
            img = np.asarray(self.fig_xy.canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            if (img_bgr.shape[1], img_bgr.shape[0]) != self.video_size:
                img_bgr = cv2.resize(img_bgr, self.video_size)

            self.video_writer.write(img_bgr)

    def close_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Video saved successfully.")