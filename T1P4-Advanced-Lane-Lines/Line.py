import numpy as np
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, moving_average=4):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #moving average, number of samples to keep
        self.moving_average = moving_average
        
    def add(self, fit):
        if fit is not None:
            # reset average, for every new sliding window
            if self.detected == False:
                self.detected = True
                self.current_fit = []
            self.current_fit.append(fit)
            # keep the recent
            if len(self.current_fit) > self.moving_average:
                self.current_fit = self.current_fit[len(self.current_fit) - self.moving_average:]
            self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False
