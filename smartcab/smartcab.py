# Import the visualization code
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

vs.plot_trials('sim_no-learning.csv')

vs.plot_trials('sim_default-learning.csv')

vs.plot_trials('sim_improved-learning.csv')