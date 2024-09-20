import matplotlib.pyplot as plt
from IPython.display import clear_output, display


class LossPlot():

    def __init__(self):
        self.losses = {
            'training': {"iter":[], "value":[]}, 
            'validation': {"iter":[], "value":[]}}
        
        # Set up the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylabel('Loss')
        self.ax.set_xlabel('Iterations')
        self.training_line, = self.ax.plot([], [], label='Training Loss')
        self.validation_line, = self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()

    def add_loss(self, value, iteration, loss_type='training'):
        assert loss_type in ['training', 'validation'],\
            'Loss type unknown, must be training or validation.'

        self.losses[loss_type]['iter'].append(iteration)
        self.losses[loss_type]['value'].append(value)

    def get_current_iteration(self, loss_type='training'):
        return self.losses[loss_type]['iter'][-1]

    def update_plot_ipynb(self):

        for key, line in zip(self.losses, [self.training_line, self.validation_line]):
            line.set_xdata(self.losses[key]['iter'])
            line.set_ydata(self.losses[key]['value'])

        self.ax.relim()
        self.ax.autoscale_view()

        clear_output(wait=True)
        display(self.fig)