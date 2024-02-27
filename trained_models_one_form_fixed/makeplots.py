import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Read the text file
manifold_list = ['fermat' , 'quintic2', 'cicy1', 'cicy2']

for manifold in manifold_list:
    print('Processing manifold {}'.format(manifold))
    
    if manifold == 'fermat' or manifold == 'quintic2':
        layer = '128_256_1024_10'
    elif manifold == 'cicy1' or manifold == 'cicy2':
        layer = '128_256_1024_15'

    plot_folder = 'plots/' + manifold + '_' + layer
    os.makedirs(plot_folder, exist_ok=True)

    model = manifold + '/' + layer
    filename = model + '_history.txt'
    with open(filename, 'r') as file:
        content = file.read()

    train_metrics = ['loss', 'avg_norm', 'max_norm', 'min_norm']
    test_metrics = ['test loss', 'test avg_norm', 'test max_norm', 'test min_norm']

    def extract_values(keyword):
        # Ignored keyword proceded by 'test ', so that when searching for train metrics,
        # the test metrics will be ingorned.
        values = re.findall(rf'(?<!test ){keyword}:\s+tf\.Tensor\(([\d\.e-]+)', content)
        return [float(value) for value in values]

    for (train_metric, test_metric) in zip(train_metrics, test_metrics):
        # Function to extract values from content based on the given keyword

        train_values = extract_values(train_metric)
        test_values= extract_values(test_metric)
        epochs = [(i + 1) * 10 for i in range(len(train_values))]

        plt.plot(epochs, train_values, linestyle='-', linewidth=1, label="Training")
        plt.plot(epochs, test_values, linestyle='-', linewidth=1, label="Testing")
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel(f'{train_metric.capitalize()} (Log Scale)')
        plt.title(f'{train_metric.capitalize()} vs Epoch')
        plt.legend()
        plt.grid(True)

        plot_filename = os.path.join(plot_folder, f'{train_metric}_log_plot.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight') 
        # Clear the plot for the next metric
        plt.clf()
