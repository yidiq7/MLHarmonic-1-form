import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()

# Read the text file
manifold_list = ['fermat' , 'quintic2', 'cicy1', 'cicy2']

for manifold in manifold_list:
    print('Processing manifold {}'.format(manifold))
    
    if manifold == 'fermat' or manifold == 'quintic2':
        layer = '128_256_1024_10'
    elif manifold == 'cicy1' or manifold == 'cicy2':
        layer = '128_256_1024_15'

    plot_folder = 'plots/{}_{}/'.format(manifold, layer)
    csvfile = 'csv/{}_{}.csv'.format(manifold, layer)
    df = pd.read_csv(csvfile, sep=' ')
    metrics = ['loss', 'average norm', 'max norm', 'min norm', 'normalized max norm', 'normalized min norm']

    window_size = 10
    for metric in metrics:
        df['Smoothed Train {}'.format(metric)] = df['Train {}'.format(metric)].rolling(window=window_size).mean()
        #df['Smoothed Train {} std'.format(metric)] = df['Train {}'.format(metric)].rolling(window=window_size).std()
        df['Smoothed Test {}'.format(metric)] = df['Test {}'.format(metric)].rolling(window=window_size).mean()
        #df['Smoothed Test {} std'.format(metric)] = df['Test {}'.format(metric)].rolling(window=window_size).std()

    for metric in metrics:
        # Function to extract values from content based on the given keyword

        #$plt.plot(epochs, train_values, linestyle='-', linewidth=1, label="Training")
        #plt.plot(epochs, test_values, linestyle='-', linewidth=1, label="Testing")
        sns.lineplot(data=df, x='Epoch', y='Smoothed Train {}'.format(metric), label='Train')
        sns.lineplot(data=df, x='Epoch', y='Smoothed Test {}'.format(metric), label='Test')
        #plt.fill_between(df['Epoch'],
        #                 df['Smoothed Train {}'.format(metric)] + df['Smoothed Train {} std'.format(metric)], 
        #                 df['Smoothed Train {}'.format(metric)] - df['Smoothed Train {} std'.format(metric)], 
        #                 alpha=0.2)

        #plt.fill_between(df['Epoch'],
        #                 df['Smoothed Test {}'.format(metric)] + df['Smoothed Test {} std'.format(metric)], 
        #                 df['Smoothed Test {}'.format(metric)] - df['Smoothed Test {} std'.format(metric)], 
        #                 alpha=0.2)

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} (Log Scale)')
        plt.title(f'{metric.capitalize()} vs Epoch')
        plt.legend()
        #plt.grid(True)

        plot_filename = os.path.join(plot_folder, '{}_log_plot.pdf'.format(metric.replace(' ', '_')))
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        # Clear the plot for the next metric
        plt.clf()


csvfile = 'csv/summary.csv'
df = pd.read_csv(csvfile, sep=' ')
sns.scatterplot(data=df, x='Harmonic loss', y='Normalized min norm', hue='Manifold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Harmonic loss')
plt.ylabel('Normalized min norm')
plt.legend()
plt.savefig('plots/summary.pdf', format='pdf', bbox_inches='tight')
