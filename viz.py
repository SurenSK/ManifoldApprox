import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('st_cos_sim.csv')

# Create a heatmap using the DataFrame
plt.imshow(df, cmap='hot', interpolation='nearest')
plt.colorbar()

# Set the x and y axis labels
plt.xlabel('Columns')
plt.ylabel('Rows')

# Show the heatmap
plt.show()