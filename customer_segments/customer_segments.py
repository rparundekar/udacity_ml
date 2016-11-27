# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# Display a description of the dataset
display(data.describe())

# Select three indices of your choice you wish to sample from the dataset
indices = [57, 207, 422]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy(deep=True)
new_data.drop(['Detergents_Paper'], axis = 1, inplace = True);

# Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, data['Detergents_Paper'], test_size=0.2, random_state=0)

# Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X_train, y_train)
predicted=regressor.predict(X_test)
#Report the score of the prediction using the testing set
from sklearn.metrics import r2_score
score = r2_score(y_test, predicted)
print "R^2 score with regression for selected column:"
display(score)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Scale the data using the natural logarithm
log_data = data.copy()
log_data=np.log(log_data)

# Scale the sample data using the natural logarithm
log_samples = samples.copy()
log_samples = np.log(log_samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Display the log-transformed sample data
display(log_samples)

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    IQR = Q3-Q1
    step = 1.5 * IQR
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# Outliers for Fresh:
#         Fresh       Milk    Grocery    Frozen  Detergents_Paper  Delicatessen
# 65   4.442651   9.950323  10.732651  3.583519         10.095388      7.260523
# 66   2.197225   7.335634   8.911530  5.164786          8.151333      3.295837
# 81   5.389072   9.163249   9.575192  5.645447          8.964184      5.049856
# 95   1.098612   7.979339   8.740657  6.086775          5.407172      6.563856
# 96   3.135494   7.869402   9.001839  4.976734          8.262043      5.379897
# 128  4.941642   9.087834   8.248791  4.955827          6.967909      1.098612
# 171  5.298317  10.160530   9.894245  6.478510          9.079434      8.740337
# 193  5.192957   8.156223   9.917982  6.865891          8.633731      6.501290
# 218  2.890372   8.923191   9.629380  7.158514          8.475746      8.759669
# 304  5.081404   8.917311  10.117510  6.424869          9.374413      7.787382
# 305  5.493061   9.468001   9.088399  6.683361          8.271037      5.351858
# 338  1.098612   5.808142   8.856661  9.655090          2.708050      6.309918
# 353  4.762174   8.742574   9.961898  5.429346          9.069007      7.013016
# 355  5.247024   6.588926   7.606885  5.501258          5.214936      4.844187
# 357  3.610918   7.150701  10.011086  4.919981          8.816853      4.700480
# 412  4.574711   8.190077   9.425452  4.584967          7.996317      4.127134

# Outliers for Milk:
#          Fresh       Milk    Grocery    Frozen  Detergents_Paper  Delicatessen
# 86   10.039983  11.205013  10.377047  6.894670          9.906981      6.805723
# 98    6.220590   4.718499   6.656727  6.796824          4.025352      4.882802
# 154   6.432940   4.007333   4.919981  4.317488          1.945910      2.079442
# 356  10.029503   4.897840   5.384495  8.057377          2.197225      6.306275

# Outliers for Grocery:
#        Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicatessen
#75   9.923192  7.036148  1.098612  8.390949          1.098612      6.882437
#154  6.432940  4.007333  4.919981  4.317488          1.945910      2.079442

# Outliers for Frozen:
#          Fresh      Milk    Grocery     Frozen  Detergents_Paper  Delicatessen
# 38    8.431853  9.663261   9.723703   3.496508          8.847360      6.070738
# 57    8.597297  9.203618   9.257892   3.637586          8.932213      7.156177
# 65    4.442651  9.950323  10.732651   3.583519         10.095388      7.260523
# 145  10.000569  9.034080  10.457143   3.737670          9.440738      8.396155
# 175   7.759187  8.967632   9.382106   3.951244          8.341887      7.436617
# 264   6.978214  9.177714   9.645041   4.110874          8.696176      7.142827
# 325  10.395650  9.728181   9.519735  11.016479          7.148346      8.632128
# 420   8.402007  8.569026   9.490015   3.218876          8.827321      7.239215
# 429   9.060331  7.467371   8.183118   3.850148          4.430817      7.824446
# 439   7.932721  7.437206   7.828038   4.174387          6.167516      3.951244

# Outliers for Detergents_Paper
#         Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicatessen
# 75   9.923192  7.036148  1.098612  8.390949          1.098612      6.882437
# 161  9.428190  6.291569  5.645447  6.995766          1.098612      7.711101

# Outliers for Delicatessen
#       Fresh       Milk    Grocery     Frozen  Detergents_Paper  \
# 66    2.197225   7.335634   8.911530   5.164786          8.151333   
# 109   7.248504   9.724899  10.274568   6.511745          6.728629   
# 128   4.941642   9.087834   8.248791   4.955827          6.967909   
# 137   8.034955   8.997147   9.021840   6.493754          6.580639   
# 142  10.519646   8.875147   9.018332   8.004700          2.995732   
# 154   6.432940   4.007333   4.919981   4.317488          1.945910   
# 183  10.514529  10.690808   9.911952  10.505999          5.476464   
# 184   5.789960   6.822197   8.457443   4.304065          5.811141   
# 187   7.798933   8.987447   9.192075   8.743372          8.148735   
# 203   6.368187   6.529419   7.703459   6.150603          6.860664   
# 233   6.871091   8.513988   8.106515   6.842683          6.013715   
# 285  10.602965   6.461468   8.188689   6.948897          6.077642   
# 289  10.663966   5.655992   6.154858   7.235619          3.465736   
# 343   7.431892   8.848509  10.177932   7.283448          9.646593   

#      Delicatessen  
# 66       3.295837  
# 109      1.098612  
# 128      1.098612  
# 137      3.583519  
# 142      1.098612  
# 154      2.079442  
# 183     10.777768  
# 184      2.397895  
# 187      1.098612  
# 203      2.890372  
# 233      1.945910  
# 285      2.890372  
# 289      3.091042  
# 343      3.610918  

# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [38,57,65,66,75,81,86,95,96,98,109,128,137,142,145,154,161,171,175,183, 184,187,193,203,218,233,264,285,289,304,305,325,338,343,353,355,356,357,412,420,429,439]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(good_data)

# Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a biplot
vs.biplot(good_data, reduced_data, pca)


from sklearn import mixture
from sklearn.metrics import silhouette_score;

for n in range(2,8):
    # Apply your clustering algorithm of choice to the reduced data
    clusterer = mixture.GMM(n_components=n, covariance_type='full')
    clusterer.fit(reduced_data)
    
    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # Find the cluster centers
    centers = clusterer.means_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='euclidean')

    print "The score for cluster of size {} is : {}".format(n, score)

#--------Selecting 2 clusters-------------------
# Apply your clustering algorithm of choice to the reduced data
clusterer = mixture.GMM(n_components=2, covariance_type='full')
clusterer.fit(reduced_data)
    
# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# Find the cluster centers
centers = clusterer.means_

# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)