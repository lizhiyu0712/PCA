dataset_full = readtable('dataset.csv');
% This dataset has 4 accelerometers placed on the shoulder,
% hip, arm, and leg. Each accelerometer has 3 axes, for a total
% of 12 dimensions.
% Explore the dataset and become familiar with it. Click on
% dataset_full in your Workspace and see how it is structured.
% In this dataset, the first 6 columns are what we call
% “metadata”, which are data that provide additional
% information.
% We want to take a subset of the dataset (the entire dataset
% is too large).
x_hat = dataset_full(155750:156250, 7:18);
x = table2array(x_hat);
% This new matrix has 12 dimensions 501 samples.
% We can visualize the 12 dimensions by plotting them.
figure
plot(x)

% Now, we want to reduce the dimensions of this dataset as we
% think there are redundancies in the dataset. For example, we
% know that the movements of the two arms during walking are
% very similar and highly correlated. So let's perform PCA to
% reduce the redundancy, and therefore the dimensionality:
% Step 1: Normalize the data
% First, we calculate the mean and standard deviation:
m_x = mean(x);
s_x = std(x);
% Then, we need to replicate the mean and standard deviations
% so we can use them for every sample (check the outcome):
[num_rows, num_columns] = size(x);
m_x_mat = repmat(m_x, num_rows, 1);
s_x_mat = repmat(s_x, num_rows, 1);
% Then we perform normalization:
x_bar = (x - m_x_mat)./s_x_mat;
figure
plot(x_bar)

% Next, we calculate the covariance matrix of the outcome
cov_x = x_bar' * x_bar;
% Then, we need to calculate the eigenvalues and eigenvectors,
% and sort both the eigenvectors and eigenvalues based on the
% eigenvalues. Luckily, Matlab has a built-in function that
% performs PCA, which we can use directly on “x” instead of all
% these steps:
[PCs, scores, latent, tsquared, explained, mu] = pca(x);
figure
plot(scores)
figure
plot(explained)
figure
plot(cumsum(explained))