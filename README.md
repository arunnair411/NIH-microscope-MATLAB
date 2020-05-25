# NIH-microscope-MATLAB

Step 1: Run Arun_20191231_beadSimulator.m - This generates the data
Step 2: Run train_val_test_split.m - This splits the generated data into train, val, and test
Step 3: Run calculate_mean_and_stddev.m - This calculates mean and standard deviation across the training set and stores it to a file, before applying the normalization to the entire dataset
Step 4: Run process_experimental_data.m - This applies the normalization to the experimental test data scene
(OR)
Step 4: Run process_experimental_data_2_direct.m - This applies the normalization to the experimental test data itself (doesn't first recover the scene)

**MISCELLANEOUS**
Arun_20191231_plotting.m - Quick and dirty script for plotting experimental data