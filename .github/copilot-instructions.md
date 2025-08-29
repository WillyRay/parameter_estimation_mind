Your task it to train a machine learning model using the sim_data.csv.  

Each row of data represents a different day (tick) in a different model iteration.

Each run is a different iteration.  each run-tick combination is a day in the iteration.  The observed data is of considerably shorter length than the simulated data.  We'll need to truncate the simulated data to match the length of the observed data.

The features are not single values per row.  Each record in the training set is comprised of the time series represented day 100-156 in each of the individual runs.  So, we will need to create a new dataset that captures this time series information for each run.

I think the features should be "run", then "count", "CDIFF", "occupancy" and "anyCP" should be time series.  The target variables will be "decayRate" and "surfaceTransferFraction", but there is only one of each for each run, irregardless of tick.

The resulting dataset will have a structure where each row corresponds to a specific run, and the columns include the aggregated time series data for the features, as well as the target variables.    
