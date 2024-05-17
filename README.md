HOW TO USE


1. To train a new model, define a new Simulation object in main_final.py and use .simulate(), as in example:

simulation = Simulation(2014, 2024, 'save_path/', assets=some_asset_list, rebalance=4, batch_size=64, t_cost=0.0005)
    simulation.simulate()

This t_cost refers to transaction cost used in model optimisation.
Before starting, in Model.py in model.compile(...) check what is the current loss function. Set sharpe_loss for Sharpe Ratio optimisation and soft_target_simple for minimising volatility.


2. Currently, simulate uses data saved in csv files. Whenever you want to use newly downloaded data from Yahoo instead of saved copy, swap load_data function call
   with download_and_process_data. All arguments can stay the same. 


3. To generate long-only portfolio, in Model.py definition of build_model method uncomment:
	outp = Softmax()(l1)
   and comment out:
	outp = Lambda(softmax_mod(l1))

4. To generate trading results, use one of the results scripts.
Notes:
a) Whenever calling load_data or download_and_process_data in results scripts, arguments test_start and test_end are not used, so they are only placeholders. Put the data range you want
for testing in arguments train_start and train_end.

b) If you use Mean-Variance benchmark, load/download the data one year before the test start to be able to calculate historical returns/covariance. The data that is passed to LSTM models
   is limited in the following lines:

dates = learn_dates[learn_dates >= pd.to_datetime('2014-01-01')]
price_data = price_data_learn[learn_dates >= pd.to_datetime('2014-01-01')]
model_data = model_data[learn_dates >= pd.to_datetime('2014-01-01')]


c) The function to load LSTM models generally looks like this:

model = keras.saving.load_model('NAME/{}-{}/model.keras'.format(year, month(r, rebalances)),
                                        safe_mode=False, custom_objects={'soft_target_simple': soft_target_simple, 'softmax_mod': softmax_mod})

where NAME is a name of folder in which models from this training are saved (for example SHR base, SHR batch 96 etc.). When loading a model, you have to pass
the loss function and new activation function as custom objects, so for minimum volatility models you pass custom_objects arguments as above, and for Sharpe Ratio models: 

custom_objects={'sharpe_loss': sharpe_loss, 'softmax_mod': softmax_mod}


d) If you want to change transaction cost that is applied in trading, pass it as a t_cost argument whenever calling portfolio_value_calc()

e) If you want to test how the model performs with less frequent recalculations, before entering for year in(...) loop to calculate model's performance, put
	rebalances = X

I believe X = 1, 2 or 4 (default) should work.


f) If you want to change list of assets the results are calculated on, you have to make 3 changes:

i) in load_data/download_and_process_data pass the length of the asset list as n_comp argument
ii) also there, as stock_lists pass your asset tickers list INSIDE SQUARE BRACKETS!!! (formally, as a single-element list, with that element being the tickers list)
iii) Change the weights in equally weighted portfolio by dividing by length of the tickers list.

Obviously, you also have to change the LSTM models that are being loaded.



RUN TIME
=====================================
1. MODEL TRAINING: with 4 recalculations per year and 10 years of testing my run time was about 2 - 2.5 hours to calculate all 40 models.
2. RESULTS GENERATION: running results.py as it is right now should give results in about 5 minutes.
