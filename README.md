# Acoustic-Emission
***
 Acoustic-Emission is a library to analyse and process acoustic emission and NC4 wear
measurement data.  
Making use of traditional AE processing techniques as well as machine learning.

## Library Layout
***
1) [`main.py`](main.py)
2) [resources](resources)
   - [`experiment.py`](resources/experiment.py)
   - [`ae.py`](resources/ae.py)
   - [`nc4.py`](resources/nc4.py)
   - [`ml_regression.py`](resources/ml_regression.py)
3) [reference](reference)
4) [figures](Figures)


## Usage
***

### [main.py](main.py) 
`main.py` contains the main code to use the library and any functions that aren't finished.

### [experiment.py](resources/experiment.py)
`experiment.py` contains the code to create Experiment classes from the test data folder.  
It also includes functions to load and save the class, as well as generate correlation plots.

### [ae.py](resources/ae.py)
`ae.py` contains the code to create AE classes from the AE TDMS files stored in the data folder.  
It also includes methods to process the AE data into useful features and plot graphs of the data.

### [nc4.py](resources/nc4.py)
`nc4.py` contains the code to create NC4 classes from the NC4 TDMS files stored in the data folder.  
It also includes methods to process the NC4 data into radius data and extract useful wear features.

### [ml_regression.py](resources/ml_regression.py)
`ml_regression.py` contains the code to generate a Keras ANN within an ML pipeline using SciKeras.  
It also includes methods to evaluate models with cross-validation, and conduct gridsearch for  
optimising hyperparameter selection.

### [reference](reference)
`referenc`e is a folder containing reference files for some fo the code to run properly.

### [figures](Figures)
`Figures` is a folder to store graphs and figures produced with the library.

## Useful Code Examples
***

### 1) Creating an Experiment object
```python
import resources
exp = resources.experiment.create_obj()
```
### 2) Loading and Saving an Experiment object
```python
import resources
exp = resources.load('Test 5')
exp.save()
```
### 3) Creating NN model and evaluate
```python
import resources
# Load in experiment obj
exp = resources.load('Test 5')
# Select features for model to use
df = exp.features.drop(columns=['Runout', 'Form error'])
# Create pipeline containing standard scaler and keras regressor
pipe = resources.create_pipeline(
    model=resources.get_regression,
    model__init_mode='glorot_normal',
    model__dropout=0.1,
    model__hidden_layer_sizes=(32, 32),
    optimizer='adam',
    optimizer__learning_rate=0.001,
    loss='mae',
    metrics=['MAE', 'MSE'],
    batch_size=10,
    epochs=700,
    verbose=0,
 )
# Separate and split dataset
X_train, X_test, y_train, y_test = resources.split_dataset(df)
# Score the model whilst training using cross validation
pipe, train_scores = resources.score_train(model=pipe, Xdata=X_train, ydata=y_train)
# Re-fit the model to the training data
pipe.fit(X_train, y_train, reg__validation_split=0.2)
# Plot the models learning during training
resources.train_history(pipe)
# Evaluate the models score with the test data
test_score = resources.score_test(pipe, X_test, y_test)
```
### 4) GridsearchCV for hyper-parameters
```python
# Create model pipeline same as previous example
# Create parameter grid for hyper-parameters to evaluate across
param_grid = dict(
   model__init_mode=['lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
   model__hidden_layer_sizes=[(80,), (30, 25), (30, 30)],
   model__dropout=[0, 0.1, 0.3, 0.5],
   loss=['mse', 'mae'],
   batch_size=[5, 8, 10, 15, 25, 32],
   reg__epochs=np.arange(200, 1025, 25)
   optimizer=['adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta'],
   optimizer__learning_rate=[0.0005, 0.0075, 0.001, 0.0025, 0.005, 0.01],
   )

# GridsearchCV with the created model and param_grid
pipe, grid_result = model_gridsearch(
   model=pipe,
   Xdata=X_train,
   ydata=y_train,
   para_grid=param_grid,
   cv=10,
)
# Plot results of grid search based on one hyper-parameter
plot_grid_results(grid_result, 'epochs')
```
### 5) Process data from Experiment object
```python
# Process the AE data calculating the features and FFT in between the trigger points
exp.ae.process(trigger=True, FFT=True)

# Process the NC4 data converting to radius and calculating wear features
exp.nc4.process()
```
### 6) Plot AE data
```python
# Plot the AE signal found in file 10
exp.ae.plotAE(fno=10)

# Plot the FFT of the AE signal found in file 10
exp.ae.plotfft(fno=10, freqres=1000)
```
### 7) Plot FFT surface of AE data
```python
# Plot the fft surface of each AE signal with resolution of 1kHz and show between 0 - 1MHz
exp.ae.fftsurf(freqres=1000, freqlim=[0, 1_000_000])
```
### 8) Plot NC4 data
```python
# Plot NC4 attributes for each measurement
exp.nc4.plot_att()
# Plot NC4 radius XY plot between files 0 - 100
exp.nc4.plot_xy(fno=[0, 100])
# Plot NC4 radius surface of every measuremnt
exp.nc4.plot_surf()
```
### Scores - Test Data

| MAE | MSE | R2  |
| --- | --- |-----|
| 0.5 | 0.9 | 1   |
