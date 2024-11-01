# COS30018 Intelligent Systems - A2
Completed by:
Michael van der Merwe
Ashraf Toor
Sumaiya Haque

## Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU、RNN).

### Requirement
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19

### Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

Or for more control

```
python gui.py
```

You can choose "lstm", "gru", "rnn" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


### Run the model

**Run command below to run the program:**

```
python main.py
```

Or

```
python gui.py
```

These are the details for the traffic flow prediction experiment.


### Reference

This Traffic Flow Prediction System is based on [xiaochus's TFPS][xiaochus_TFPS_link]

[xiaochus_TFPS_link]: https://github.com/xiaochus/TrafficFlowPrediction


### Copyright
See [LICENSE](LICENSE) for details.
