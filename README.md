# Learning an approximation of network-based simulations of COVID-19 spread

## Proposal

Network-based methods have proven successful in modeling the spread of the COVID-19 disease, 
as e.g. shown in [1]. However, simulations on large networks tend to be computationally expensive. 
In an attempt to mitigate this, we propose to train a fast surrogate neural network to predict 
the time series of infection/death numbers from the model parameters. This can significantly reduce 
the computational resources needed to simulate the effects of countermeasures, facilitating a more 
comprehensive exploration of their effects under various conditions. Recurrent neural networks [2] 
can be used here and a time series prediction of COVID-19 spread was already conducted on real data in [3], 
using LSTM networks. This approach might be feasible for us as well. If the approach proves infeasible, 
e.g. because of higher computational demands for LSTM training, we can consider [4] as an alternative 
that is presumably more tractable. 

The necessary data can also be obtained by simulation. 
For comparison with real-world time series of COVID-19 the necessary data is publicly available, 
e.g. provided by the ECDC (https://www.ecdc.europa.eu/en/covid-19/data).

Modeling countermeasures against COVID-19 is done in terms of reduction of the average degree in the graph
of the network-based simulation [1], i.e. we need to estimate an aggregated “countermeasure intensity”. 
For simplicity, this can be done manually. If time allows for a more advanced methods, we could attempt 
to estimate the “countermeasure intensity” from detailed COVID policy data, like from the Oxford Covid-19 
Response Tracker (https://github.com/OxCGRT/covid-policy-tracker). 

Generating training data via simulation might pose a computational challenge. In that case  we have to restrict 
ourselves to smaller networks or leverage cloud resources.

We suggest the following roadmap for the project, denoted in bold is the minimal extent of the project. 
Later points are optional and will be tackled as long as our time budget of ~90 hours per person is not exceeded.
- __Implement the network-based simulation given in [1]__
- __Implement LSTM-NN model analogous to [3]__
    - __Fallback: surrogate model as in [4]__
- __Train & evaluate surrogate model__
- Extend network-based simulation with counter-measures
- Retrain & evaluate surrogate model
- Compare predicted case series with real-world COVID-19 data.


## Notes

### Outline

- Introduction
- Related work
- Method
    - Network based simulation
    - LSTM
        - Motivation/Basics
        - Approach for simulated TS prediction
            - input/output
            - training
                - loss?
            - hyperparameter tuning (via GP?)
    - RNN for comparison?
    - Dynamic change of simulation params
        - Varying degree, transmission prob, infect duration
- Results
    - Network based Simulation
        - Variance (to better judge LSTM performance)
            - In repeated runs
            - Sensitivity to initial state
        - Computational Performance
    - LSTM
        - Prediction accuracy
        - Computational performance
- Conclusion

## References

[1] A network-based explanation of why most COVID-19 infection curves are linear, Stefan Thurner, Peter Klimek, Rudolf Hanel, Proceedings of the National Academy of Sciences Sep 2020, 117 (37) 22684-22689; DOI: 10.1073/pnas.2010398117

[2] Michael Hüsken, Peter Stagge, Recurrent neural networks for time series classification, Neurocomputing, Volume 50, 2003, Pages 223-235, ISSN 0925-2312, https://doi.org/10.1016/S0925-2312(01)00706-8.

[3] Vinay Kumar Reddy Chimmula, Lei Zhang, Time series forecasting of COVID-19 transmission in Canada using LSTM networks, Chaos, Solitons & Fractals, Volume 135, 2020, 109864, ISSN 0960-0779, https://doi.org/10.1016/j.chaos.2020.109864.

[4] Melin P, Monica JC, Sanchez D, Castillo O. Multiple Ensemble Neural Network Models with Fuzzy Response Aggregation for Predicting COVID-19 Time Series: The Case of Mexico. Healthcare. 2020; 8(2):181. https://doi.org/10.3390/healthcare8020181
