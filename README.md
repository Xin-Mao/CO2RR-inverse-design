# CO2RR-inverse-design

The code used in the work “Inverse Design of Promising Alloys for Electrocatalytic CO<sub>2</sub> Reduction via Generative Graph Neural Networks Combined with Bird Swarm Algorithm”

## Description

This is part of the Text-to-Materials project, aimed at the inverse design of novel CO<sub>2</sub> reduction electrocatalysts.

## Getting Started

### Installing

* Clone this respostory
```
git clone https://github.com/szl666/CO2RR-inverse-design.git
cd CO2RR-inverse-design
```
* Install dependencies 
```
pip install -r requirements.txt
```

### Executing program

* To begin the inverse design process, ensure that the following model files are placed in the appropriate directories:

* Surface generation model: Place the trained CDVAE model in the slab_generation_model folder.
* Adsorption energy prediction model: Place the model files for CO and H adsorption energy predictions in the adsorption_predictor_model folder.

Once the models are in place, run the following code to start the inverse design:
```
python bsa_opt.py
```




