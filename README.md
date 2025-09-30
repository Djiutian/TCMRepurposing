# TCMRepurposing
The code of paper Semantic Repurposing Model for Traditional Chinese Ancient Formulas Based on Knowledge Graph 

## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+
  
## Running the code 
To reproduce the results , run the following commands.

### Formula-to-Formula

python FormulaToFormula.py  

#### Changing the Formula Name

To use a different formula, open the code and go to line 66:

```python
input_prescription = '复方丹参片'
```


Replace '复方丹参片' with the name of the formula you want to use.

### Symptom-to-Formula
python SymptomToFormula.py
#### Changing the Symptoms

To use a different symptoms, open the code and go to line 66:

```python
input_prescription = '复方丹参片'
```

