# **SIMOFF**
SIMulated annealing Objective Function Finder (SIMOFF)
## **Examples**
Example applications are provided for Chinese Hamster Ovary (CHO) cells and Yeast cells. Examples are Jupyter notebooks. Examples can be found in the 'examples' folder.
## **Implementation**
To use SIMOFF with your own data, download the SIMOFF.py file (from 'src' folder) and save this into the same folder as your application notebook. Within the application notebook, type 'import SIMOFF as smf', then smf.simoff() with the correct arguments specified within brackets will allow you to run the SIMOFF function. Ensure all appropriate requirements (requirements.txt) file have been installed and imported into notebook.
SIMOFF has not yet been made into a Python package format. 

By default, SIMOFF stops when the first objective function reaches 100% agreement with the qualitative criteria. To continue the successful annealing run to `max_iter`/`maxfun` and collect all maximum-accuracy solutions, set `stop_at_first_perfect=False` and use the 'SIMOFF_190826.py' version:

```python
result, max_accuracy_df = smf.simoff(
    model,
    input_reactions,
    qualitative_criteria,
    bounds=None,
    max_iter=300,
    initialtemp=50,
    maxfun=5000,
    stop_at_first_perfect=False,
)
```
