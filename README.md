# Evaluating Feature Attribution Methods
This GitHub contains the code from a 
<a href="https://www.studiegids.tudelft.nl/a101_displayCourse.do?course_id=61505" target="_blank">Bachelor thesis project</a>
for the TU Delft on evaluating three feature attribution methods in the context of fact-checking.
In the `src` folder, the code for running the feature attribution methods and the code for evaluating these methods can be found.
The `data` folder contains the results for 100 instances of the `test` and `train` set from the total dataset, 
and in the `colorize` folder, there are .html files with heatmaps on these instances.
Lastly, there are figures from the evaluation in the `figures` folder.


- [How to run the feature attribution methods and evaluation](#how-to-run-the-feature-attribution-methods-and-evaluation)
- [Using the code with other models](#using-the-code-with-other-models)

## How to run the feature attribution methods and evaluation
When one wants to run the code, first, a trained fact-checking model is needed. 
Currently, in the code, a trained version of the implementation from the paper 
<a href="https://dl.acm.org/doi/abs/10.1145/3437963.3441758" target="_blank">Explain and Predict, and then Predict Again (ExPred)</a> 
is used. 
<a href="https://dl.acm.org/doi/abs/10.1145/3437963.3441758" target="_blank">ExPred</a> 
is an NLP architecture that can be used for fact-checking.
The dataset used is a version of <a href="https://fever.ai" target="_blank">FEVER</a>, 
which comes from <a href="http://www.eraserbenchmark.com" target="_blank">Eraser Benchmark</a>. 
An <a href="https://github.com/JoshuaGhost/expred" target="_blank">untrained version of ExPred</a> can be found on GitHub.

### Running the code with ExPred:
<details><summary><b>Expand the instructions</b></summary>

1. Add a trained version of ExPred.

2. Setting up the Python virtual environment:
```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Update all `/dir/to/your/` lines to the where your data can be found and the results can be stored.

5. Run an instance by scrolling down in the feature attribution methods file and choosing the number of iterations and the instance.<br>
**OR**<br>
Run multiple instances by scrolling down in the feature attribution methods file, and choosing 
the number of iterations and changing the instances number to `int(sys.argv[1])`. Go to the run.py file, and run the amount of instances you want.
</details>

### A list of the technologies used:
* <a href="https://captum.ai" target="_blank">Captum</a>
* <a href="https://matplotlib.org" target="_blank">Matplotlib</a>
* <a href="https://numpy.org" target="_blank">NumPy</a>
* <a href="https://pytorch.org" target="_blank">PyTorch</a>
* <a href="https://scipy.org" target="_blank">SciPy</a>
* <a href="https://pypi.org/project/tabulate/" target="_blank">tabulate</a>

We used Python version 3.8.


## Using the code with other models
The code is written for the combination of a claim and context; 
if someone wants to use a different input, adaptations are needed in the feature attribution files. 
These adaptations would take quite some time because the `wrapper` methods need to be changed. 
These methods were implemented to make sure 
<a href="https://captum.ai" target="_blank">Captum</a> and the chosen model worked well with each other.
See the <a href="https://captum.ai/tutorials/" target="_blank">tutorials from Captum</a> for examples.