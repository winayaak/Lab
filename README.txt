Logistic Regression Practical (Experiment 02)

Files:
- Student.csv: dataset used by the experiment
- main.py: trains the Logistic Regression model, prints accuracy and a sample prediction, and displays a graph.

Setup:
1. Install Python 3 from https://www.python.org/downloads/
2. Open Command Prompt in this folder.
3. Install dependencies:
   python -m pip install pandas matplotlib numpy scikit-learn
4. Run:
   python main.py

Note:
The last dataset row in the supplied PDF appears to have only four values, although the CSV has five columns. This package uses the 20 complete rows and omits that incomplete row rather than guessing the missing Previous_Marks value.
