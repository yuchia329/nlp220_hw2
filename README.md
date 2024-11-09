 # one_vs_rest_nlp220
 Run the below code to reproduce results:
 ```
 python run.py {data_file_path} True
 ```

It finds results of SVM, decision tree, and logistic regression models with three selected features. It also plots precision-recall and ROC curves for OneVsRest classifier.

Feel free to add or remove any feature-model combination in run.py as you wish

To reproduce hyper-parameter search, use the following code:
```
python search_hyper_param.py {data_file_path} {model_name} {feature#}
```
model_name = {"svm"|"logistic_regression"|"decision_tree"} \
and feature = {1|2|3}