## K-Nearest Neighbors
### Euclidean Distance
$$ \sqrt {(x_2^2 - x_1^2) + (y_2^2 - y_1^2)} $$

## Model Evaluation
### Accuracy and Error Rates
Accuracy Rate
$$ AR = \frac{Correct}{Total} = \frac{TN + TP}{Total} $$
where $TN$ and $TP$ are the total true negatives and true positives respectively

Error Rate
$$ ER = \frac{Incorrect}{Total} = \frac{FN + FP}{Total} $$
where $FN$ and $FP$ are the total false negatives and false positives respectively

### Precision and Recall
Precision
$$ \frac{TP}{TP + FP} $$

Recall(Sensitivity)
$$ \frac{TP}{TP + FN} $$

### F1 Score
$$ F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}} = \frac{2 * (Precision * Recall)}{Precision + Recall}$$