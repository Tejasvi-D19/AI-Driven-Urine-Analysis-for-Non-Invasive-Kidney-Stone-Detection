# AI-Driven-Urine-Analysis-for-Non-Invasive-Kidney-Stone-Detection

ABSTRACT - 
The urological problem of kidney stones becomes a serious health concern when left untreated that may lead to unbearable pain and important medical complications. Patients often face two issues when using traditional diagnosis methods such as CT scans and ultrasounds: these tests cost a large amount of money and expose people to radiation while also being unavailable in areas that lack resources. The research evaluates a machine learning solution which examines urine test results to identify kidney stones because it aims to develop a cost-effective diagnostic method that operates through non-invasive methods with open accessibility.

Among the machine learning classifiers used in this study, Logistic Regression, Support Vector Machine (SVM), Random Forest and XGBoost were included because they are known to do well in biomedical classification. If your data is straightforward and linear, Logistic Regression is a simple method, but SVM deals well with more complex and unstructured data. Random Forest and XGBoost are types of ensemble models and they can manage both feature interactions and noise. While Random Forest helps more with smooth results and easy interpretation, it is XGBoost that stands out in making accurate predictions by applying gradient boosting.

Analysis of the urine was done using six important factors: specific gravity, pH, osmolality, conductivity, urea and calcium. The 1000 samples were checked for quality, normalized and looked into further to prepare for the model. Including interaction terms and clinical transformations into features improved how the model worked. Part of the data or 80%, was used for training and the remaining 20% was used

for testing the classification model using accuracy, precision, recall, F1-score and AUC-ROC.

Random Forest had the best results, with an accuracy of 98.0%, precision of 0.989, recall of 0.968, F1-score of 0.978 and an AUC-ROC score of 0.986. XGBoost gave satisfactory results, but Random Forest did better at balancing sensitivity and specificity, making it the best option for early detection in the clinic. The results strengthen the idea that AI might help with kidney stone diagnosis without the need for imaging, especially when money is tight or proper care is far away.

By using machine learning with biochemical tests in a thoughtful manner, it is possible to improve detection earlier and to reduce the number of invasive procedures.

Keywords - non-invasive diagnosis,  Random Forest, Logistic Regression, Support Vector Machine, XGBoost, urine analysis, kidney stone detection, machine learning techniques.
