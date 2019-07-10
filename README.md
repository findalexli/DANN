# DANN

## PyTorch Modern implementation of DANN
https://www.ncbi.nlm.nih.gov/pubmed/?term=25338716
Anotating genetic variants, especially non-coding variants, for the purpose of identifying pathogenic variants remains a challenge. Combined annotation-dependent depletion (CADD) is an algorithm designed to annotate both coding and non-coding variants, and has been shown to outperform other annotation algorithms. CADD trains a linear kernel support vector machine (SVM) to differentiate evolutionarily derived, likely benign, alleles from simulated, likely deleterious, variants. However, SVMs cannot capture non-linear relationships among the features, which can limit performance. To address this issue, we have developed DANN.

I implemented the DANN with PyTorch, the open source modern framework for deep learning. Achieved slightly higher performance than the origional paper in terms of accuracy. I am currently evaluating other models to significantly outperform the paper. 
