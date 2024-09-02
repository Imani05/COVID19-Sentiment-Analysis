# COVID19-Sentiment-Analysis

This research explores the optimization of transfer learning strategies for sentiment analysis, focusing on COVID-19 related tweets. The study comparatively analyses the performance of BERT and RoBERTa models against a Naive Bayes baseline, addressing the unique challenges posed by pandemic-related social media content. The research aims to enhance our understanding of public sentiment during a global health crisis and improve the efficacy of sentiment analysis techniques in this context.

## Datasets
The study utilizes two datasets, `Corona_train.csv` and `Corona_test.csv`, comprising COVID-19 related tweets. A comprehensive preprocessing pipeline is implemented, including emoji removal, hashtag cleaning, and special character filtering.

## Methodology
The BERT and RoBERTa models are fine-tuned using a maximum sequence length of 128 tokens, with model-specific tokenization strategies.

## Results
Results demonstrate the superiority of transformer-based models over the Naive Bayes baseline. BERT and RoBERTa achieved overall accuracies of 90% and 90.36% respectively, significantly outperforming Naive Bayes (71%). Both models exhibited high performance in distinguishing between positive and negative sentiments, with F1-scores exceeding 0.91. However, the neutral class proved challenging for all models, highlighting the complexity of capturing neutrality in social media content.

## Model Convergence
The study reveals interesting differences in model convergence patterns, with BERT showing faster initial convergence but potential overfitting in later epochs, while RoBERTa demonstrated more gradual but steady improvement. These findings contribute to our understanding of the trade-offs between different transfer learning approaches in sentiment analysis.

## Broader Implications
The research also addresses broader implications, including the potential application of these models in crisis communication and public health monitoring. It highlights the importance of computational efficiency and ethical considerations in deploying such models in real-world scenarios.

## Limitations and Future Work
While the study provides valuable insights, it acknowledges limitations such as dataset constraints, language limitations, and the need for more fine-grained sentiment analysis. Future research directions are suggested, including the development of multilingual models, exploration of model interpretability techniques, and investigation of real-time sentiment analysis systems for crisis management.

## Conclusion
This study contributes to the growing body of research on advanced NLP techniques for sentiment analysis in the context of public health crises, offering insights that can inform both academic understanding and practical applications in crisis communication and sentiment monitoring.
