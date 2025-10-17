---
license: apache-2.0
task_categories:
- time-series-forecasting
tags:
- timeseries
- forecasting
- benchmark
- gifteval
size_categories:
- 100K<n<1M

---

## GIFT-Eval
<!-- Provide a quick summary of the dataset. -->
![gift eval main figure](gifteval.png)

We present GIFT-Eval, a benchmark designed to advance zero-shot time series forecasting by facilitating evaluation across diverse datasets. GIFT-Eval includes 23 datasets covering 144,000 time series and 177 million data points, with data spanning seven domains, 10 frequencies, and a range of forecast lengths. This benchmark aims to set a new standard, guiding future innovations in time series foundation models.

To facilitate the effective pretraining and evaluation of foundation models, we also provide a non-leaking pretraining dataset --> [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain).

[📄 Paper](https://arxiv.org/abs/2410.10393)

[🖥️ Code](https://github.com/SalesforceAIResearch/gift-eval)

[📔 Blog Post]()

[🏎️ Leader Board](https://huggingface.co/spaces/Salesforce/GIFT-Eval)

## Submitting your results

If you want to submit your own results to our leaderborad please follow the instructions detailed in our [github repository](https://github.com/SalesforceAIResearch/gift-eval)

## Ethical Considerations

This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP. 

## Citation

If you find this benchmark useful, please consider citing:

```
@article{aksu2024giftevalbenchmarkgeneraltime,
      title={GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation}, 
      author={Taha Aksu and Gerald Woo and Juncheng Liu and Xu Liu and Chenghao Liu and Silvio Savarese and Caiming Xiong and Doyen Sahoo},
      journal = {arxiv preprint arxiv:2410.10393},
      year={2024},
}
```