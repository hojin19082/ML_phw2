# Programming Homework 2

Auto ML for clustering

Team member 최재민, 권순완, 정호진

## run_cfg(cfg, x, verbose=False, random_state=18, score_only=False)

    Goal : For each model, we find a combination of scaling, encoding, parameters, good subset of features.
    print best silhouette score and best combination for each model.

## How to operate:

    Scaling and encoding the selected features using the scaler and encoder stored in cfg.
    After calculating the silhouette score for each clustering model, the combination with the highest score is output by comparing.
    Visualize each clustering result through tsne.
  
  
### Parameter

    :param cfg: a dataset containing the selected features, scaler, encoder, n_cluster, and parameters of each model.
    :param x: dataset excluding 'median_house_value' feature
    :param verbose: False
    :param random_state: 18
    :param score_only: False
