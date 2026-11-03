# Model card: DeepFM + Lantern

## Purpose and scope

A research/portfolio recommender for exploring the historical MovieLens 1M catalog. It demonstrates taking a notebook model into a working inference and retrieval application. It is not a streaming catalog integration, LLM recommender or production personalization platform.

## Offline data and features

Source of truth: [`Data/Deep_FM_Pytorch.ipynb`](../Data/Deep_FM_Pytorch.ipynb).

- 6,040 users, 3,883 movie records, 1,000,209 explicit ratings in the local files.
- Ratings ≥ 4 define the positive interaction pool. Training uses selected positives and generated negatives rather than all observed low ratings directly.
- Features: gender (F=0, M=1), seven age buckets, 21 occupation codes, movie ID mapped to a contiguous index, and the first listed movie genre encoded in sorted order.
- Gender/age/occupation are dataset codes. They are not inferred from a user's browsing, and all fields can be left unknown in the UI.
- At most 20 positive examples per user; 4 sampled negatives per positive. Negative sampling excludes that user's full positive set, mixing popularity-based and uniform sampling. An unrated film can still be a false negative.
- Popularity-aware item weighting: `1 / (1 + positive_count)^0.6`. Negative sampling uses exponent 0.75 and a 50/50 popularity/uniform mix.

## Architecture and training configuration

Five categorical fields each have a 16-dimensional embedding and a linear term. The model adds the linear logit, pairwise factorization-machine interaction logit and DNN logit. The DNN concatenates field embeddings, then uses layers 128 → 64 → 1, ReLU and dropout 0.2. Sigmoid converts the final logit for ranking.

The notebook uses Adam, learning rate 0.001, batch size 1,024, weighted binary cross entropy and four epochs with seed 42. The checked-in checkpoint has **85,516 parameters**. The item embedding export is **3,883 × 16 float32** and exactly matches `emb.item_id.weight` in that checkpoint, as checked by `scripts/check_artifacts.py`.

## Evaluation provenance

| Metric | Saved test output |
| --- | --- |
| ROC-AUC | 0.7841 |
| Log loss | 0.4156 |
| Accuracy at 0.5 | 0.8094 |

These are existing notebook outputs, not a new evaluation run. Artifact validation verifies the serving files' consistency; it does not certify these metrics or prove which training run produced them.

The notebook stratifies sampled rows by label: 10% test, then 10% of the remaining rows for validation (approximately 81/9/10 train/validation/test). Users can appear in all splits. This setup is appropriate for describing an interaction-classification experiment, with important limits:

- It does not establish unseen-user generalization or performance on future interactions.
- Negative sampling occurs before splitting; repeated user–item pairs can cross splits because negatives are sampled per positive. This can make offline results optimistic.
- Popularity statistics and positive-exclusion sets use the full positive pool before splitting. A stricter experiment would split first and calculate statistics from training data only.
- The dataset is not balanced after four negatives per positive, so accuracy alone is not a useful headline.
- No Recall@K, NDCG@K, coverage, calibration, latency percentile or online A/B result is claimed here.

A stronger future evaluation would use a per-user time split, training-only sampling/statistics, pair-level de-duplication and ranking metrics against a popularity baseline. Retraining and that evaluation are outside this portfolio finishing pass.

## Serving behavior

`POST /recommend` scores all mapped films using one profile. Unknown in any profile field selects the saved popularity list. Full profiles use PyTorch inference mode and cached tensors/model weights.

`GET /similar_items` normalizes a copy of movie embeddings, adds them to a FAISS `IndexFlatIP`, and queries a normalized movie vector. At this catalog size, this is exact inner-product retrieval, not approximate nearest-neighbor search. The query film is excluded.

The UI does not convert model outputs into star ratings or fabricated match percentages. Popularity placeholder scores, sigmoid model scores and cosine similarities have distinct meanings.

## Online taste adaptation

`POST /event` performs three weighted logistic-gradient steps on a session user vector with learning rate 0.2 and L2 regularization 0.001. Positive events point the vector toward the movie embedding; dismiss moves it away. The movie embeddings remain fixed.

Event weights: click 0.5, like 1.0, finish 1.2, dismiss 0.3. The interface exposes likes; other event types are API capabilities. Main-film exploration does not automatically emit click events. Once the vector norm reaches 0.2, `/lantern/recommend` ranks with cosine similarity; weaker or missing vectors use popularity fallback. A single like may still be below the threshold.

Taste summaries count genres of positive events and list recent interactions. They are transparent summaries, not a separate trained persona model. Server state is held in memory; browser session identifiers are demo isolation, not access control.

## Limitations

Historical catalog, a first-genre simplification, original binary gender coding, demographic stereotypes and popularity effects can limit diversity and relevance. The model has no user-ID embedding; two users with the same demographic profile initially receive the same ranking. Lantern can separate them only through their session interactions. This distinction is useful when explaining the system in interviews.
