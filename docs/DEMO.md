# A two-minute demo

Run the app using the root README. This walkthrough uses live model/API results. With `REACT_APP_TMDB_API_KEY` configured, cards use real TMDB posters and the featured film uses a backdrop. Without a key, the explicit artwork-unavailable fallback is used.

1. **Open Discover.** Ten popular films load immediately, without requiring personal details.
2. **Try profile ranking.** Expand **Personalize your picks**, choose a gender, age group and occupation, then select **Find my films**. The source label changes to **Picked for your profile**. This path invokes the real PyTorch checkpoint.
3. **Explore a film.** Select a cover. The dialog shows six FAISS neighbors and their cosine scores. Close using Escape; focus returns to the film button.
4. **Like three or four films.** Genre chips appear in the taste panel. Open **For your taste**. After the session-vector norm crosses the 0.2 threshold, the label changes to **Inspired by your likes**. The exact number of likes needed depends on the embeddings; one like may still use popularity fallback.
5. **Reset.** Select **Reset** to clear the session vector and taste summary. Taste recommendations return to popularity fallback.

The [model card](MODEL_CARD.md) explains evaluation methodology. The saved classification AUC is not a top-K ranking or engagement score.

## Manual presentation check

- Check desktop and a narrow/mobile viewport: forms, film rails and dialog should fit without page-level horizontal scrolling.
- Navigate the form by keyboard, use arrow keys on result tabs, open a film and close with Escape.
- Check the retry state with the backend temporarily unavailable, then restore it.
- Review the [dataset notes](DATA.md) before redistributing dataset files.

Automated interaction tests cover the main flows. They do not certify visual layout or replace a browser check. This repository currently provides a local demo, not a hosted streaming service.
