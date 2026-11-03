# Movie Recommender frontend

React 19 + Create React App. See the [root README](../README.md) for the complete Python/React setup.

```sh
npm ci
npm start
npm test -- --watchAll=false --runInBand
npm run build
```

Copy `.env.example` to `.env.local` for an alternate API URL or optional TMDB posters. Posters are fetched independently; the demo works without a key.

- `src/components/RecommendForm.jsx`: discovery, collapsible profile controls and session taste orchestration.
- `MovieCard.jsx`: accessible film actions and nonblocking TMDB artwork.
- `MovieDialog.jsx`: cancellable similarity requests, focus handling and Escape/Tab support.
- `src/lib/api.js`: API client, response normalization, poster caching and tab-session identifier.
- `src/App.test.js`: main user interactions with API calls mocked; backend tests use real model inference.

The frontend keeps film collections in memory and its demo session ID in `sessionStorage`. This is not an authentication system. `FeaturedMovie.jsx` shows a recommended movie backdrop and description. `MovieRail.jsx` provides horizontal recommendation and similar-film rows. No custom application logo is used. Without artwork, cards show a neutral unavailable state.
