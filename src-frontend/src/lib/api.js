import axios from "axios";

export const api = axios.create({
  baseURL: (process.env.REACT_APP_API_URL || "http://localhost:5000").replace(
    /\/$/,
    "",
  ),
  timeout: 15000,
});

export function getSessionId() {
  try {
    let id = sessionStorage.getItem("lantern-session");
    if (!id) {
      id = `demo-${crypto.randomUUID()}`;
      sessionStorage.setItem("lantern-session", id);
    }
    return id;
  } catch {
    return `demo-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
}

export function normalizeMovies(items = []) {
  return items.map((item) =>
    Array.isArray(item)
      ? { item_id: item[0], title: item[1], score: item[2] }
      : item,
  );
}

function searchTitle(title) {
  return title
    .replace(/\s*\(\d{4}\)$/, "")
    .replace(/^(.*), (The|A|An)$/, "$2 $1");
}

// Cards, featured backdrops and dialogs share one lookup per title.
const artworkCache = new Map();
export function getArtwork(title) {
  const key = process.env.REACT_APP_TMDB_API_KEY;
  if (!key || !title) return Promise.resolve(null);
  if (!artworkCache.has(title)) {
    const year = title.match(/\((\d{4})\)$/)?.[1];
    const query = searchTitle(title);
    artworkCache.set(
      title,
      axios
        .get("https://api.themoviedb.org/3/search/movie", {
          params: {
            api_key: key,
            query,
            ...(year ? { primary_release_year: year } : {}),
          },
          timeout: 5000,
        })
        .then(({ data }) => {
          const candidates = data.results || [];
          const normalize = (value) =>
            value?.toLowerCase().replace(/[^a-z0-9]/g, "");
          const match =
            candidates.find(
              (item) =>
                [item.title, item.original_title].some(
                  (name) => normalize(name) === normalize(query),
                ) &&
                (!year || item.release_date?.startsWith(year)),
            ) ||
            candidates.find(
              (item) => !year || item.release_date?.startsWith(year),
            );
          if (!match) return null;
          return {
            poster: match.poster_path
              ? `https://image.tmdb.org/t/p/w342${match.poster_path}`
              : null,
            backdrop: match.backdrop_path
              ? `https://image.tmdb.org/t/p/w1280${match.backdrop_path}`
              : null,
            overview: match.overview || "",
            tmdbId: match.id,
          };
        })
        .catch(() => {
          artworkCache.delete(title);
          return null;
        }),
    );
  }
  return artworkCache.get(title);
}
