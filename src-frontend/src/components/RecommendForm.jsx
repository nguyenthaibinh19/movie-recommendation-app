import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, getSessionId, normalizeMovies } from "../lib/api";
import FeaturedMovie from "./FeaturedMovie";
import { MovieRail, SimilarRail } from "./MovieRail";
import MovieDialog from "./MovieDialog";
import { Icon } from "./Icons";

const OCCUPATIONS = [
  "Other / not specified",
  "Academic / educator",
  "Artist",
  "Clerical / admin",
  "College / grad student",
  "Customer service",
  "Doctor / healthcare",
  "Executive / manager",
  "Farmer",
  "Homemaker",
  "K–12 student",
  "Lawyer",
  "Programmer",
  "Retired",
  "Sales / marketing",
  "Scientist",
  "Self-employed",
  "Technician / engineer",
  "Tradesperson",
  "Unemployed",
  "Writer",
];
const AGES = [
  [1, "Under 18"],
  [18, "18–24"],
  [25, "25–34"],
  [35, "35–44"],
  [45, "45–49"],
  [50, "50–55"],
  [56, "56+"],
];
const EMPTY_PROFILE = { gender: "UNKNOWN", age: -1, occupation: -1 };

export default function RecommendForm() {
  const [userId] = useState(getSessionId);
  const [profile, setProfile] = useState(EMPTY_PROFILE);
  const [movies, setMovies] = useState([]);
  const [source, setSource] = useState("popular");
  const [tasteMovies, setTasteMovies] = useState([]);
  const [tasteSource, setTasteSource] = useState("popular");
  const [insight, setInsight] = useState(null);
  const [tab, setTab] = useState("discover");
  const [profileOpen, setProfileOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [tasteError, setTasteError] = useState("");
  const [tasteBusy, setTasteBusy] = useState(false);
  const [liked, setLiked] = useState(new Set());
  const [selected, setSelected] = useState(null);
  const [notice, setNotice] = useState("");
  const timer = useRef(null);
  const results = useRef(null);
  const mutation = useRef(false);

  const refreshTaste = useCallback(async () => {
    const [summary, picks] = await Promise.all([
      api.get("/lantern/profile", { params: { user_id: userId } }),
      api.post("/lantern/recommend", { user_id: userId, exclude: [] }),
    ]);
    setInsight(summary.data);
    setTasteMovies(normalizeMovies(picks.data.recommendations));
    setTasteSource(
      picks.data.note === "popular_fallback" ? "popular" : "taste",
    );
    setTasteError("");
  }, [userId]);

  useEffect(() => {
    let active = true;
    api
      .post("/recommend", EMPTY_PROFILE)
      .then(({ data }) => {
        if (active) {
          setMovies(normalizeMovies(data.recommendations));
          setSource(data.source);
        }
      })
      .catch(() => {
        if (active)
          setError(
            "We couldn’t reach the recommendation server. Check that the backend is running, then try again.",
          );
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    refreshTaste().catch(() => {
      if (active)
        setTasteError("Your taste profile is unavailable. Try refreshing it.");
    });
    return () => {
      active = false;
      clearTimeout(timer.current);
    };
  }, [refreshTaste]);

  function notify(message) {
    setNotice(message);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setNotice(""), 3500);
  }

  async function recommend(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    setTab("discover");
    try {
      const { data } = await api.post("/recommend", profile);
      setMovies(normalizeMovies(data.recommendations));
      setSource(data.source);
    } catch {
      setError(
        "We couldn’t load your recommendations. Check the backend connection and try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function likeMovie(movie) {
    if (mutation.current || liked.has(movie.item_id)) return;
    mutation.current = true;
    setTasteBusy(true);
    try {
      await api.post("/event", {
        user_id: userId,
        item_id: movie.item_id,
        event: "like",
      });
      setLiked((current) => new Set([...current, movie.item_id]));
      notify("Saved to your taste. Lantern is learning.");
      await refreshTaste();
    } catch {
      setTasteError(
        "We couldn’t update your taste. Try refreshing, or like a film again.",
      );
    } finally {
      mutation.current = false;
      setTasteBusy(false);
    }
  }

  async function refresh() {
    if (mutation.current) return;
    mutation.current = true;
    setTasteBusy(true);
    try {
      await refreshTaste();
    } catch {
      setTasteError(
        "Your taste profile is unavailable. Check the backend and try again.",
      );
    } finally {
      mutation.current = false;
      setTasteBusy(false);
    }
  }

  async function resetTaste() {
    if (mutation.current) return;
    mutation.current = true;
    setTasteBusy(true);
    try {
      await api.post("/lantern/reset", { user_id: userId });
      setLiked(new Set());
      setInsight(null);
      await refreshTaste();
      notify("A fresh start. Your taste has been reset.");
    } catch {
      setTasteError("We couldn’t reset your taste. Please try again.");
    } finally {
      mutation.current = false;
      setTasteBusy(false);
    }
  }

  const closeDialog = useCallback(() => setSelected(null), []);
  const visible = tab === "discover" ? movies : tasteMovies;
  const currentSource = tab === "discover" ? source : tasteSource;
  const genres = insight?.top_genres || [];
  const hasTaste = genres.length > 0;
  const sourceLabel =
    currentSource === "model"
      ? "Picked for your profile"
      : currentSource === "taste"
        ? "Inspired by your likes"
        : "Popular in MovieLens";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="site-name" href="#hero-title">
          Movie Recommender
        </a>
        <nav aria-label="Main navigation">
          <a href="#discover">Discover</a>
          <a href="#preferences" onClick={() => setProfileOpen(true)}>
            Your profile
          </a>
          <a href="#how-it-works">About</a>
        </nav>
        <a
          className="source-link"
          href="https://github.com/nguyenthaibinh19/movie-recommendation-app"
          target="_blank"
          rel="noreferrer"
        >
          <Icon name="github" />
          <span>View project</span>
        </a>
      </header>
      <main>
        <FeaturedMovie
          movie={movies[0]}
          onOpen={setSelected}
          onPersonalize={() => {
            setProfileOpen(true);
            document
              .getElementById("preferences")
              ?.scrollIntoView({ block: "center", behavior: "smooth" });
          }}
        />
        <div className="content-shell">
          <section
            id="preferences"
            className={`profile-panel ${profileOpen ? "is-open" : ""}`}
          >
            <button
              className="profile-toggle"
              onClick={() => setProfileOpen((open) => !open)}
              aria-expanded={profileOpen}
              aria-controls="profile-form"
            >
              <span className="profile-toggle-icon">
                <Icon name="sliders" />
              </span>
              <span>
                <strong>Personalize your picks</strong>
                <span>Optional details. A different starting point.</span>
              </span>
              <Icon name={profileOpen ? "close" : "chevronRight"} />
            </button>
            {profileOpen && (
              <form id="profile-form" onSubmit={recommend}>
                <div className="profile-field">
                  <label htmlFor="gender">Gender</label>
                  <select
                    id="gender"
                    value={profile.gender}
                    onChange={(e) =>
                      setProfile({ ...profile, gender: e.target.value })
                    }
                  >
                    <option value="UNKNOWN">Prefer not to say</option>
                    <option value="F">Female</option>
                    <option value="M">Male</option>
                  </select>
                </div>
                <div className="profile-field">
                  <label htmlFor="age">Age group</label>
                  <select
                    id="age"
                    value={profile.age}
                    onChange={(e) =>
                      setProfile({ ...profile, age: Number(e.target.value) })
                    }
                  >
                    <option value={-1}>Prefer not to say</option>
                    {AGES.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="profile-field">
                  <label htmlFor="occupation">Occupation</label>
                  <select
                    id="occupation"
                    value={profile.occupation}
                    onChange={(e) =>
                      setProfile({
                        ...profile,
                        occupation: Number(e.target.value),
                      })
                    }
                  >
                    <option value={-1}>Prefer not to say</option>
                    {OCCUPATIONS.map((name, value) => (
                      <option key={value} value={value}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={loading}
                >
                  {loading ? "Finding your films…" : "Find my films"}
                  <Icon name="arrow" />
                </button>
                <p className="form-note">
                  Keep any detail private and we’ll start with popular
                  favorites.
                </p>
              </form>
            )}
          </section>

          <section
            id="discover"
            className="results"
            ref={results}
            aria-label="Movie discovery"
          >
            <div className="results-toolbar">
              <div
                className="result-tabs"
                role="tablist"
                aria-label="Recommendation type"
              >
                <button
                  role="tab"
                  id="discover-tab"
                  aria-controls="movie-results"
                  aria-selected={tab === "discover"}
                  tabIndex={tab === "discover" ? 0 : -1}
                  onKeyDown={(e) => {
                    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
                      e.preventDefault();
                      setTab("taste");
                      document.getElementById("taste-tab").focus();
                    }
                  }}
                  onClick={() => setTab("discover")}
                >
                  Discover
                </button>
                <button
                  role="tab"
                  id="taste-tab"
                  aria-controls="movie-results"
                  aria-selected={tab === "taste"}
                  tabIndex={tab === "taste" ? 0 : -1}
                  onKeyDown={(e) => {
                    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
                      e.preventDefault();
                      setTab("discover");
                      document.getElementById("discover-tab").focus();
                    }
                  }}
                  onClick={() => setTab("taste")}
                >
                  For your taste
                </button>
              </div>
              <span className="source-badge">
                <span className="status-dot" />
                {sourceLabel}
              </span>
            </div>
            <div
              id="movie-results"
              role="tabpanel"
              aria-labelledby={
                tab === "discover" ? "discover-tab" : "taste-tab"
              }
              aria-busy={loading || tasteBusy}
            >
              {tab === "taste" && (
                <div className="taste-intro">
                  <Icon name="heart" />
                  <p>
                    {tasteSource === "popular"
                      ? "Like a few films to turn popular favorites into more personal recommendations."
                      : "Inspired by the films you like. Keep exploring to refine your recommendations."}
                  </p>
                </div>
              )}
              {loading && tab === "discover" ? (
                <div className="loading-state" role="status">
                  <p>Finding your next watch...</p>
                  <div className="skeleton-rail">
                    {[0, 1, 2, 3, 4, 5].map((i) => (
                      <div className="skeleton" key={i} />
                    ))}
                  </div>
                </div>
              ) : error && tab === "discover" ? (
                <div className="error-message" role="alert">
                  <h2>A brief intermission.</h2>
                  <p>{error}</p>
                  <button className="primary-button" onClick={recommend}>
                    Try again <Icon name="refresh" />
                  </button>
                </div>
              ) : visible.length ? (
                <MovieRail
                  movies={visible}
                  title={
                    tab === "taste"
                      ? "Picked for your taste"
                      : source === "model"
                        ? "Recommended for you"
                        : "Popular favorites"
                  }
                  description={
                    tab === "taste"
                      ? "Your next discovery starts with what you love."
                      : "A collection of classics, ready to explore."
                  }
                  source={currentSource}
                  liked={liked}
                  busy={tasteBusy}
                  onLike={likeMovie}
                  onOpen={setSelected}
                  label="Top recommendations"
                />
              ) : (
                <div className="empty-state">
                  <h2>Your next discovery is on its way.</h2>
                  <p>Refresh your taste to find a starting point.</p>
                  <button className="text-link" onClick={refresh}>
                    Refresh taste <Icon name="refresh" />
                  </button>
                </div>
              )}
            </div>
          </section>

          <section className="taste-panel" aria-labelledby="taste-title">
            <div className="taste-summary">
              <span className="taste-icon">
                <Icon name="heart" />
              </span>
              <div>
                <span className="eyebrow">BUILT AROUND YOU</span>
                <h2 id="taste-title">
                  {hasTaste
                    ? "Your taste. Your next discovery."
                    : "A few likes go a long way."}
                </h2>
                <p>
                  {hasTaste
                    ? "These are a few of the genres behind your latest likes."
                    : "Like the films you love. Your recommendations learn from every one."}
                </p>
                {hasTaste && (
                  <div className="genre-tags">
                    {genres.slice(0, 4).map(([genre]) => (
                      <span key={genre}>{genre}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <div className="taste-actions">
              <button
                className="secondary-button"
                onClick={() => {
                  setTab("taste");
                  results.current?.scrollIntoView({
                    block: "start",
                    behavior: "smooth",
                  });
                }}
              >
                Explore your taste <Icon name="arrow" />
              </button>
              <div>
                <button
                  className="text-link"
                  onClick={refresh}
                  disabled={tasteBusy}
                >
                  <Icon name="refresh" /> Refresh taste
                </button>
                <button
                  className="text-link muted"
                  onClick={resetTaste}
                  disabled={tasteBusy || !hasTaste}
                >
                  Reset
                </button>
              </div>
            </div>
            {tasteError && (
              <p className="taste-error" role="alert">
                {tasteError}
              </p>
            )}
          </section>

          {!loading && !error && (
            <SimilarRail
              movie={movies[0]}
              liked={liked}
              busy={tasteBusy}
              onLike={likeMovie}
              onOpen={setSelected}
            />
          )}

          <section
            className="how-section"
            id="how-it-works"
            aria-labelledby="how-title"
          >
            <div>
              <span className="eyebrow">ABOUT THIS PROJECT</span>
              <h2 id="how-title">Find something worth watching.</h2>
              <p>
                A movie discovery demo built with the MovieLens 1M collection.
                Profile-based recommendations, similar-film discovery, and a
                taste that evolves with your likes.
              </p>
            </div>
            <div className="project-facts">
              <span>
                <strong>3,883</strong>films in the collection
              </span>
              <span>
                <strong>1M+</strong>historical ratings
              </span>
              <span>
                <strong>Your taste</strong>learns in this session
              </span>
            </div>
          </section>
        </div>
      </main>
      <footer className="site-footer">
        <div>
          <span className="footer-title">Movie Recommender</span>
          <p>A discovery demo. Film playback is not provided.</p>
        </div>
        <div className="footer-links">
          <a
            href="https://grouplens.org/datasets/movielens/1m/"
            target="_blank"
            rel="noreferrer"
          >
            MovieLens 1M
          </a>
          <a href="#how-it-works">About this project</a>
        </div>
        {process.env.REACT_APP_TMDB_API_KEY && (
          <div className="artwork-credit">
            <a
              href="https://www.themoviedb.org"
              target="_blank"
              rel="noreferrer"
            >
              <img
                src="https://files.readme.io/29c6fee-blue_short.svg"
                alt="TMDB"
                width="92"
                height="13"
              />
              Movie artwork &amp; descriptions
            </a>
            <p>
              This product uses the TMDB API but is not endorsed or certified by
              TMDB.
            </p>
          </div>
        )}
      </footer>
      {notice && (
        <div className="toast" role="status">
          <Icon name="heart" />
          {notice}
        </div>
      )}
      {selected && <MovieDialog movie={selected} onClose={closeDialog} />}
    </div>
  );
}
