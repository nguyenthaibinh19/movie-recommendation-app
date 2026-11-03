import React, { useEffect, useRef, useState } from "react";
import { api, normalizeMovies } from "../lib/api";
import { MovieCover, movieTitle } from "./MovieCard";
import { Icon } from "./Icons";

export default function MovieDialog({ movie, onClose }) {
  const dialog = useRef(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    api
      .get("/similar_items", {
        params: { item_id: movie.item_id, k: 6 },
        signal: controller.signal,
      })
      .then(({ data }) => {
        if (active) setItems(normalizeMovies(data.similar));
      })
      .catch(() => {
        if (active)
          setError("We couldn’t load similar films. Please try again.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [movie.item_id, attempt]);

  useEffect(() => {
    const previous = document.activeElement;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current.querySelector("button").focus();
    function handleKey(event) {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const controls = dialog.current.querySelectorAll(
          'button, a[href], [tabindex="0"]',
        );
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", handleKey);
      previous?.focus();
    };
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={dialog}
        className="movie-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="similar-title"
        aria-describedby="similar-description"
      >
        <button
          className="icon-button modal-close"
          onClick={onClose}
          aria-label="Close similar films"
        >
          <Icon name="close" />
        </button>
        <span className="eyebrow">KEEP EXPLORING</span>
        <h2 id="similar-title">If you like {movieTitle(movie.title)}.</h2>
        <p id="similar-description">
          Six nearby films in the learned movie embedding space.
        </p>
        {loading && (
          <p role="status" className="dialog-message">
            Finding similar films…
          </p>
        )}
        {error && (
          <div role="alert" className="error-message">
            {error}
            <button onClick={() => setAttempt((n) => n + 1)}>Try again</button>
          </div>
        )}
        {!loading && !error && (
          <div className="similar-grid">
            {items.map((item) => (
              <article key={item.item_id}>
                <MovieCover movie={item} />
                <h3>{movieTitle(item.title)}</h3>
                <p>Cosine similarity · {Number(item.score).toFixed(3)}</p>
              </article>
            ))}
            {!items.length && <p>No similar films found.</p>}
          </div>
        )}
        <div className="dialog-footnote">
          Similarity reflects learned patterns, not a star rating.
        </div>
      </section>
    </div>
  );
}
