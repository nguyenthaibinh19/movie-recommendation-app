import React, { useEffect, useRef, useState } from "react";
import MovieCard, { movieTitle } from "./MovieCard";
import { api, normalizeMovies } from "../lib/api";
import { Icon } from "./Icons";

export function MovieRail({
  movies,
  title,
  description,
  source,
  liked,
  busy,
  onLike,
  onOpen,
  label,
}) {
  const rail = useRef(null);
  function scroll(direction) {
    rail.current?.scrollBy({
      left: direction * rail.current.clientWidth * 0.8,
      behavior: "smooth",
    });
  }
  return (
    <section className="film-rail" aria-label={label || title}>
      <div className="rail-heading">
        <div>
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        <div className="rail-controls">
          <button
            onClick={() => scroll(-1)}
            aria-label={`Scroll ${label || title} backward`}
          >
            <Icon name="chevronLeft" />
          </button>
          <button
            onClick={() => scroll(1)}
            aria-label={`Scroll ${label || title} forward`}
          >
            <Icon name="chevronRight" />
          </button>
        </div>
      </div>
      <div
        className="movie-rail"
        ref={rail}
        tabIndex={0}
        aria-label={`${label || title} film list`}
      >
        {movies.map((movie, index) => (
          <MovieCard
            key={movie.item_id}
            movie={movie}
            index={index}
            source={source}
            liked={liked.has(movie.item_id)}
            busy={busy}
            onLike={onLike}
            onOpen={onOpen}
          />
        ))}
      </div>
    </section>
  );
}

export function SimilarRail({ movie, liked, busy, onLike, onOpen }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!movie) return;
    let active = true;
    const controller = new AbortController();
    setItems([]);
    setError(false);
    api
      .get("/similar_items", {
        params: { item_id: movie.item_id, k: 6 },
        signal: controller.signal,
      })
      .then(({ data }) => {
        if (active) setItems(normalizeMovies(data.similar));
      })
      .catch(() => {
        if (active) setError(true);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [movie, attempt]);
  if (!movie) return null;
  if (error)
    return (
      <section className="neighbor-error">
        <p>Similar films are temporarily unavailable.</p>
        <button className="text-link" onClick={() => setAttempt((n) => n + 1)}>
          Try again <Icon name="refresh" />
        </button>
      </section>
    );
  if (!items.length) return null;
  return (
    <MovieRail
      movies={items}
      title={`More like ${movieTitle(movie.title)}`}
      description="Explore nearby films in the collection."
      source="similar"
      liked={liked}
      busy={busy}
      onLike={onLike}
      onOpen={onOpen}
      label="Similar discoveries"
    />
  );
}
