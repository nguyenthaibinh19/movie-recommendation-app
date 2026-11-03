import React, { useEffect, useState } from "react";
import { getArtwork } from "../lib/api";
import { Icon } from "./Icons";

export function movieTitle(title) {
  return title
    .replace(/\s*\(\d{4}\)$/, "")
    .replace(/^(.*), (The|A|An)$/, "$2 $1");
}

export function useMovieArtwork(title) {
  const [artwork, setArtwork] = useState(null);
  const [loading, setLoading] = useState(Boolean(title));
  useEffect(() => {
    let active = true;
    setArtwork(null);
    setLoading(Boolean(title));
    getArtwork(title).then((data) => {
      if (active) {
        setArtwork(data);
        setLoading(false);
      }
    });
    return () => {
      active = false;
    };
  }, [title]);
  return { artwork, loading };
}

export function MovieCover({ movie }) {
  const { artwork, loading } = useMovieArtwork(movie.title);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [movie.title]);
  return (
    <div
      className={`movie-cover ${loading ? "artwork-loading" : ""}`}
      aria-hidden="true"
    >
      {artwork?.poster && !failed ? (
        <img
          src={artwork.poster}
          alt=""
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="artwork-fallback">
          <Icon name="film" />
          <span>{loading ? "Loading artwork" : "Poster unavailable"}</span>
        </div>
      )}
    </div>
  );
}

export default function MovieCard({
  movie,
  index,
  liked,
  busy,
  onLike,
  onOpen,
  source,
}) {
  return (
    <article className="movie-card">
      <button
        className="movie-open"
        onClick={() => onOpen(movie)}
        aria-label={`Explore ${movieTitle(movie.title)}`}
      >
        <MovieCover movie={movie} />
        {source !== "similar" && (
          <span className="movie-rank">
            {String(index + 1).padStart(2, "0")}
          </span>
        )}
        <span className="cover-cta">
          <Icon name="info" /> More like this
        </span>
      </button>
      <div className="movie-info">
        <h3>
          <button onClick={() => onOpen(movie)}>
            {movieTitle(movie.title)}
          </button>
        </h3>
        <button
          className={`like-button ${liked ? "is-liked" : ""}`}
          onClick={() => onLike(movie)}
          disabled={busy || liked}
          aria-label={`${liked ? "Liked" : "Like"} ${movieTitle(movie.title)}`}
          aria-pressed={liked}
        >
          <Icon name="heart" />
        </button>
        <p>
          {movie.title.match(/\((\d{4})\)$/)?.[1] || "MovieLens"}
          <span>·</span>
          {source === "model"
            ? "For your profile"
            : source === "taste"
              ? "For your taste"
              : source === "similar"
                ? "Similar film"
                : "Popular pick"}
        </p>
      </div>
    </article>
  );
}
