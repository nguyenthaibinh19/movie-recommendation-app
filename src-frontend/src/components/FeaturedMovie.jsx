import React, { useEffect, useState } from "react";
import { movieTitle, useMovieArtwork } from "./MovieCard";
import { Icon } from "./Icons";

export default function FeaturedMovie({ movie, onOpen, onPersonalize }) {
  const { artwork } = useMovieArtwork(movie?.title);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [artwork?.backdrop]);
  return (
    <section
      className={`featured-movie ${artwork?.backdrop && !failed ? "has-backdrop" : ""}`}
      aria-labelledby="hero-title"
    >
      {artwork?.backdrop && !failed && (
        <img
          className="featured-backdrop"
          src={artwork.backdrop}
          alt=""
          fetchPriority="high"
          onError={() => setFailed(true)}
        />
      )}
      <div className="featured-shade" />
      <div className="featured-copy">
        <div className="eyebrow">
          <span className="status-dot" /> YOUR NEXT MOVIE NIGHT
        </div>
        <h1 id="hero-title">
          {movie ? movieTitle(movie.title) : "Find your next favorite."}
        </h1>
        <div className="featured-meta">
          <span>
            {movie?.title.match(/\((\d{4})\)$/)?.[1] || "Movie discovery"}
          </span>
          <span className="meta-divider" />
          <span>From the MovieLens collection</span>
        </div>
        <p>
          {artwork?.overview ||
            "A great movie is just the beginning. Discover classics, explore similar films, and find more of what you love."}
        </p>
        <div className="featured-actions">
          {movie && (
            <button className="primary-button" onClick={() => onOpen(movie)}>
              <Icon name="info" /> More info
            </button>
          )}
          <button className="secondary-button" onClick={onPersonalize}>
            <Icon name="sliders" /> Personalize picks
          </button>
        </div>
        <span className="featured-note">
          Discover films. Shape your recommendations.
        </span>
      </div>
      <div className="featured-caption">
        <span>SPOTLIGHT</span>
        <span>01 / 10</span>
      </div>
    </section>
  );
}
