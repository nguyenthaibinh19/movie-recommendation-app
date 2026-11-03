import React from "react";

const paths = {
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v6M12 7h.01" />
    </>
  ),
  film: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M7 4v16M17 4v16M3 9h4m-4 6h4m10-6h4m-4 6h4" />
    </>
  ),
  sliders: (
    <>
      <path d="M4 7h5m4 0h7M4 17h9m4 0h3" />
      <circle cx="11" cy="7" r="2" />
      <circle cx="15" cy="17" r="2" />
    </>
  ),
  chevronLeft: <path d="m14 6-6 6 6 6" />,
  chevronRight: <path d="m10 6 6 6-6 6" />,
  arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
  heart: (
    <path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z" />
  ),
  close: <path d="m6 6 12 12M6 18 18 6" />,
  refresh: (
    <>
      <path d="M20 7v5h-5M4 17v-5h5" />
      <path d="M6.1 6.1a8 8 0 0 1 13.3 3M4.6 14.9a8 8 0 0 0 13.3 3" />
    </>
  ),
  spark: (
    <>
      <path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4Z" />
      <path d="M21 2v4m-2-2h4" />
    </>
  ),
  github: (
    <>
      <path d="M9 19c-4.3 1.3-4.3-2-6-2m12 5v-3.4c0-1 .1-1.5-.5-2.1 3.3-.4 6.8-1.6 6.8-7.3A5.7 5.7 0 0 0 19.8 5a5.2 5.2 0 0 0-.2-4s-1.2-.4-4.1 1.5a14 14 0 0 0-7 0C5.6.6 4.4 1 4.4 1A5.2 5.2 0 0 0 4.2 5a5.7 5.7 0 0 0-1.5 4.2c0 5.7 3.5 6.9 6.8 7.3-.5.5-.6 1.2-.5 2.1V22" />
    </>
  ),
};

export function Icon({ name, ...props }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
