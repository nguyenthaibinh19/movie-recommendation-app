import axios from "axios";
import { getArtwork } from "./api";

jest.mock("axios", () => ({ create: () => ({}), get: jest.fn() }));
const originalKey = process.env.REACT_APP_TMDB_API_KEY;

afterEach(() => {
  if (originalKey === undefined) delete process.env.REACT_APP_TMDB_API_KEY;
  else process.env.REACT_APP_TMDB_API_KEY = originalKey;
  jest.clearAllMocks();
});

test("does not request external artwork without a configured key", async () => {
  delete process.env.REACT_APP_TMDB_API_KEY;
  expect(await getArtwork("No key film (1994)")).toBeNull();
  expect(axios.get).not.toHaveBeenCalled();
});

test("matches title and year and shares one poster/backdrop lookup across components", async () => {
  process.env.REACT_APP_TMDB_API_KEY = "test-key";
  axios.get.mockResolvedValue({
    data: {
      results: [
        {
          id: 99,
          title: "Another Movie",
          release_date: "1994-01-01",
          poster_path: "/wrong.jpg",
        },
        {
          id: 100,
          title: "The Professional",
          release_date: "1994-09-14",
          poster_path: "/poster.jpg",
          backdrop_path: "/backdrop.jpg",
          overview: "Film description.",
        },
      ],
    },
  });
  const first = getArtwork("Professional, The (1994)");
  const second = getArtwork("Professional, The (1994)");
  expect(second).toBe(first);
  expect(await first).toEqual({
    poster: "https://image.tmdb.org/t/p/w342/poster.jpg",
    backdrop: "https://image.tmdb.org/t/p/w1280/backdrop.jpg",
    overview: "Film description.",
    tmdbId: 100,
  });
  expect(axios.get).toHaveBeenCalledTimes(1);
  expect(axios.get).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      params: {
        api_key: "test-key",
        query: "The Professional",
        primary_release_year: "1994",
      },
    }),
  );
});

test("allows retry after a failed image service request", async () => {
  process.env.REACT_APP_TMDB_API_KEY = "test-key";
  axios.get.mockRejectedValueOnce(new Error("offline"));
  expect(await getArtwork("Retry film (1995)")).toBeNull();
  axios.get.mockResolvedValueOnce({
    data: {
      results: [
        {
          id: 1,
          title: "Retry film",
          release_date: "1995-01-01",
          poster_path: "/retry.jpg",
        },
      ],
    },
  });
  expect((await getArtwork("Retry film (1995)")).poster).toContain(
    "/retry.jpg",
  );
  expect(axios.get).toHaveBeenCalledTimes(2);
});
