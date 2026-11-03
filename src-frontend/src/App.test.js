import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import App from "./App";
import { api } from "./lib/api";

jest.mock("./lib/api", () => ({
  api: { get: jest.fn(), post: jest.fn() },
  getSessionId: () => "test-session",
  normalizeMovies: (items = []) =>
    items.map((item) =>
      Array.isArray(item)
        ? { item_id: item[0], title: item[1], score: item[2] }
        : item,
    ),
  getArtwork: () => Promise.resolve(null),
}));

const popular = [
  [1, "Toy Story (1995)", 1],
  [260, "Star Wars (1977)", 1],
];
let tasteLearned;
beforeEach(() => {
  jest.clearAllMocks();
  tasteLearned = false;
  api.get.mockImplementation(async (url) =>
    url === "/similar_items"
      ? {
          data: {
            similar: [{ item_id: 2, title: "Jumanji (1995)", score: 0.8 }],
          },
        }
      : {
          data: {
            top_genres: tasteLearned ? [["Animation", 1]] : [],
            recent: [],
            persona: tasteLearned ? "Animation" : "Unknown",
          },
        },
  );
  api.post.mockImplementation(async (url, body) => {
    if (url === "/event") {
      tasteLearned = true;
      return { data: { ok: true } };
    }
    if (url === "/lantern/reset") {
      tasteLearned = false;
      return { data: { ok: true } };
    }
    if (url === "/lantern/recommend")
      return {
        data: {
          recommendations: popular,
          ...(tasteLearned ? {} : { note: "popular_fallback" }),
        },
      };
    return {
      data: {
        recommendations: popular,
        source: body.gender === "UNKNOWN" ? "popular" : "model",
      },
    };
  });
});

async function ready() {
  render(<App />);
  await screen.findByRole("button", { name: "Explore Toy Story" });
  await screen.findByRole("button", { name: "Explore Jumanji" });
  await waitFor(() =>
    expect(screen.queryAllByText("Loading artwork")).toHaveLength(0),
  );
}

test("loads popular discoveries without requesting personal details or a poster key", async () => {
  await ready();
  expect(
    screen.getByRole("heading", { name: "Toy Story", level: 1 }),
  ).toBeInTheDocument();
  expect(screen.getByText("Popular in MovieLens")).toBeInTheDocument();
  expect(api.post).toHaveBeenCalledWith("/recommend", {
    gender: "UNKNOWN",
    age: -1,
    occupation: -1,
  });
});

test("submits the selected profile and explains the model recommendation source", async () => {
  await ready();
  fireEvent.click(
    screen.getByRole("button", { name: /Personalize your picks/i }),
  );
  fireEvent.change(screen.getByLabelText("Gender"), { target: { value: "F" } });
  fireEvent.change(screen.getByLabelText("Age group"), {
    target: { value: "25" },
  });
  fireEvent.change(screen.getByLabelText("Occupation"), {
    target: { value: "12" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Find my films" }));
  expect(
    await screen.findByText("Picked for your profile"),
  ).toBeInTheDocument();
  expect(api.post).toHaveBeenCalledWith("/recommend", {
    gender: "F",
    age: 25,
    occupation: 12,
  });
});

test("likes a film in an isolated session, updates taste, prevents duplicate likes and resets it", async () => {
  await ready();
  fireEvent.click(screen.getByRole("button", { name: "Like Toy Story" }));
  expect(await screen.findByText("Animation")).toBeInTheDocument();
  expect(api.post).toHaveBeenCalledWith("/event", {
    user_id: "test-session",
    item_id: 1,
    event: "like",
  });
  expect(
    screen.getByRole("button", { name: "Liked Toy Story" }),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("tab", { name: /for your taste/i }));
  expect(screen.getByText("Inspired by your likes")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  await waitFor(() =>
    expect(screen.queryByText("Animation")).not.toBeInTheDocument(),
  );
  expect(api.post).toHaveBeenCalledWith("/lantern/reset", {
    user_id: "test-session",
  });
  expect(await screen.findByText("Popular in MovieLens")).toBeInTheDocument();
});

test("opens similar films in an accessible dialog and closes with Escape", async () => {
  await ready();
  const trigger = screen.getByRole("button", { name: "Explore Toy Story" });
  trigger.focus();
  fireEvent.click(trigger);
  const dialog = screen.getByRole("dialog", { name: "If you like Toy Story." });
  expect(
    await within(dialog).findByRole("heading", { name: "Jumanji" }),
  ).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith(
    "/similar_items",
    expect.objectContaining({ params: { item_id: 1, k: 6 } }),
  );
  expect(
    within(dialog).getByRole("button", { name: "Close similar films" }),
  ).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("shows a recoverable error when the backend is unavailable", async () => {
  const implementation = api.post.getMockImplementation();
  api.post.mockImplementation((url, body) =>
    url === "/recommend"
      ? Promise.reject(new Error("offline"))
      : implementation(url, body),
  );
  render(<App />);
  expect(await screen.findByText("A brief intermission.")).toBeInTheDocument();
  api.post.mockImplementation(implementation);
  fireEvent.click(screen.getByRole("button", { name: /try again/i }));
  expect(
    await screen.findByRole("button", { name: "Explore Toy Story" }),
  ).toBeInTheDocument();
});

test("allows switching result tabs with arrow keys", async () => {
  await ready();
  const discover = screen.getByRole("tab", { name: "Discover" });
  fireEvent.keyDown(discover, { key: "ArrowRight" });
  const taste = screen.getByRole("tab", { name: /for your taste/i });
  expect(taste).toHaveAttribute("aria-selected", "true");
  expect(taste).toHaveFocus();
  expect(screen.getByRole("tabpanel")).toHaveAttribute(
    "aria-labelledby",
    "taste-tab",
  );
});
