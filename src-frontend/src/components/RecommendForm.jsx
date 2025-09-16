import React, { useState, useEffect } from 'react';
import axios from 'axios';

// ⭐ BỎ auto-refresh top-K khi click item (giữ false)
const ENABLE_AUTO_EVENT = false;

function RecommendForm() {
  // Lantern panel + picks
  const [lantern, setLantern] = useState(null); // { persona, top_genres, recent[], u_norm, ... }
  const [lanternPicks, setLanternPicks] = useState([]); // ⭐ gợi ý riêng theo Lantern
  const [toast, setToast] = useState(null); // ⭐ thông báo nhỏ sau khi like

  // ⭐ Form state: mặc định Unknown
  const [gender, setGender] = useState('UNKNOWN');
  const [age, setAge] = useState(-1);
  const [occupation, setOccupation] = useState(-1);

  // Recommendations theo demographic (/recommend)
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState(null);
  const [lastQuery, setLastQuery] = useState(null);

  // Modal Similar
  const [modalOpen, setModalOpen] = useState(false);
  const [modalBaseItem, setModalBaseItem] = useState(null);
  const [modalSimilar, setModalSimilar] = useState([]);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState('');

  // -------- Lantern helpers --------
  const fetchLanternProfile = async () => {
    try {
      const res = await axios.get('http://localhost:5000/lantern/profile', { params: { user_id: 'guest' } });
      setLantern(res.data);
    } catch (e) {
      console.warn('[Lantern profile] fail:', e?.message);
    }
  };

  const fetchLanternPicks = async () => {
    try {
      const res = await axios.post('http://localhost:5000/lantern/recommend', {
        user_id: 'guest',
        exclude: [], // có thể thêm exclude nếu cần
      });
      const enriched = await enrichWithPosters(res.data.recommendations || []);
      setLanternPicks(enriched);
    } catch (e) {
      console.warn('[Lantern picks] fail:', e?.message);
    }
  };

  const lanternReset = async () => {
    try {
      await axios.post('http://localhost:5000/lantern/reset', { user_id: 'guest' });
      setLantern(null);
      setLanternPicks([]);
      setToast('Lantern reset ✓');
      setTimeout(() => setToast(null), 1200);
    } catch (e) {
      console.warn('[Lantern reset] fail:', e?.message);
    }
  };

  // -------- Poster helpers --------
  const getPosterURL = async (title) => {
    try {
      const cleanedTitle = title.replace(/\s*\(\d{4}\)$/, '');
      const res = await axios.get('https://api.themoviedb.org/3/search/movie', {
        params: { api_key: 'c24381c0a201e1fe30d94c02463bc6ca', query: cleanedTitle }
      });
      const posterPath = res.data.results[0]?.poster_path;
      return posterPath ? `https://image.tmdb.org/t/p/w200${posterPath}` : null;
    } catch (err) {
      console.error(`[❌ TMDB Error]: Failed to get poster for ${title}`, err);
      return null;
    }
  };

  const enrichWithPosters = async (recs) => {
    return Promise.all(
      recs.map(async (r) => {
        const [iid, ttl, sc] = Array.isArray(r) ? r : [r.item_id, r.title, r.score];
        const poster = await getPosterURL(ttl);
        return { item_id: iid, title: ttl, score: sc, poster };
      })
    );
  };

  // -------- Submit form -> /recommend (demographic) --------
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError(null);

      // ⭐ Cho phép Unknown: nếu user không chọn gì, vẫn gửi lên để BE dùng default
      const payload = {
        gender,  // 'M' | 'F' | 'UNKNOWN'
        age: Number(age),           // -1 = unknown
        occupation: Number(occupation) // -1 = unknown
      };

      setLastQuery(payload);

      const res = await axios.post('http://localhost:5000/recommend', payload);
      const enriched = await enrichWithPosters(res.data.recommendations || []);
      setRecommendations(enriched);

      // Reset modal nếu đang mở
      setModalOpen(false);
      setModalBaseItem(null);
      setModalSimilar([]);
      setModalError('');

      // Cập nhật Lantern panel + picks
      fetchLanternProfile();
      fetchLanternPicks();
    } catch (err) {
      console.error('[❌ Network Error]:', err);
      setError('Something went wrong. Please try again.');
    }
  };

  // -------- Refresh demographic recommendations --------
  const handleRefreshRecommend = async () => {
    if (!lastQuery) return;
    try {
      setError(null);
      const res = await axios.post('http://localhost:5000/recommend', lastQuery);
      const enriched = await enrichWithPosters(res.data.recommendations || []);
      setRecommendations(enriched);
      fetchLanternProfile();
      fetchLanternPicks();
    } catch (err) {
      console.error('[❌ Refresh Error]:', err);
      setError('Refresh failed. Please try again.');
    }
  };

  // -------- (Tuỳ chọn) gửi event khi user click item --------
  const sendEventIfEnabled = async (item_id, event = 'click') => {
    if (!ENABLE_AUTO_EVENT) return;
    try {
      await axios.post('http://localhost:5000/event', { user_id: 'guest', item_id, event });
      // ⭐ Không reset danh sách top-K chính
      fetchLanternProfile();
      fetchLanternPicks();
    } catch (e) {
      console.warn('[ℹ] Event not sent:', e?.message);
    }
  };

  // -------- Like button ở mỗi item (chỉ ghi nhận) --------
  const handleLike = async (item_id) => {
    try {
      await axios.post('http://localhost:5000/event', {
        user_id: 'guest',
        item_id,
        event: 'like',
      });
      // ⭐ Chỉ cập nhật Lantern insight + picks, KHÔNG reset top-K chính
      fetchLanternProfile();
      fetchLanternPicks();
      setToast('Liked ✓');
      setTimeout(() => setToast(null), 1000);
    } catch (e) {
      console.warn('[ℹ] Like failed:', e?.message);
    }
  };

  // -------- Lấy similar cho item và mở modal --------
  const openSimilarModal = async (item) => {
    const { item_id, title } = item;
    setModalOpen(true);
    setModalBaseItem({ item_id, title });
    setModalLoading(true);
    setModalError('');
    setModalSimilar([]);

    try {
      await sendEventIfEnabled(item_id, 'click'); // chỉ chạy nếu ENABLE_AUTO_EVENT = true

      const res = await axios.get('http://localhost:5000/similar_items', { params: { item_id, k: 10 } });
      const raw = res.data.similar || [];
      const enriched = await enrichWithPosters(raw);
      setModalSimilar(enriched);
    } catch (e) {
      console.error('[❌ fetch similar error]:', e);
      setModalError('Failed to load similar movies.');
    } finally {
      setModalLoading(false);
    }
  };

  // 🔄 Load Lantern insight/picks ngay khi mở trang (trường hợp đã có session trước đó)
  useEffect(() => {
    fetchLanternProfile();
    fetchLanternPicks();
  }, []);

  return (
    <div style={{ maxWidth: 980, margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', margin: '16px 0' }}>🎬 Movie Recommender</h2>

      {/* --- FORM: nhập user-features --- */}
      <form onSubmit={handleSubmit}>
        <label>
          Gender:
          {/* ⭐ Thêm Unknown (default) */}
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="UNKNOWN">Unknown</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </label>
        <br /><br />

        <label>Age:</label>
        <select value={age} onChange={(e) => setAge(Number(e.target.value))}>
          <option value={-1}>Unknown</option>
          <option value={1}>Under 18</option>
          <option value={18}>18–24</option>
          <option value={25}>25–34</option>
          <option value={35}>35–44</option>
          <option value={45}>45–49</option>
          <option value={50}>50–55</option>
          <option value={56}>56+</option>
        </select>
        <br /><br />

        <label>Occupation:</label>
        <select value={occupation} onChange={(e) => setOccupation(Number(e.target.value))}>
          <option value={-1}>Unknown</option>
          <option value={0}>Other / Not Specified</option>
          <option value={1}>Academic / Educator</option>
          <option value={2}>Artist</option>
          <option value={3}>Clerical / Admin</option>
          <option value={4}>College / Grad Student</option>
          <option value={5}>Customer Service</option>
          <option value={6}>Doctor / Healthcare</option>
          <option value={7}>Executive / Managerial</option>
          <option value={8}>Farmer</option>
          <option value={9}>Homemaker</option>
          <option value={10}>K–12 Student</option>
          <option value={11}>Lawyer</option>
          <option value={12}>Programmer</option>
          <option value={13}>Retired</option>
          <option value={14}>Sales / Marketing</option>
          <option value={15}>Scientist</option>
          <option value={16}>Self-Employed</option>
          <option value={17}>Technician / Engineer</option>
          <option value={18}>Tradesman / Craftsman</option>
          <option value={19}>Unemployed</option>
          <option value={20}>Writer</option>
        </select>
        <br /><br />

        <button type="submit">Get Recommendations</button>
        <button
          type="button"
          onClick={handleRefreshRecommend}
          disabled={!lastQuery}
          style={{ marginLeft: 8 }}
          title={lastQuery ? 'Reload top-K with last input' : 'Submit once to enable'}
        >
          Refresh recommendations
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* ⭐ Toast nhỏ khi like/reset */}
      {toast && (
        <div style={{
          position: 'fixed', bottom: 18, right: 18, background: '#222', color: '#fff',
          padding: '8px 12px', borderRadius: 8, opacity: 0.9
        }}>
          {toast}
        </div>
      )}

      {/* --- HÀNG 2: Lantern insight (bên trái) + Lantern Picks (bên phải) --- */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 16, marginTop: 16 }}>
        {/* Lantern insight */}
        <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8 }}>
          <h4 style={{ marginTop: 0 }}>Lantern insight</h4>
          {lantern ? (
            <>
              <div>Persona: <b>{lantern.persona || '—'}</b>{typeof lantern.u_norm === 'number' && lantern.u_norm > 0 ? ` (‖u‖=${lantern.u_norm.toFixed(3)})` : ''}</div>
              {Array.isArray(lantern.top_genres) && lantern.top_genres.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  Top genres: {lantern.top_genres.map(([g, c]) => `${g}(${c})`).join(', ')}
                </div>
              )}
              {Array.isArray(lantern.recent) && lantern.recent.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  Recent likes/finishes: {lantern.recent.map(r => r.title).join(' · ')}
                </div>
              )}
            </>
          ) : (
            <div style={{ color: '#666' }}>No taste data yet. Like some movies or press 🔮 to see Lantern work.</div>
          )}

          <div style={{ marginTop: 8 }}>
            <button
              onClick={() => { fetchLanternPicks(); fetchLanternProfile(); }}
              style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #ccc' }}
              title="Recommend purely from your interactions"
            >
              🔮 Lantern recommend
            </button>
            <button
              onClick={lanternReset}
              style={{ marginLeft: 8, padding: '6px 10px', borderRadius: 6, border: '1px solid #ccc' }}
              title="Clear your session taste"
            >
              🧹 Reset taste
            </button>
            <button
              onClick={fetchLanternProfile}
              style={{ marginLeft: 8, padding: '6px 10px', borderRadius: 6, border: '1px solid #ccc' }}
              title="Refresh insight"
            >
              🔄 Refresh insight
            </button>
          </div>
        </div>

        {/* Lantern picks (gợi ý theo suy luận) */}
        <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8 }}>
          <h4 style={{ marginTop: 0 }}>Lantern Picks</h4>
          {lanternPicks.length > 0 ? (
            <ul style={{ marginTop: 8 }}>
              {lanternPicks.map((item, i) => (
                <li key={`lantern-${item.item_id}-${i}`} style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '6px 0', cursor: 'pointer' }} title="Click to see similar movies" onClick={() => openSimilarModal(item)}>
                  {item.poster && <img src={item.poster} alt={item.title} style={{ width: 48, borderRadius: 6 }} />}
                  <div><strong>{item.title}</strong> — {Number(item.score).toFixed(3)}</div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleLike(item.item_id); }}
                    style={{ marginLeft: 'auto', padding: '2px 8px', borderRadius: 6, border: '1px solid #ccc', cursor: 'pointer' }}
                    title="Like this movie"
                  >
                    ❤️ Like
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div style={{ color: '#666' }}>No Lantern picks yet — interact with some movies.</div>
          )}
        </div>
      </div>

      {/* --- DANH SÁCH TOP-K theo demographic --- */}
      {recommendations.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <h3>Top Recommendations (Demographic):</h3>
          <ul className="recommendation-list">
            {recommendations.map((item, idx) => {
              const { item_id, title, score, poster } = item;
              return (
                <li
                  key={item_id ?? idx}
                  className="recommendation-item"
                  onClick={() => openSimilarModal(item)}
                  style={{ cursor: 'pointer', display: 'flex', gap: '12px', alignItems: 'center', padding: '8px 0' }}
                  title="Click to see similar movies"
                >
                  {poster && <img src={poster} alt={title} className="poster" style={{ width: 64, borderRadius: 6 }} />}
                  <div>
                    <strong>{title}</strong><br />
                    Score: {Number(score).toFixed(4)}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleLike(item_id); }}
                      style={{ marginLeft: 10, padding: '2px 8px', borderRadius: 6, border: '1px solid #ccc', cursor: 'pointer' }}
                      title="Like this movie"
                    >
                      ❤️ Like
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* --- MODAL: phim liên quan --- */}
      {modalOpen && (
        <div
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, zIndex: 1000
          }}
          onClick={() => setModalOpen(false)}
        >
          <div
            style={{ background: '#fff', borderRadius: 12, width: 'min(720px, 96vw)', padding: 16, maxHeight: '88vh', overflow: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <h4 style={{ margin: 0 }}>
                Similar movies for {modalBaseItem ? `#${modalBaseItem.item_id}` : ''}{modalBaseItem?.title ? ` — ${modalBaseItem.title}` : ''}
              </h4>
              <button onClick={() => setModalOpen(false)} style={{ padding: '6px 10px', borderRadius: 8, border: '1px solid #ccc' }}>
                Close
              </button>
            </div>

            {modalError && <p style={{ color: 'red', marginTop: 12 }}>{modalError}</p>}
            {modalLoading && <p style={{ color: '#555', marginTop: 12 }}>Loading similar…</p>}

            {!modalLoading && !modalError && (
              <ul style={{ marginTop: 12 }}>
                {modalSimilar.map((s, i) => (
                  <li key={`${modalBaseItem?.item_id}-sim-${i}`} style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '6px 0' }}>
                    {s.poster && <img src={s.poster} alt={s.title} style={{ width: 48, borderRadius: 6 }} />}
                    <div>
                      <strong>{s.title}</strong> — {Number(s.score).toFixed(3)}
                    </div>
                  </li>
                ))}
                {modalSimilar.length === 0 && !modalLoading && <li style={{ color: '#666' }}>No similar movies found.</li>}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default RecommendForm;
