import React, { useState } from 'react';
import axios from 'axios';

// ⭐ Flag: nếu muốn mỗi lần click phim sẽ gửi event lên BE để BE trả lại danh sách Top-K mới,
// hãy bật true. MẶC ĐỊNH = false để KHÔNG auto-refresh top-K khi click.
const ENABLE_AUTO_EVENT = false;

function RecommendForm() {
  const [gender, setGender] = useState('M');
  const [age, setAge] = useState('');
  const [occupation, setOccupation] = useState('');

  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState(null);

  // Nhớ lại input gần nhất để Refresh
  const [lastQuery, setLastQuery] = useState(null); // { gender, age:Number, occupation:Number }

  // Modal Similar
  const [modalOpen, setModalOpen] = useState(false);
  const [modalBaseItem, setModalBaseItem] = useState(null); // { item_id, title }
  const [modalSimilar, setModalSimilar] = useState([]);     // [{item_id,title,score,poster}]
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState('');

  // -------- Helpers --------
  const getPosterURL = async (title) => {
    try {
      const cleanedTitle = title.replace(/\s*\(\d{4}\)$/, '');
      const res = await axios.get('https://api.themoviedb.org/3/search/movie', {
        params: {
          api_key: 'c24381c0a201e1fe30d94c02463bc6ca',
          query: cleanedTitle
        }
      });
      const posterPath = res.data.results[0]?.poster_path;
      return posterPath ? `https://image.tmdb.org/t/p/w200${posterPath}` : null;
    } catch (err) {
      console.error(`[❌ TMDB Error]: Failed to get poster for ${title}`, err);
      return null;
    }
  };

  const enrichWithPosters = async (recs) => {
    // recs có thể là [[item_id,title,score], ...] hoặc [{item_id,title,score}, ...]
    return Promise.all(
      recs.map(async (r) => {
        const [iid, ttl, sc] = Array.isArray(r) ? r : [r.item_id, r.title, r.score];
        const poster = await getPosterURL(ttl);
        return { item_id: iid, title: ttl, score: sc, poster };
      })
    );
  };

  // -------- Submit form -> /recommend --------
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError(null);
      if (age === '' || occupation === '') {
        setError('Please select both Age and Occupation.');
        return;
      }

      const payload = { gender, age: Number(age), occupation: Number(occupation) };
      setLastQuery(payload);

      const res = await axios.post('http://localhost:5000/recommend', payload);
      const enriched = await enrichWithPosters(res.data.recommendations || []);
      setRecommendations(enriched);

      // Reset modal nếu đang mở
      setModalOpen(false);
      setModalBaseItem(null);
      setModalSimilar([]);
      setModalError('');
    } catch (err) {
      console.error('[❌ Network Error]:', err);
      setError('Something went wrong. Please try again.');
    }
  };

  // -------- Refresh recommendations với lastQuery --------
  const handleRefreshRecommend = async () => {
    if (!lastQuery) return;
    try {
      setError(null);
      const res = await axios.post('http://localhost:5000/recommend', lastQuery);
      const enriched = await enrichWithPosters(res.data.recommendations || []);
      setRecommendations(enriched);
      // Không đụng modal
    } catch (err) {
      console.error('[❌ Refresh Error]:', err);
      setError('Refresh failed. Please try again.');
    }
  };

  // -------- (Tuỳ chọn) gửi event khi user click item --------
  const sendEventIfEnabled = async (item_id, event = 'click') => {
    if (!ENABLE_AUTO_EVENT) return;
    try {
      const res = await axios.post('http://localhost:5000/event', {
        user_id: 'guest',
        item_id,
        event
      });
      if (Array.isArray(res.data?.recommendations)) {
        const enriched = await enrichWithPosters(res.data.recommendations);
        setRecommendations(enriched);
      }
    } catch (e) {
      console.warn('[ℹ] Event not sent or BE no refresh:', e?.message);
    }
  };

  // -------- Like button ở mỗi item --------
  const handleLike = async (item_id) => {
    try {
      const res = await axios.post('http://localhost:5000/event', {
        user_id: 'guest',
        item_id,
        event: 'like',
      });
      if (Array.isArray(res.data?.recommendations)) {
        const enriched = await enrichWithPosters(res.data.recommendations);
        setRecommendations(enriched);
      }
    } catch (e) {
      console.warn('[ℹ] Like failed or no refresh from BE:', e?.message);
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

      const res = await axios.get('http://localhost:5000/similar_items', {
        params: { item_id, k: 10 }
      });
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

  return (
    <div>
      {/* --- FORM: nhập user-features --- */}
      <form onSubmit={handleSubmit}>
        <label>
          Gender:
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </label>
        <br /><br />

        <label>Age:</label>
        <select value={age} onChange={(e) => setAge(Number(e.target.value))}>
          <option value="">-- Select Age Group --</option>
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
          <option value="">-- Select Occupation --</option>
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

      {/* --- DANH SÁCH TOP-K --- */}
      {recommendations.length > 0 && (
        <div>
          <h3>Top Recommendations:</h3>
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
