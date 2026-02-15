import { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import './ElectionTracker.css';

const DEFAULT_QUERY = '2026 US midterms political and polling news';
const NEWS_CACHE_KEY = 'election_news_cache_v1';

const ElectionNews = () => {
    const { PRIMARY_API_URL, primaryModel } = useApp();
    const [news, setNews] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [query, setQuery] = useState(DEFAULT_QUERY);
    const [searchInput, setSearchInput] = useState('');
    const fetchIdRef = useRef(0);
    const [cacheLoaded, setCacheLoaded] = useState(false);

    const fetchNews = useCallback(async (q) => {
        const searchQuery = (q || query || DEFAULT_QUERY).trim() || DEFAULT_QUERY;
        const thisFetchId = ++fetchIdRef.current;
        setLoading(true);
        setError(null);
        try {
            if (!primaryModel) {
                setError('No model loaded. Load a model in Settings to run AI-driven news search.');
                setNews([]);
                setLoading(false);
                return;
            }
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const modelParam = primaryModel ? `&model=${encodeURIComponent(primaryModel)}` : '';
            const res = await fetch(`${baseUrl}/election/news?query=${encodeURIComponent(searchQuery)}${modelParam}`);
            const data = await res.json();
            if (data?.error) throw new Error(data.error);
            if (!res.ok) throw new Error(data.message || 'Failed to load news');
            const articles = data.articles || [];
            setQuery(searchQuery);
            setSearchInput(searchQuery);
            setNews((prev) => {
                if (thisFetchId !== fetchIdRef.current) return prev;
                if (articles.length > 0) return articles;
                return prev.length > 0 ? prev : articles;
            });
            if (articles.length > 0) {
                try {
                    localStorage.setItem(NEWS_CACHE_KEY, JSON.stringify({
                        query: searchQuery,
                        articles,
                        savedAt: Date.now()
                    }));
                } catch (e) {
                    console.warn('Failed to cache news results', e);
                }
            }
        } catch (e) {
            console.error('News fetch error', e);
            if (thisFetchId === fetchIdRef.current) {
                setError(e.message || 'Could not load news.');
                setNews([]);
            }
        } finally {
            if (thisFetchId === fetchIdRef.current) setLoading(false);
        }
    }, [query, PRIMARY_API_URL, primaryModel]);

    const onSearch = () => {
        const q = searchInput.trim() || DEFAULT_QUERY;
        setSearchInput(q);
        fetchNews(q);
    };

    useEffect(() => {
        if (cacheLoaded) return;
        try {
            const cachedRaw = localStorage.getItem(NEWS_CACHE_KEY);
            if (cachedRaw) {
                const cached = JSON.parse(cachedRaw);
                if (cached?.articles?.length) {
                    setNews(cached.articles);
                    setQuery(cached.query || DEFAULT_QUERY);
                    setSearchInput(cached.query || '');
                    setLoading(false);
                }
            }
        } catch (e) {
            console.warn('Failed to read cached news', e);
        } finally {
            setCacheLoaded(true);
        }
    }, [cacheLoaded]);

    useEffect(() => {
        if (!cacheLoaded) return;
        if (news.length > 0) return;
        if (!primaryModel) return;
        fetchNews(DEFAULT_QUERY);
    }, [cacheLoaded, news.length, primaryModel, fetchNews]);

    return (
        <div className="news-container">
            <div className="election-header" style={{ marginBottom: '1rem', background: 'transparent', padding: '0' }}>
                <h3 style={{ margin: 0, marginBottom: '0.75rem' }}>2026 Midterms &amp; Polling News</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
                    <input
                        type="text"
                        value={searchInput}
                        onChange={(e) => setSearchInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && onSearch()}
                        placeholder="Search for news (e.g. Pennsylvania senate polls, 2026 governor race)"
                        style={{
                            flex: '1',
                            minWidth: '200px',
                            padding: '0.5rem 0.75rem',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-primary)',
                            fontSize: '0.9rem'
                        }}
                    />
                    <button
                        className="election-tab"
                        onClick={onSearch}
                        disabled={loading}
                        style={{ border: '1px solid var(--border-color)' }}
                    >
                        {loading ? 'Searching…' : 'Search'}
                    </button>
                    <button
                        className="election-tab"
                        onClick={() => fetchNews(DEFAULT_QUERY)}
                        disabled={loading}
                        style={{ border: '1px solid var(--border-color)' }}
                    >
                        Default
                    </button>
                    <button
                        className="election-tab"
                        onClick={() => fetchNews(query || DEFAULT_QUERY)}
                        disabled={loading}
                        style={{ border: '1px solid var(--border-color)' }}
                    >
                        Update
                    </button>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem', marginBottom: 0 }}>
                    AI-driven web search. Ask the election AI to suggest a search, then run it here.
                </p>
            </div>

            {error && (
                <div className="error-message" style={{ marginBottom: '1rem' }}>{error}</div>
            )}

            {loading && news.length === 0 ? (
                <div className="loading-spinner">Searching…</div>
            ) : news.length === 0 ? (
                <div className="loading-message">No articles found. Try a different search or click Default.</div>
            ) : (
                <div className="news-grid">
                    {news.map((item, i) => (
                        <a
                            key={i}
                            href={item.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="news-card"
                            style={{ textDecoration: 'none', color: 'inherit' }}
                        >
                            <div className="news-source">{item.source}</div>
                            <div className="news-title">{item.title}</div>
                            {item.snippet && <div className="news-snippet">{item.snippet}</div>}
                        </a>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ElectionNews;
