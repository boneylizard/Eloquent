import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
  Cell,
} from 'recharts';
import { Button } from './ui/button';
import {
  TrendingUp,
  Wallet,
  BarChart3,
  History,
  Trophy,
  RefreshCw,
  Zap,
  ChevronRight,
} from 'lucide-react';

function getBackendUrl() {
  const port = parseInt(localStorage.getItem('Eloquent-backend-port') || '8000', 10);
  return `http://localhost:${port}`;
}

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'tournament', label: 'Strategy Tournament', icon: <Trophy className="w-4 h-4" /> },
  { id: 'monte-carlo', label: 'Monte Carlo', icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'trades', label: 'Trade History', icon: <History className="w-4 h-4" /> },
  { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-4 h-4" /> },
];

export default function MarketSimTab() {
  const { PRIMARY_API_URL, primaryModel } = useApp();
  const baseUrl = PRIMARY_API_URL || getBackendUrl();

  const [activeSubTab, setActiveSubTab] = useState('dashboard');
  const [portfolio, setPortfolio] = useState(null);
  const [sp500Current, setSp500Current] = useState(null);
  const [sp500History, setSp500History] = useState([]);
  const [trades, setTrades] = useState([]);
  const [tournament, setTournament] = useState(null);
  const [snapshots, setSnapshots] = useState([]);
  const [quotes, setQuotes] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tournamentRunning, setTournamentRunning] = useState(false);
  const [tournamentResult, setTournamentResult] = useState(null);

  const fetchPortfolio = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/portfolio`, { cache: 'no-store' });
      const data = await res.json();
      setPortfolio(data.portfolio);
      setSp500Current(data.sp500_current);
      setError(null);
    } catch (e) {
      setError(e?.message || 'Failed to load portfolio');
    }
  }, [baseUrl]);

  const fetchTrades = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/trades?limit=50`, { cache: 'no-store' });
      const data = await res.json();
      setTrades(data.trades || []);
    } catch (_) {}
  }, [baseUrl]);

  const fetchTournament = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/tournament/latest`, { cache: 'no-store' });
      const data = await res.json();
      setTournament(data.tournament);
    } catch (_) {}
  }, [baseUrl]);

  const fetchSp500History = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/sp500/history?days=30`, { cache: 'no-store' });
      const data = await res.json();
      setSp500History(data.data || []);
    } catch (_) {}
  }, [baseUrl]);

  const fetchSnapshots = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/snapshots?limit=30`, { cache: 'no-store' });
      const data = await res.json();
      setSnapshots(data.snapshots || []);
    } catch (_) {}
  }, [baseUrl]);

  const resetPortfolioToStartingCapital = useCallback(async () => {
    if (
      typeof window !== 'undefined' &&
      !window.confirm(
        'Reset portfolio to starting cash only (no positions)? Trade and tournament history stay in the database.'
      )
    ) {
      return;
    }
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/market-sim/portfolio/reset`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Reset failed');
      setPortfolio(data.portfolio);
      setSp500Current(data.sp500_current);
      await fetchSnapshots();
    } catch (e) {
      setError(e?.message || 'Reset failed');
    }
  }, [baseUrl, fetchSnapshots]);

  const fetchQuotes = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/market-sim/quotes?symbols=SPY,AAPL,^GSPC`, { cache: 'no-store' });
      const data = await res.json();
      setQuotes(data.quotes || {});
    } catch (_) {}
  }, [baseUrl]);

  const runTournament = async (executeTrade = false) => {
    setTournamentRunning(true);
    setTournamentResult(null);
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/market-sim/tournament/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ execute_trade: executeTrade, model: primaryModel || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.error || 'Tournament failed');
      setTournamentResult(data);
      await fetchTournament();
      await fetchPortfolio();
      await fetchTrades();
    } catch (e) {
      setError(e?.message || 'Tournament run failed');
    } finally {
      setTournamentRunning(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    fetchTrades();
    fetchTournament();
    fetchSp500History();
    fetchSnapshots();
    fetchQuotes();
    const t = setInterval(() => {
      fetchPortfolio();
      fetchQuotes();
    }, 60000);
    return () => clearInterval(t);
  }, [fetchPortfolio, fetchTrades, fetchTournament, fetchSp500History, fetchSnapshots, fetchQuotes]);

  const chartData = React.useMemo(() => {
    const snaps = [...(snapshots || [])].reverse();
    if (!snaps.length) return [];
    const baseValue = snaps[0]?.total_value || 250000;
    const baseSp = snaps[0]?.sp500_value;
    return snaps.map((s, i) => ({
      date: s.created_at?.slice(0, 10) || `Day ${i}`,
      portfolio: s.total_value,
      portfolioPct: baseValue ? ((s.total_value / baseValue - 1) * 100).toFixed(2) : 0,
      sp500: s.sp500_value,
      sp500Pct: baseSp && s.sp500_value ? ((s.sp500_value / baseSp - 1) * 100).toFixed(2) : null,
    }));
  }, [snapshots]);

  const strategyLeaderboard = React.useMemo(() => {
    const r = tournament?.results || tournament?.tournament?.results || {};
    return Object.entries(r)
      .map(([id, v]) => ({
        id,
        name: v.analysis?.strategy_name || id,
        expectedValue: v.analysis?.expected_value,
        meanReturn: v.analysis?.mean_return,
        sharpe: v.analysis?.sharpe_ratio,
        sharpeVsRf: v.analysis?.sharpe_excess_vs_4pct_rf,
        maxDrawdown: v.analysis?.max_drawdown,
        winRate: v.analysis?.win_rate,
        winRateNote: v.analysis?.win_rate_note,
        beatSp500: v.analysis?.beat_sp500_prob,
        beatSp500Note: v.analysis?.beat_sp500_note,
      }))
      .sort((a, b) => (b.sharpe ?? -999) - (a.sharpe ?? -999));
  }, [tournament]);

  const formatCurrency = (n) => {
    if (n == null) return '—';
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(n);
  };

  const formatPct = (n) => {
    if (n == null) return '—';
    return `${(n * 100).toFixed(1)}%`;
  };

  return (
    <div className="flex flex-col h-full" style={{ minHeight: 400 }}>
      <div className="flex items-center justify-between border-b px-4 py-2 bg-card">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Zap className="w-5 h-5 text-primary" />
          Market Simulator
        </h2>
        <p className="text-sm text-muted-foreground">
          AI-managed $250,000 portfolio • Live market data • Monte Carlo tournaments
        </p>
      </div>

      <div className="flex border-b bg-card/50">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveSubTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeSubTab === tab.id ? 'border-b-2 border-primary text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="mx-4 mt-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
          {error}
        </div>
      )}

      <div className="flex-1 overflow-auto p-4">
        {activeSubTab === 'dashboard' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" size="sm" onClick={resetPortfolioToStartingCapital}>
                Reset to starting balance ($250k cash, clear positions)
              </Button>
              <span className="text-xs text-muted-foreground">
                Use if portfolio still shows an old $10k row from SQLite.
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
                  <Wallet className="w-4 h-4" /> Portfolio Value
                </div>
                <div className="text-2xl font-bold">{formatCurrency(portfolio?.total_value)}</div>
                <div className="text-sm text-muted-foreground mt-1">Cash: {formatCurrency(portfolio?.cash)}</div>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
                  <TrendingUp className="w-4 h-4" /> S&P 500
                </div>
                <div className="text-2xl font-bold">{formatCurrency(sp500Current)}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {quotes['^GSPC']?.change_pct != null ? `${quotes['^GSPC'].change_pct >= 0 ? '+' : ''}${quotes['^GSPC'].change_pct.toFixed(2)}% today` : '—'}
                </div>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">Positions</div>
                <div className="text-lg font-semibold">
                  {Object.keys(portfolio?.positions || {}).length || 0} holding(s)
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  {Object.entries(portfolio?.positions || {}).map(([s, q]) => `${s}: ${q}`).join(', ') || '—'}
                </div>
              </div>
            </div>

            {chartData.length > 0 ? (
              <div className="rounded-lg border bg-card p-4">
                <h3 className="text-sm font-medium mb-3">Portfolio vs S&P 500 (snapshots)</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                      <YAxis tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" tickFormatter={(v) => `$${v}`} />
                      <Tooltip formatter={(v) => formatCurrency(v)} />
                      <Area type="monotone" dataKey="portfolio" stroke="hsl(var(--primary))" fill="hsl(var(--primary) / 0.2)" name="Portfolio" />
                      <Area type="monotone" dataKey="sp500" stroke="hsl(var(--muted-foreground))" fill="hsl(var(--muted-foreground) / 0.1)" name="S&P 500" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border bg-card p-6 text-center text-muted-foreground">
                <p>Run a strategy tournament to generate portfolio snapshots and charts.</p>
              </div>
            )}

            <div className="rounded-lg border bg-card p-4">
              <h3 className="text-sm font-medium mb-2">Recent Trades</h3>
              {trades.length > 0 ? (
                <ul className="space-y-2">
                  {trades.slice(0, 5).map((t) => (
                    <li key={t.id} className="flex justify-between items-center text-sm">
                      <span className={t.side === 'buy' ? 'text-green-600' : 'text-red-600'}>
                        {t.side.toUpperCase()} {t.shares} {t.symbol} @ {formatCurrency(t.price)}
                      </span>
                      <span className="text-muted-foreground text-xs">{t.created_at?.slice(0, 16)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-muted-foreground text-sm">No trades yet.</p>
              )}
            </div>
          </div>
        )}

        {activeSubTab === 'tournament' && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 items-center">
              <Button
                onClick={() => runTournament(false)}
                disabled={tournamentRunning}
              >
                {tournamentRunning ? <RefreshCw className="w-4 h-4 animate-spin mr-2" /> : <Zap className="w-4 h-4 mr-2" />}
                Run Monte Carlo Tournament
              </Button>
              <Button variant="outline" onClick={() => runTournament(true)} disabled={tournamentRunning}>
                Run & Execute Winner
              </Button>
            </div>

            {tournamentResult && (
              <div className="rounded-lg border bg-primary/10 p-4">
                <h3 className="font-medium text-primary mb-2">Last Run Result</h3>
                <p className="text-sm mb-2"><strong>Winner:</strong> {tournamentResult.winner_id}</p>
                <p className="text-sm mb-2"><strong>Confidence:</strong> {((tournamentResult.confidence ?? 0) * 100).toFixed(0)}%</p>
                <p className="text-sm mb-2"><strong>Reasoning:</strong> {tournamentResult.reasoning || '—'}</p>
                <p className="text-sm"><strong>Trade Action:</strong> {JSON.stringify(tournamentResult.trade_action || {})}</p>
                {tournamentResult.trade_executed && (
                  <p className="text-sm mt-2 text-green-600">Trade executed: {tournamentResult.trade_executed.symbol}</p>
                )}
              </div>
            )}

            <div className="rounded-lg border bg-card p-4">
              <h3 className="text-sm font-medium mb-2">Strategy Leaderboard</h3>
              <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
                Each row is one strategy run through <strong>10,000 parallel one-year simulations</strong> (not
                one timeline). <strong>Return÷risk</strong> is mean simulated return divided by how spread out
                those outcomes are—higher means more return per unit of scenario uncertainty.{' '}
                <strong>Beat S&amp;P</strong> is the share of scenarios where that strategy beat the simulated
                1× index; it is <strong>not applicable</strong> for Index Fund (identical path).{' '}
                <strong>Win rate</strong> is the share of scenarios with a positive return; cash is always 0%
                because it is modeled as flat.
              </p>
              {tournament?.simulation_calibration && (
                <div className="rounded-md bg-muted/50 border p-2 mb-3 text-xs text-muted-foreground space-y-1">
                  <div className="font-medium text-foreground">Why this run is actionable</div>
                  <p>{tournament.simulation_calibration.design_intent}</p>
                  <p>
                    Simulated index: mean return{' '}
                    <strong>{((tournament.simulation_calibration.simulated_index_mean_return ?? 0) * 100).toFixed(2)}%</strong>
                    , median{' '}
                    <strong>{((tournament.simulation_calibration.simulated_index_median_return ?? 0) * 100).toFixed(2)}%</strong>
                    , positive in{' '}
                    <strong>{((tournament.simulation_calibration.simulated_index_pct_positive_scenarios ?? 0) * 100).toFixed(1)}%</strong>
                    {' '}of scenarios.
                  </p>
                </div>
              )}
              {strategyLeaderboard.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2">Strategy</th>
                        <th className="text-right py-2">Expected value</th>
                        <th className="text-right py-2">Mean return</th>
                        <th className="text-right py-2" title="Mean ÷ std of 10k one-year outcomes">
                          Return÷risk
                        </th>
                        <th className="text-right py-2">Max DD (median)</th>
                        <th className="text-right py-2">Win rate</th>
                        <th className="text-right py-2">Beat S&P</th>
                      </tr>
                    </thead>
                    <tbody>
                      {strategyLeaderboard.map((s) => (
                        <tr key={s.id} className="border-b hover:bg-muted/50">
                          <td className="py-2 font-medium">{s.name}</td>
                          <td className="text-right">{formatCurrency(s.expectedValue)}</td>
                          <td className="text-right">{formatPct(s.meanReturn)}</td>
                          <td className="text-right" title={s.sharpeVsRf != null ? `vs 4% RF variant: ${(s.sharpeVsRf ?? 0).toFixed(3)}` : ''}>
                            {(s.sharpe ?? 0).toFixed(2)}
                          </td>
                          <td className="text-right">{formatPct(s.maxDrawdown)}</td>
                          <td className="text-right" title={s.winRateNote || undefined}>
                            {formatPct(s.winRate)}
                          </td>
                          <td
                            className="text-right"
                            title={s.beatSp500Note || undefined}
                          >
                            {s.beatSp500 == null ? '—' : formatPct(s.beatSp500)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">Run a tournament to see results.</p>
              )}
            </div>
          </div>
        )}

        {activeSubTab === 'monte-carlo' && (
          <div className="space-y-4">
            {tournament?.results ? (
              <>
                <div className="rounded-lg border bg-card p-4">
                  <h3 className="text-sm font-medium mb-2">Monte Carlo Overview</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    10,000 one-year paths • Two half-year regime segments (may shift mid-year) • Recession / Fed
                    drag on drift • Rare stress shocks (~1.2%) • Strategy-specific rules (DCA ramp, momentum,
                    hedged sleeve, etc.)
                  </p>
                </div>
                <div className="rounded-lg border bg-card p-4">
                  <h3 className="text-sm font-medium mb-3">Performance by Regime</h3>
                  {tournament.regime_performance ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                      {Object.entries(tournament.regime_performance).map(([regime, vals]) => (
                        <div key={regime} className="rounded border p-2">
                          <div className="font-medium capitalize">{regime}</div>
                          {vals && typeof vals === 'object' && Object.entries(vals).slice(0, 3).map(([sid, v]) => (
                            <div key={sid} className="text-muted-foreground text-xs">{sid}: {formatCurrency(v)}</div>
                          ))}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-muted-foreground text-sm">Run a tournament first.</p>
                  )}
                </div>
              </>
            ) : (
              <div className="rounded-lg border bg-card p-6 text-center text-muted-foreground">
                <p>Run a strategy tournament to view Monte Carlo results.</p>
              </div>
            )}
          </div>
        )}

        {activeSubTab === 'trades' && (
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium mb-3">Trade History</h3>
            {trades.length > 0 ? (
              <ul className="space-y-3">
                {trades.map((t) => (
                  <li key={t.id} className="border-b pb-2">
                    <div className="flex justify-between items-start">
                      <span className={t.side === 'buy' ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                        {t.side.toUpperCase()} {t.shares} {t.symbol} @ {formatCurrency(t.price)} = {formatCurrency(t.total)}
                      </span>
                      <span className="text-xs text-muted-foreground">{t.created_at}</span>
                    </div>
                    {t.strategy_name && <div className="text-xs text-muted-foreground mt-1">Strategy: {t.strategy_name}</div>}
                    {t.ai_reasoning && <div className="text-xs mt-1 p-2 bg-muted rounded">{t.ai_reasoning.slice(0, 200)}…</div>}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground text-sm">No trades yet.</p>
            )}
          </div>
        )}

        {activeSubTab === 'analytics' && (
          <div className="space-y-4">
            {strategyLeaderboard.length > 0 ? (
              <>
                <div className="rounded-lg border bg-card p-4">
                  <h3 className="text-sm font-medium mb-1">Return ÷ scenario risk</h3>
                  <p className="text-xs text-muted-foreground mb-2">Mean one-year return over std across 10k paths (not textbook daily Sharpe).</p>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={strategyLeaderboard.slice(0, 10)} layout="vertical" margin={{ left: 80 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Bar dataKey="sharpe" fill="hsl(var(--primary))" name="Return÷risk" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="rounded-lg border bg-card p-4">
                  <h3 className="text-sm font-medium mb-2">Data Quality (Nate Silver)</h3>
                  <p className="text-xs text-muted-foreground">
                    Every data point includes confidence: Analyst ratings 0.31, Prediction markets 0.73,
                    Fundamental 0.85, News sentiment 0.38, Insider 0.62, Price history 0.90.
                  </p>
                </div>
              </>
            ) : (
              <div className="rounded-lg border bg-card p-6 text-center text-muted-foreground">
                <p>Run a tournament to view analytics.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
