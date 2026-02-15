import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ElectionTracker.css';

const PollTrends = ({ data, raceType }) => {
    const isNewTrendFormat = Array.isArray(data) && data.length > 0 && data[0]?.date && (data[0].dem_avg != null || data[0].approve_avg != null);

    if (!data || data.length === 0) {
        return (
            <div className="trends-container">
                <h3>Polling Trends</h3>
                <p>No trend data available for this race. Data is built from stored polls over time.</p>
            </div>
        );
    }

    if (isNewTrendFormat) {
        const isApproval = raceType === 'approval';
        const dataKey1 = isApproval ? 'approve_avg' : 'dem_avg';
        const dataKey2 = isApproval ? 'disapprove_avg' : 'gop_avg';
        const name1 = isApproval ? 'Approve' : 'Democrat';
        const name2 = isApproval ? 'Disapprove' : 'Republican';
        const chartData = data.map((row) => ({
            ...row,
            date: row.date,
            [dataKey1]: row[dataKey1] != null ? Number(row[dataKey1]) : null,
            [dataKey2]: row[dataKey2] != null ? Number(row[dataKey2]) : null
        })).filter((row) => row[dataKey1] != null || row[dataKey2] != null);

        if (chartData.length === 0) {
            return (
                <div className="trends-container">
                    <h3>Polling Trends</h3>
                    <p>No numeric trend points yet. More polls will populate this chart.</p>
                </div>
            );
        }

        return (
            <div className="trends-container">
                <h3>Polling Averages Over Time</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                    Weekly averages (week ending) from stored polls.
                </p>
                <div className="chart-wrapper" style={{ height: 320 }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                            <XAxis dataKey="date" stroke="#888" fontSize={12} />
                            <YAxis domain={[35, 55]} stroke="#888" />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#222', border: '1px solid #444' }}
                                labelStyle={{ color: '#fff' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey={dataKey1} name={name1} stroke="#3b82f6" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey={dataKey2} name={name2} stroke="#ef4444" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        );
    }

    return (
        <div className="trends-container">
            <h3>Polling Trends</h3>
            <p>No trend data available for this race.</p>
        </div>
    );
};

export default PollTrends;
