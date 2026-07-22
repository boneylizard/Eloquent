import React, { useEffect, useState, useRef } from 'react';
import { Clock } from 'lucide-react';

export default function BreakoutTimer({ endTime, onExpire }) {
  const [remaining, setRemaining] = useState('');
  const [severity, setSeverity] = useState('normal');
  const expiredRef = useRef(false);

  useEffect(() => {
    const tick = () => {
      const diff = new Date(endTime).getTime() - Date.now();
      if (diff <= 0) {
        setRemaining('00:00');
        if (!expiredRef.current) {
          expiredRef.current = true;
          onExpire?.();
        }
        return;
      }
      const mins = Math.floor(diff / 60000);
      const secs = Math.floor((diff % 60000) / 1000);
      setRemaining(`${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`);
      if (mins < 1) setSeverity('critical');
      else if (mins < 5) setSeverity('warning');
      else setSeverity('normal');
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [endTime, onExpire]);

  const colors = {
    normal: 'text-white/80',
    warning: 'text-amber-400',
    critical: 'text-red-400 animate-pulse',
  };

  return (
    <div className={`flex items-center gap-1.5 text-xs font-mono font-semibold ${colors[severity]}`}>
      <Clock className="w-3.5 h-3.5" />
      <span>{remaining}</span>
    </div>
  );
}
