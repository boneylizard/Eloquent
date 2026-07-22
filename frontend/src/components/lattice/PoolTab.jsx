import React, { useCallback, useState, Fragment, useEffect } from 'react';
import { Sparkles, Grid3X3, Users, Activity, Heart, X, MessageSquare, ChevronLeft, User, Newspaper, Info, Terminal } from 'lucide-react';
import { usePool, PoolProvider } from '../../contexts/PoolContext';
import { useApp } from '../../contexts/AppContext';
import IncubatorPanel from './IncubatorPanel';
import DummyRivalsPanel from './DummyRivalsPanel';
import PoolSection from './PoolSection';
import UserProfilePage from './UserProfilePage';
import FeedTab from './FeedTab';
import AboutMirror from './AboutMirror';
import NeuralSexPanel from './NeuralSexPanel';
import StoriesBar from './StoriesBar';
import DMThreads from './DMThreads';
import MirrorTutorial, { isTutorialComplete } from './MirrorTutorial';
import SpeedDatingEvent from './SpeedDatingEvent';
import DeveloperConsole from './DeveloperConsole';

const SECTION_STYLES = {
  Intimate: { active: 'bg-pink-500 text-white shadow-lg shadow-pink-500/30', inactive: 'text-pink-400 hover:text-pink-300 hover:bg-pink-500/10', dot: 'bg-pink-500' },
  Erotic: { active: 'bg-red-500 text-white shadow-lg shadow-red-500/30', inactive: 'text-red-400 hover:text-red-300 hover:bg-red-500/10', dot: 'bg-red-500' },
  Experimental: { active: 'bg-purple-500 text-white shadow-lg shadow-purple-500/30', inactive: 'text-purple-400 hover:text-purple-300 hover:bg-purple-500/10', dot: 'bg-purple-500' },
};

const NAV_ITEMS = [
  { id: 'profile', label: 'My Profile', icon: User },
  { id: 'feed', label: 'Feed', icon: Newspaper },
  { id: 'dms', label: 'DMs', icon: MessageSquare },
  { id: 'pool', label: 'Pool', icon: Heart },
  { id: 'incubator', label: 'Incubator', icon: Sparkles },
  { id: 'dummies', label: 'Rivals', icon: Users },
  { id: 'activity', label: 'Activity', icon: Activity },
  { id: 'dev', label: 'Dev', icon: Terminal },
  { id: 'about', label: 'About', icon: Info },
];

function ActivityPanel() {
  const { agenticActionLog, runTickForAll, tickEnabled, setTickEnabled, isGenerating } = usePool();

  return React.createElement('div', { className: 'space-y-3' },
    React.createElement('div', { className: 'flex items-center justify-between bg-card border rounded-lg p-3' },
      React.createElement('div', { className: 'flex items-center gap-2' },
        React.createElement(Activity, { className: 'w-4 h-4 text-emerald-500' }),
        React.createElement('span', { className: 'text-sm font-semibold' }, 'Agentic Activity')
      ),
      React.createElement('div', { className: 'flex items-center gap-2' },
        React.createElement('label', { className: 'flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer' },
          React.createElement('input', { type: 'checkbox', checked: tickEnabled, onChange: e => setTickEnabled(e.target.checked), className: 'rounded' }),
          ' Auto'
        ),
        React.createElement('button', {
          onClick: runTickForAll,
          disabled: isGenerating,
          className: 'text-xs text-primary hover:underline disabled:opacity-50',
        }, 'Tick all')
      )
    ),
    agenticActionLog.length === 0
      ? React.createElement('div', { className: 'text-center py-12 text-sm text-muted-foreground' },
          React.createElement(Activity, { className: 'w-8 h-8 mx-auto mb-2 opacity-30' }),
          React.createElement('p', null, 'No agentic activity yet.'),
          React.createElement('p', { className: 'text-xs mt-1' }, 'Generate entities and enable auto-tick to begin.')
        )
      : React.createElement('div', { className: 'space-y-1' },
          ...agenticActionLog.map((entry, i) =>
            React.createElement('div', { key: i, className: 'bg-card border rounded-lg p-3 text-xs space-y-1' },
              React.createElement('div', { className: 'flex items-center justify-between' },
                React.createElement('span', { className: 'font-semibold' }, entry.characterName),
                React.createElement('div', { className: 'flex items-center gap-2' },
                  entry.emotional_state ? React.createElement('span', { className: 'text-[10px] text-muted-foreground italic' }, entry.emotional_state) : null,
                  React.createElement('span', { className: 'text-[10px] text-muted-foreground' }, new Date(entry.timestamp).toLocaleTimeString())
                )
              ),
              React.createElement('div', { className: 'flex gap-1.5' },
                React.createElement('span', { className: 'px-1.5 py-0.5 rounded bg-primary/10 text-primary text-[10px] font-medium' }, entry.action),
                entry.target ? React.createElement('span', { className: 'text-muted-foreground' }, '→ ', entry.target) : null
              ),
              entry.content ? React.createElement('p', { className: 'text-muted-foreground/70 line-clamp-2' }, entry.content) : null,
              entry.reasoning ? React.createElement('p', { className: 'text-[10px] text-muted-foreground/50 italic' }, entry.reasoning) : null
            )
          )
        )
  );
}

function PoolContent() {
  const { sections, activeSection, setActiveSection, selectedCharacter, setSelectedCharacter, showNeuralSex, neuralSexCharacter, setShowNeuralSex, stories, fetchStories, viewedStoryIds, markStoryViewed, dmThreads, speedDatingSession, startSpeedDating, selectDMThreadById } = usePool();
  const { pendingDMThreadId, setPendingDMThreadId } = useApp();
  const [subtab, setSubtab] = useState('pool');
  const [showTutorial, setShowTutorial] = useState(() => !isTutorialComplete());

  useEffect(() => {
    if (pendingDMThreadId) {
      setSubtab('dms');
      setPendingDMThreadId(null);
      selectDMThreadById(pendingDMThreadId);
    }
  }, [pendingDMThreadId, setPendingDMThreadId, selectDMThreadById]);

  useEffect(() => { fetchStories(); }, [fetchStories]);

  const handleTutorialNavigate = useCallback((targetTab) => {
    setSubtab(targetTab);
  }, []);

  const elements = [];

  if (showTutorial) {
    elements.push(React.createElement(MirrorTutorial, { key: 'tutorial', onNavigate: handleTutorialNavigate, onComplete: () => setShowTutorial(false) }));
  }

  elements.push(
    React.createElement('div', {
      key: 'main',
      className: 'h-full flex flex-col bg-gradient-to-b from-background to-background/95',
    },
      React.createElement('div', {
        key: 'nav',
        className: 'flex items-center gap-2 px-4 pt-3 pb-1.5 border-b border-border/40',
      },
        React.createElement('span', {
          key: 'title',
          className: 'text-sm font-bold tracking-tight text-foreground/80 mr-2',
        }, React.createElement('img', {
          key: 'title',
          src: '/logos/MirrorAIDating (1).webp',
          alt: 'Mirror AI Dating',
          className: 'h-5 w-auto object-contain',
        })),
        React.createElement('div', {
          key: 'buttons',
          className: 'flex gap-0.5 bg-muted/60 rounded-lg p-0.5',
        },
          NAV_ITEMS.map(item => {
            const Icon = item.icon;
            const isActive = subtab === item.id;
            const unread = item.id === 'dms' ? dmThreads.reduce((sum, t) => sum + (t.unread_count || 0), 0) : 0;
            return React.createElement('button', {
              key: item.id,
              onClick: () => { setSubtab(item.id); setSelectedCharacter(null); },
              className: `flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md transition-all relative ${isActive ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`,
            },
              React.createElement(Icon, { className: 'w-3.5 h-3.5' }),
              item.label,
              unread > 0
                ? React.createElement('span', {
                    className: 'absolute -top-1 -right-1 w-4 h-4 rounded-full bg-primary text-[8px] font-bold text-primary-foreground flex items-center justify-center',
                  }, Math.min(unread, 9))
                : null
            );
          })
        )
      ),
              subtab === 'profile'
                ? React.createElement('div', { key: 'profile', className: 'flex-1 overflow-y-auto px-4 py-4' },
                    React.createElement(UserProfilePage, null)
                  )
                : subtab === 'dms'
                  ? React.createElement('div', { key: 'dms', className: 'flex-1 overflow-y-auto px-4 py-4' },
                      React.createElement(DMThreads, null)
                    )
                  : subtab === 'feed'
          ? React.createElement('div', { key: 'feed', className: 'flex-1 overflow-y-auto px-4 py-4' },
              React.createElement(FeedTab, null)
            )
          : subtab === 'pool'
            ? React.createElement('div', { key: 'pool', className: 'flex-1 flex flex-col min-h-0' },
                React.createElement(StoriesBar, {
                  key: 'stories',
                  stories,
                  viewedStoryIds,
                  onMarkViewed: markStoryViewed,
                  className: 'px-4',
                }),
                React.createElement('div', { key: 'section-tabs', className: 'flex gap-1.5 px-4 pt-1 pb-2 items-center' },
                  ...sections.map(s => {
                    const style = SECTION_STYLES[s];
                    const isActive = activeSection === s;
                    return React.createElement('button', {
                      key: s,
                      onClick: () => setActiveSection(s),
                      className: `flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-full transition-all ${isActive ? style.active : style.inactive}`,
                    },
                      React.createElement('div', { className: `w-1.5 h-1.5 rounded-full ${style.dot}` }),
                      s
                    );
                  }).concat(React.createElement('button', {
                    key: 'speed-date',
                    onClick: startSpeedDating,
                    disabled: speedDatingSession?.active,
                    className: 'ml-auto flex items-center gap-1 px-2.5 py-1.5 text-xs font-semibold rounded-full bg-gradient-to-r from-pink-500/20 to-purple-500/20 text-pink-400 hover:from-pink-500/30 hover:to-purple-500/30 transition-all border border-pink-500/20 disabled:opacity-40',
                  }, '⚡ Speed Date'))
                ),
                React.createElement('div', { key: 'section-content', className: 'flex-1 min-h-0 overflow-y-auto px-4 pb-4' },
                  React.createElement(PoolSection, { section: activeSection })
                )
              )
            : subtab === 'incubator'
              ? React.createElement('div', { key: 'incubator', className: 'flex-1 overflow-y-auto px-4 py-4' },
                  React.createElement(IncubatorPanel, { section: activeSection })
                )
              : subtab === 'dummies'
                ? React.createElement('div', { key: 'dummies', className: 'flex-1 overflow-y-auto px-4 py-4' },
                    React.createElement(DummyRivalsPanel, null)
                  )
                : subtab === 'about'
                  ? React.createElement('div', { key: 'about', className: 'flex-1 overflow-y-auto px-4 py-4' },
                      React.createElement(AboutMirror, null)
                    )
                  : subtab === 'activity'
                  ? React.createElement('div', { key: 'activity', className: 'flex-1 overflow-y-auto px-4 py-4' },
                      React.createElement(ActivityPanel, null)
                    )
                  : subtab === 'dev'
                  ? React.createElement('div', { key: 'dev', className: 'flex-1 overflow-y-auto px-4 py-4' },
                      React.createElement(DeveloperConsole, null)
                    )
                  : null
    )
  );

  if (showNeuralSex && neuralSexCharacter) {
    elements.push(React.createElement(NeuralSexPanel, {
      key: 'neural-sex',
      character: neuralSexCharacter,
      onClose: () => setShowNeuralSex(false),
    }));
  }

  if (speedDatingSession?.active || speedDatingSession?.complete) {
    elements.push(React.createElement(SpeedDatingEvent, {
      key: 'speed-dating',
      onClose: () => {},
    }));
  }

  return React.createElement(Fragment, null, ...elements);
}

export default function PoolTab() {
  return React.createElement(PoolProvider, null,
    React.createElement(PoolContent, null)
  );
}
