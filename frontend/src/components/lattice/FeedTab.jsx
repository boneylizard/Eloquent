import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Heart, MessageSquare, Share2, Send, Loader2, Newspaper, Volume2, Volume1, Sparkles, Pin, Trash2, CheckSquare, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePool } from '../../contexts/PoolContext';
import { matchModelName } from '../../utils/modelDisplayNames';

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function FeedPostCard({ post, onReply, onDelete, onLike, likedIds }) {
  const [replyText, setReplyText] = useState('');
  const [isReplying, setIsReplying] = useState(false);
  const [showReplyInput, setShowReplyInput] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const modelInfo = post.is_user ? null : matchModelName(post.character_snapshot?.generated_by);
  const { playFeedPostTTS, togglePinPost } = usePool();
  const liked = likedIds?.has(post.id);
  const likeCount = post.likes?.length || 0;
  const replyCount = post.replies?.length || 0;
  const reactions = post.reactions || [];
  const emojiCounts = useMemo(() => {
    const counts = {};
    for (const r of reactions) {
      counts[r.emoji] = (counts[r.emoji] || 0) + 1;
    }
    return counts;
  }, [reactions]);

  const handlePlayTTS = useCallback(async () => {
    if (isPlaying) return;
    setIsPlaying(true);
    const character = post.character_snapshot || { name: post.character_name, voice_id: null, avatar: post.character_avatar };
    await playFeedPostTTS(character, post.content);
    setIsPlaying(false);
  }, [post, playFeedPostTTS, isPlaying]);

  const handleSubmitReply = useCallback(async () => {
    if (!replyText.trim()) return;
    setIsReplying(true);
    await onReply(post.id, replyText, post.section);
    setReplyText('');
    setIsReplying(false);
  }, [replyText, post.id, post.section, onReply]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitReply();
    }
  };

  const isJealousyPost = post.mood === 'jealousy' || post.is_jealousy;
  const isIcebreakerPost = post.is_icebreaker;

  return (
    <div className={`bg-card border rounded-xl overflow-hidden transition-all hover:border-primary/20 ${post.is_user ? 'border-primary/20 bg-primary/5' : ''} ${isJealousyPost ? 'border-l-2 border-red-500/20' : ''} ${isIcebreakerPost ? 'border-l-2 border-sky-500/30' : ''}`}>
      <div className="p-4 space-y-3">
        <div className="flex items-center gap-3">
          {post.character_avatar ? (
            <img src={post.character_avatar} alt={post.character_name} className="w-10 h-10 rounded-full object-cover border border-border" />
          ) : (
            <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center text-sm font-bold text-muted-foreground">
              {post.character_name?.[0] || '?'}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-sm font-semibold">{post.character_name}</span>
              {post.is_user ? (
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold bg-primary/10 text-primary">You</span>
              ) : modelInfo ? (
                <span className={`text-[8px] px-1.5 py-0.5 rounded font-semibold ${modelInfo.color}`}>
                  {modelInfo.short}
                </span>
              ) : null}
              {isIcebreakerPost && (
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold bg-gradient-to-r from-sky-500/20 to-purple-500/20 text-sky-400 border border-sky-500/20">
                  Icebreaker
                </span>
              )}
              {isJealousyPost && (
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold bg-red-500/10 text-red-400">
                  Drama
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <span>{timeAgo(post.created_at)}</span>
              {post.section && (
                <span className="px-1.5 py-0.5 rounded bg-muted text-[9px]">{post.section}</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <p className="text-sm leading-relaxed flex-1 min-w-0">{post.content}</p>
          {!post.is_user && (
            <button
              onClick={handlePlayTTS}
              disabled={!post.character_snapshot?.voice_id || isPlaying}
              className="shrink-0 w-6 h-6 rounded flex items-center justify-center hover:bg-muted transition-colors disabled:opacity-30 mt-0.5"
              title={post.character_snapshot?.voice_id ? `Play with ${post.character_name}'s voice` : 'No voice set'}
            >
              {isPlaying ? <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" /> : <Volume2 className="w-3 h-3 text-muted-foreground" />}
            </button>
          )}
        </div>

        {Object.keys(emojiCounts).length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {Object.entries(emojiCounts).map(([emoji, count]) => (
              <span key={emoji} className="inline-flex items-center gap-1 text-[10px] bg-muted/50 px-1.5 py-0.5 rounded-full">
                {emoji} <span className="text-muted-foreground text-[9px]">{count}</span>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 pt-1 border-t border-border/30">
          <button
            onClick={() => onLike(post.id)}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${liked ? 'text-red-400' : 'text-muted-foreground hover:text-red-400 hover:bg-muted'}`}
          >
            <Heart className={`w-3.5 h-3.5 ${liked ? 'fill-red-400' : ''}`} />
            {likeCount > 0 && <span>{likeCount}</span>}
            {likeCount === 0 && 'Like'}
          </button>
          <button
            onClick={() => setShowReplyInput(!showReplyInput)}
            className={`flex items-center gap-1 text-xs transition-colors px-2 py-1 rounded ${
              showReplyInput ? 'text-primary bg-primary/10' : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            {replyCount > 0 ? replyCount : 'Reply'}
          </button>
          <button
            onClick={() => { navigator.clipboard.writeText(window.location.href); }}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded hover:bg-muted"
          >
            <Share2 className="w-3.5 h-3.5" />
            Share
          </button>
          {post.is_user && (
            <button
              onClick={() => togglePinPost(post.id)}
              className={`flex items-center gap-1 text-xs transition-colors px-2 py-1 rounded ${post.pinned ? 'text-amber-400 bg-amber-500/10' : 'text-muted-foreground hover:text-amber-400 hover:bg-muted'}`}
              title={post.pinned ? 'Unpin' : 'Pin to top'}
            >
              <Pin className={`w-3.5 h-3.5 ${post.pinned ? 'fill-amber-400' : ''}`} />
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(post.id)}
              className="flex items-center gap-1 text-xs text-muted-foreground/40 hover:text-red-400 transition-colors px-1.5 py-0.5"
              title="Delete post"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>

        {post.replies?.length > 0 && (
          <div className="space-y-2 pt-1 border-t border-border/20">
            {post.replies.map((reply, i) => {
              const isCharInteraction = reply.is_character_interaction;
              const replyAuthor = reply.character_name || 'Character';
              const replyModelInfo = isCharInteraction ? matchModelName(reply.generated_by) : null;
              return (
              <div key={reply.id || i} className={`flex items-start gap-2 pl-3 border-l-2 ${
                reply.is_user ? 'border-primary/30' :
                isCharInteraction ? 'border-amber-500/40' :
                'border-muted'
              } ${isCharInteraction ? 'bg-amber-500/[0.02]' : ''}`}>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    {isCharInteraction && (
                      <span className="text-[8px] text-amber-500/60 font-medium mr-0.5">↩</span>
                    )}
                    <span className={`text-[11px] font-medium ${reply.is_user ? 'text-primary' : isCharInteraction ? 'text-amber-400/90' : ''}`}>
                      {replyAuthor}
                    </span>
                    {!reply.is_user && replyAuthor !== 'You' && (
                      <span className={`text-[8px] px-1.5 py-0.5 rounded font-semibold ${replyModelInfo?.color || 'bg-muted text-muted-foreground'}`}>
                        {replyModelInfo?.short || '?'}
                      </span>
                    )}
                    <span className="text-[9px] text-muted-foreground">{timeAgo(reply.created_at)}</span>
                  </div>
                  <p className="text-xs mt-0.5 leading-relaxed">{reply.content}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <button
                      onClick={() => onLike(post.id, reply.id)}
                      className={`flex items-center gap-1 text-[9px] transition-colors ${likedIds?.has(`${post.id}_${reply.id}`) ? 'text-red-400' : 'text-muted-foreground/50 hover:text-red-400'}`}
                    >
                      <Heart className={`w-2.5 h-2.5 ${likedIds?.has(`${post.id}_${reply.id}`) ? 'fill-red-400' : ''}`} />
                      {reply.likes?.length > 0 && <span>{reply.likes.length}</span>}
                    </button>
                  </div>
                </div>
              </div>
            );
            })}
          </div>
        )}

        {showReplyInput && (
          <div className="flex items-center gap-2 pt-1">
            <input
              value={replyText}
              onChange={e => setReplyText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Write a reply..."
              className="flex-1 h-9 text-xs bg-muted border rounded-lg px-3 outline-none focus:border-primary/50 transition-colors"
            />
            <Button size="sm" onClick={handleSubmitReply} disabled={!replyText.trim() || isReplying} className="h-9 w-9 p-0">
              {isReplying ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function FeedTab() {
  const { feedPosts, fetchFeed, replyToPost, createUserFeedPost, runTickForAll, poolCharacters, deleteFeedPost, toggleLikePost, likedIds, togglePinPost } = usePool();
  const [postText, setPostText] = useState('');
  const [postSection, setPostSection] = useState('');
  const [isPosting, setIsPosting] = useState(false);
  const [batchMode, setBatchMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [deleting, setDeleting] = useState(false);

  const sortedPosts = useMemo(() => {
    const posts = feedPosts || [];
    const pinned = posts.filter(p => p.pinned);
    const unpinned = posts.filter(p => !p.pinned);
    return [...pinned, ...unpinned];
  }, [feedPosts]);

  useEffect(() => {
    fetchFeed();
  }, [fetchFeed]);

  const handleReply = useCallback(async (postId, text, section) => {
    await replyToPost(postId, text, section);
  }, [replyToPost]);

  const handleCreatePost = useCallback(async () => {
    if (!postText.trim() || isPosting) return;
    setIsPosting(true);
    const result = await createUserFeedPost(postText, postSection);
    if (result) {
      setPostText('');
      setPostSection('');
    }
    setIsPosting(false);
  }, [postText, postSection, isPosting, createUserFeedPost]);

  const handleDelete = useCallback(async (postId) => {
    await deleteFeedPost(postId);
  }, [deleteFeedPost]);

  const toggleSelect = useCallback((postId) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(postId)) next.delete(postId);
      else next.add(postId);
      return next;
    });
  }, []);

  const handleBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    if (!window.confirm(`Delete ${selectedIds.size} posts?`)) return;
    setDeleting(true);
    try {
      const apiUrl = window.location.origin + '/api';
      await fetch(`${apiUrl}/lattice/feed-posts/batch-delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ post_ids: Array.from(selectedIds) }),
      });
      setSelectedIds(new Set());
      await fetchFeed();
    } catch (e) {
      console.warn('Batch delete failed:', e);
    }
    setDeleting(false);
  }, [selectedIds, fetchFeed]);

  const handleDeleteAll = useCallback(async () => {
    if (!window.confirm('Delete ALL feed posts? This cannot be undone.')) return;
    setDeleting(true);
    try {
      const apiUrl = window.location.origin + '/api';
      await fetch(`${apiUrl}/lattice/feed-posts/all`, { method: 'DELETE' });
      await fetchFeed();
    } catch (e) {
      console.warn('Delete all failed:', e);
    }
    setDeleting(false);
  }, [fetchFeed]);

  const handleDeleteCharacter = useCallback(async (charId) => {
    if (!window.confirm(`Delete all posts from this character?`)) return;
    setDeleting(true);
    try {
      const apiUrl = window.location.origin + '/api';
      await fetch(`${apiUrl}/lattice/feed-posts/character/${charId}`, { method: 'DELETE' });
      await fetchFeed();
    } catch (e) {
      console.warn('Delete character posts failed:', e);
    }
    setDeleting(false);
  }, [fetchFeed]);

  return (
    <div className="space-y-3 max-w-2xl mx-auto pb-8">
      <div>
        <h2 className="text-lg font-bold flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-amber-500" />
          Feed
        </h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Post to the feed and AI women will respond. Reply to their posts to start conversations.
        </p>
      </div>

      <div className="bg-card border rounded-xl p-4 space-y-3">
        <textarea
          value={postText}
          onChange={e => setPostText(e.target.value)}
          placeholder={`What's on your mind? ${poolCharacters.length > 0 ? `The women will see this and respond.` : ''}`}
          rows={2}
          className="w-full text-sm bg-muted border rounded-lg px-3 py-2 outline-none resize-none focus:border-primary/50 transition-colors"
        />
        <div className="flex items-center justify-between gap-2">
          <select
            value={postSection}
            onChange={e => setPostSection(e.target.value)}
            className="h-8 text-xs bg-muted border rounded-md px-2 outline-none"
          >
            <option value="">All sections</option>
            <option value="Intimate">Intimate</option>
            <option value="Erotic">Erotic</option>
            <option value="Experimental">Experimental</option>
          </select>
          <Button
            onClick={handleCreatePost}
            disabled={!postText.trim() || isPosting}
            size="sm"
            className="gap-1.5"
          >
            {isPosting ? (
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Posting... (AI women responding)</>
            ) : (
              <><Sparkles className="w-3.5 h-3.5" /> Post</>
            )}
          </Button>
        </div>
      </div>

      {feedPosts.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <img src="/logos/mirrorlogosamle2.webp" alt="" className="w-16 h-16 object-contain mb-3 opacity-30" />
          <p className="text-sm text-muted-foreground">No posts yet.</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Be the first to post! AI women will respond.</p>
          <button onClick={fetchFeed} className="text-xs text-primary hover:underline mt-2">Refresh</button>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => { setBatchMode(!batchMode); setSelectedIds(new Set()); }}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors ${batchMode ? 'bg-primary/10 border-primary/30 text-primary' : 'border-border text-muted-foreground hover:text-foreground'}`}
            >
              {batchMode ? <CheckSquare className="w-3.5 h-3.5" /> : <Square className="w-3.5 h-3.5" />}
              {batchMode ? 'Cancel selection' : 'Select posts'}
            </button>
            {batchMode && selectedIds.size > 0 && (
              <button
                onClick={handleBatchDelete}
                disabled={deleting}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete {selectedIds.size} selected
              </button>
            )}
            {!batchMode && (
              <button
                onClick={handleDeleteAll}
                disabled={deleting}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-red-500/20 text-red-400/60 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50 ml-auto"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete all
              </button>
            )}
          </div>
          {sortedPosts.map(post => (
            <div key={post.id} className="relative">
              {batchMode && (
                <button
                  onClick={() => toggleSelect(post.id)}
                  className={`absolute top-3 left-3 z-10 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${selectedIds.has(post.id) ? 'bg-primary border-primary text-primary-foreground' : 'border-border bg-card hover:border-primary/50'}`}
                >
                  {selectedIds.has(post.id) && <CheckSquare className="w-3 h-3" />}
                </button>
              )}
              <FeedPostCard
                post={post}
                onReply={handleReply}
                onDelete={handleDelete}
                onLike={toggleLikePost}
                likedIds={likedIds}
              />
            </div>
          ))}
        </>
      )}
    </div>
  );
}
