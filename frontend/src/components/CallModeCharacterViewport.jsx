import React, { useCallback, useEffect, useRef } from 'react';

import { Sparkles, Loader2 } from 'lucide-react';

import CharacterAvatarMedia from './CharacterAvatarMedia';

import {

  clampCallModeFullscreenZoom,

  shouldApplyCallModeFullscreenTransform,

} from '../utils/characterAvatars';

import { cn } from '@/lib/utils';



/** Portrait frame — max box; media uses object-contain (no forced aspect crop). */

export const CALL_PORTRAIT_FRAME_CLASS =

  'relative w-[min(88vw,420px)] h-[min(68dvh,640px)] max-h-[calc(100dvh-15rem)]';



/** True viewport fill — parent shell is fixed inset-0; this fills the shell. */

export const CALL_FULLSCREEN_FRAME_CLASS =

  'absolute inset-0 h-[100dvh] w-[100vw] max-h-[100dvh] max-w-[100vw] aspect-auto rounded-none';



/**

 * Call-mode character display: portrait window, full frame visible (contain),

 * black letterboxing when aspect ratios differ.

 */

export default function CallModeCharacterViewport({

  avatarUrl,

  isVideoAvatar,

  characterName,

  isSpeaking,

  isTtsActive,

  avatarVideoRef,

  videoKey,

  videoPlaybackPaused,

  videoPlaybackMuted,

  videoRestartToken,

  avatarListLength,

  onToggleVideo,

  onError,

  onPointerActivity,

  onPointerLeave,

  aboutHotspotEnabled = false,

  showAboutHotspot = true,

  showCharacterName = true,

  onAboutClick,

  isAboutLoading = false,

  fullscreen = false,

  mediaZoom = 1,

  mediaPanX = 0,

  mediaPanY = 0,

  onMediaPanChange,

  children,

}) {

  const panDragRef = useRef({ active: false, startX: 0, startY: 0, originX: 0, originY: 0 });
  const pendingPanRef = useRef(null);
  const panRafRef = useRef(null);

  const effectiveZoom = clampCallModeFullscreenZoom(mediaZoom);

  const applyTransform =

    fullscreen && shouldApplyCallModeFullscreenTransform(effectiveZoom, mediaPanX, mediaPanY);



  const handlePanPointerDown = useCallback(

    (e) => {

      if (!fullscreen || !onMediaPanChange || !applyTransform) return;

      if (e.button !== 0) return;

      if (e.target.closest('[data-avatar-video-controls]')) return;

      if (e.target.closest('[data-character-about-hotspot]')) return;

      panDragRef.current = {

        active: true,

        startX: e.clientX,

        startY: e.clientY,

        originX: mediaPanX,

        originY: mediaPanY,

      };

      e.currentTarget.setPointerCapture?.(e.pointerId);

    },

    [fullscreen, onMediaPanChange, applyTransform, mediaPanX, mediaPanY]

  );

  const clearPanCursor = useCallback(() => {
    if (document.body.style.cursor === 'grabbing') {
      document.body.style.cursor = '';
    }
  }, []);

  useEffect(() => () => clearPanCursor(), [clearPanCursor]);



  const handlePanPointerMove = useCallback(

    (e) => {

      if (!panDragRef.current.active || !onMediaPanChange) return;

      const dx = e.clientX - panDragRef.current.startX;

      const dy = e.clientY - panDragRef.current.startY;

      const nextX = Math.max(-40, Math.min(40, panDragRef.current.originX + dx * 0.12));

      const nextY = Math.max(-40, Math.min(40, panDragRef.current.originY + dy * 0.12));

      pendingPanRef.current = { x: nextX, y: nextY };
      if (panRafRef.current != null) return;
      panRafRef.current = requestAnimationFrame(() => {
        panRafRef.current = null;
        const pending = pendingPanRef.current;
        if (!pending) return;
        pendingPanRef.current = null;
        onMediaPanChange(pending.x, pending.y);
      });

    },

    [onMediaPanChange]

  );



  const handlePanPointerEnd = useCallback((e) => {

    if (!panDragRef.current.active) return;

    panDragRef.current.active = false;
    if (panRafRef.current != null) {
      cancelAnimationFrame(panRafRef.current);
      panRafRef.current = null;
    }
    const pending = pendingPanRef.current;
    pendingPanRef.current = null;
    if (pending && onMediaPanChange) onMediaPanChange(pending.x, pending.y);

    e.currentTarget.releasePointerCapture?.(e.pointerId);
    clearPanCursor();

  }, [onMediaPanChange, clearPanCursor]);



  const frameClass = cn(

    fullscreen ? CALL_FULLSCREEN_FRAME_CLASS : CALL_PORTRAIT_FRAME_CLASS,

    'overflow-hidden',

    fullscreen

      ? 'border-0 shadow-none bg-black'

      : 'rounded-[2rem] sm:rounded-[2.75rem] border border-white/20 shadow-[0_24px_80px_rgba(0,0,0,0.55)] bg-black',

    'transition-shadow duration-500',

    !fullscreen && isSpeaking && 'shadow-[0_0_36px_rgba(56,189,248,0.1),0_24px_80px_rgba(0,0,0,0.55)] ring-1 ring-cyan-400/15',

    fullscreen && isSpeaking && 'ring-1 ring-cyan-400/10',

    'group cursor-default',

  );



  const mediaTransform = applyTransform

    ? `translate3d(${mediaPanX}%, ${mediaPanY}%, 0) scale(${effectiveZoom})`

    : undefined;



  return (

    <div

      data-call-portrait-frame

      className={frameClass}

      onMouseMove={onPointerActivity}

      onMouseEnter={onPointerActivity}

      onMouseLeave={onPointerLeave}

      onPointerDown={handlePanPointerDown}

      onPointerMove={handlePanPointerMove}

      onPointerUp={handlePanPointerEnd}

      onPointerCancel={handlePanPointerEnd}

      title={undefined}

      onClick={(e) => {

        if (!isVideoAvatar) return;

        if (e.target.closest('[data-avatar-video-controls]')) return;

        if (e.target.closest('[data-character-about-hotspot]')) return;

        onToggleVideo?.();

      }}

    >

      <div className="absolute inset-0 flex items-center justify-center bg-black">

        {!avatarUrl && (

          <span className="text-7xl font-bold text-white/25 sm:text-8xl">

            {characterName?.charAt(0)?.toUpperCase() || 'A'}

          </span>

        )}

      </div>



      {avatarUrl ? (

        <>

          {isVideoAvatar && (

            <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden" aria-hidden>

              <CharacterAvatarMedia

                url={avatarUrl}

                alt=""

                fit="cover"

                className="h-full w-full scale-110 object-cover opacity-40 blur-2xl saturate-125"

                videoKey={`${videoKey}-bg`}

                callMode

                playbackPaused={videoPlaybackPaused}

                playbackMuted

                restartToken={videoRestartToken}

              />

              <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/45" />

            </div>

          )}



          <div

            className={cn(

              'absolute inset-0 z-[1] flex min-h-0 min-w-0 items-center justify-center',

              fullscreen ? 'p-0' : 'p-2 sm:p-3'

            )}

          >

            <div

              className="flex min-h-0 min-w-0 max-h-full max-w-full items-center justify-center"

              style={

                mediaTransform

                  ? { transform: mediaTransform, transformOrigin: 'center center', willChange: 'transform' }

                  : undefined

              }

            >

              <CharacterAvatarMedia

                ref={isVideoAvatar ? avatarVideoRef : undefined}

                url={avatarUrl}

                alt={characterName || 'Character'}

                fit="contain"

                className="drop-shadow-[0_8px_32px_rgba(0,0,0,0.4)]"

                videoKey={videoKey}

                callMode={isVideoAvatar}

                playbackPaused={videoPlaybackPaused}

                playbackMuted={videoPlaybackMuted}

                restartToken={videoRestartToken}

                onError={onError}

              />

            </div>

          </div>



          {fullscreen ? (

            <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[3] h-1/4 bg-gradient-to-t from-black/40 to-transparent" />

          ) : null}

        </>

      ) : null}



      {characterName && showCharacterName && (

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[4] px-5 pb-4 pt-10 text-center opacity-0 transition-opacity duration-200 group-hover:opacity-100">

          <p className="text-xl font-semibold tracking-wide text-white drop-shadow-md sm:text-2xl">

            {characterName}

          </p>

        </div>

      )}



      {aboutHotspotEnabled && showAboutHotspot && (

        <button

          type="button"

          data-character-about-hotspot

          className={cn(

            'absolute left-1/2 top-[36%] z-[6] flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/20 bg-black/50 text-white/85 backdrop-blur-sm transition-all duration-300',

            'pointer-events-none scale-95 opacity-0 group-hover:pointer-events-auto group-hover:scale-100 group-hover:opacity-100',

            isAboutLoading && 'pointer-events-auto scale-100 opacity-100'

          )}

          disabled={isAboutLoading}

          aria-label={isAboutLoading ? 'Loading character insight' : 'About this character'}

          title="About this character — uses your configured model + prompt"

          onClick={(e) => {

            e.stopPropagation();

            onAboutClick?.();

          }}

        >

          {isAboutLoading ? (

            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />

          ) : (

            <Sparkles className="h-4 w-4" aria-hidden />

          )}

        </button>

      )}



      {children}

    </div>

  );

}


