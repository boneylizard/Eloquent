import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { isAvatarVideoUrl } from '../utils/characterAvatars';

/**
 * Renders a character avatar as image or video.
 * Chat/editor: muted loop. Call mode: parent controls audio, pause, restart.
 */
const CharacterAvatarMedia = forwardRef(function CharacterAvatarMedia(
  {
    url,
    alt = 'Character',
    className = '',
    style,
    onError,
    videoKey,
    callMode = false,
    /** 'contain' = full frame visible (letterbox). 'cover' = fill box, may crop. */
    fit = 'cover',
    playbackPaused = false,
    playbackMuted = true,
    restartToken = 0,
  },
  ref
) {
  const videoRef = useRef(null);
  const isVideo = isAvatarVideoUrl(url);

  const playVideo = async () => {
    const el = videoRef.current;
    if (!el) return;
    try {
      await el.play();
    } catch (_) {}
  };

  const pauseVideo = () => {
    const el = videoRef.current;
    if (!el) return;
    try {
      el.pause();
    } catch (_) {}
  };

  const restartVideo = async () => {
    const el = videoRef.current;
    if (!el) return;
    try {
      el.currentTime = 0;
    } catch (_) {}
    await playVideo();
  };

  useImperativeHandle(
    ref,
    () => ({
      play: playVideo,
      pause: pauseVideo,
      restart: restartVideo,
      isPaused: () => videoRef.current?.paused ?? true,
      isVideo: () => isVideo,
    }),
    [isVideo]
  );

  useEffect(() => {
    if (!isVideo || !videoRef.current) return;
    const el = videoRef.current;
    el.muted = callMode ? playbackMuted : true;
    if (playbackPaused) {
      pauseVideo();
    } else {
      playVideo();
    }
  }, [isVideo, url, videoKey, callMode, playbackPaused, playbackMuted]);

  useEffect(() => {
    if (!isVideo || restartToken === 0) return;
    restartVideo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restartToken, isVideo]);

  if (!url) return null;

  const fitClass =
    fit === 'contain'
      ? 'block h-auto w-auto max-h-full max-w-full object-contain object-center'
      : 'object-cover object-center';

  if (isVideo) {
    const videoClass = [className, fitClass].filter(Boolean).join(' ');
    return (
      <video
        ref={videoRef}
        key={videoKey ?? url}
        src={url}
        className={videoClass}
        style={style}
        autoPlay
        loop
        muted={callMode ? playbackMuted : true}
        playsInline
        disablePictureInPicture
        onError={onError}
      />
    );
  }

  const imgClass = [className, fitClass].filter(Boolean).join(' ');
  return (
    <img
      src={url}
      alt={alt}
      className={imgClass}
      style={style}
      onError={onError}
    />
  );
});

export default CharacterAvatarMedia;
