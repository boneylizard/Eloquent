import React, { useState, useMemo } from 'react';
import { Chess } from 'chess.js';

const PIECE_SYMBOLS = {
  w: { k: '♔', q: '♕', r: '♖', b: '♗', n: '♘', p: '♙' },
  b: { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' },
};

function coordToSquare(file, rank) {
  return String.fromCharCode(97 + file) + (8 - rank);
}

/**
 * Minimal chess board for React 18. Uses chess.js for state.
 * Click a square to select, then click another to move (or click same to deselect).
 */
export default function SimpleChessBoard({ position, onMove, disabled }) {
  const [selected, setSelected] = useState(null);

  const { board, game } = useMemo(() => {
    const g = new Chess(position);
    return { board: g.board(), game: g };
  }, [position]);

  const handleClick = (file, rank) => {
    if (disabled) return;
    const square = coordToSquare(file, rank);
    if (!selected) {
      const piece = board[rank][file];
      if (piece && piece.color === 'w') setSelected(square);
      return;
    }
    if (selected === square) {
      setSelected(null);
      return;
    }
    const moveOpts = { from: selected, to: square };
    const piece = game.get(selected);
    if (piece?.type === 'p' && (rank === 0 || rank === 7)) moveOpts.promotion = 'q';
    const move = game.move(moveOpts);
    setSelected(null);
    if (move && onMove) onMove(selected, square, move);
  };

  return (
    <div className="simple-chess-board inline-block border-2 border-border rounded overflow-hidden select-none">
      <div className="grid grid-cols-8 w-full aspect-square max-w-[min(100%,400px)]">
        {board.map((row, rank) =>
          row.map((piece, file) => {
            const isLight = (rank + file) % 2 === 0;
            const square = coordToSquare(file, rank);
            const isSelected = selected === square;
            const symbol = piece ? PIECE_SYMBOLS[piece.color][piece.type] : '';
            return (
              <button
                key={square}
                type="button"
                className={`
                  w-full aspect-square flex items-center justify-center text-2xl sm:text-3xl md:text-4xl
                  ${isLight ? 'bg-amber-100 dark:bg-amber-900/40' : 'bg-amber-200 dark:bg-amber-800/50'}
                  ${isSelected ? 'ring-2 ring-blue-500 ring-inset' : ''}
                  hover:opacity-90 disabled:opacity-70 disabled:cursor-not-allowed
                `}
                onClick={() => handleClick(file, rank)}
                disabled={disabled}
                aria-label={piece ? `${piece.color} ${piece.type} on ${square}` : `empty ${square}`}
              >
                {symbol}
              </button>
            );
          })
        )}
      </div>
      <div className="flex border-t border-border bg-muted/50 text-xs">
        <span className="w-[12.5%] text-center">a</span>
        <span className="w-[12.5%] text-center">b</span>
        <span className="w-[12.5%] text-center">c</span>
        <span className="w-[12.5%] text-center">d</span>
        <span className="w-[12.5%] text-center">e</span>
        <span className="w-[12.5%] text-center">f</span>
        <span className="w-[12.5%] text-center">g</span>
        <span className="w-[12.5%] text-center">h</span>
      </div>
    </div>
  );
}
