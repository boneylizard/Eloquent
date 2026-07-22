import { Sparkles, Heart, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function RoomImageGalleryCard({ image, isSelected, onSelect, onFavorite, onDelete }) {
  return (
    <div
      className={cn(
        'relative group aspect-video rounded-lg overflow-hidden cursor-pointer border transition-all duration-200',
        isSelected
          ? 'ring-2 ring-primary ring-offset-2'
          : 'border-border hover:ring-2 hover:ring-ring hover:ring-offset-1'
      )}
      onClick={() => onSelect?.(image.path)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onSelect?.(image.path)}
    >
      <img
        src={image.thumbnail_path || image.path}
        alt={image.display_name}
        className="w-full h-full object-cover"
        loading="lazy"
      />

      {image.source === 'generated' && (
        <span className="absolute top-1.5 right-1.5 inline-flex items-center justify-center h-5 w-5 rounded-full bg-black/50">
          <Sparkles className="h-3 w-3 text-yellow-400" />
        </span>
      )}

      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100">
        <button
          type="button"
          className="p-1.5 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
          onClick={(e) => { e.stopPropagation(); onFavorite?.(image.id); }}
          title="Favorite"
        >
          <Heart className="h-4 w-4 text-white" />
        </button>
        {image.source !== 'bundled' && (
          <button
            type="button"
            className="p-1.5 rounded-full bg-white/20 hover:bg-red-400/30 transition-colors"
            onClick={(e) => { e.stopPropagation(); onDelete?.(image.id); }}
            title="Remove from gallery"
          >
            <Trash2 className="h-4 w-4 text-white" />
          </button>
        )}
      </div>

      <div className="absolute bottom-0 left-0 right-0 px-2 py-1 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
        <p className="text-xs text-white truncate">{image.display_name}</p>
      </div>
    </div>
  );
}
