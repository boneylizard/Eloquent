import { useState, useEffect, useCallback } from 'react';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Search, ImageOff, RefreshCw, X } from 'lucide-react';
import { getBackendUrl } from '@/config/api';
import { cn } from '@/lib/utils';
import RoomImageGalleryCard from './RoomImageGalleryCard';

export default function RoomImageGalleryModal({ open, onOpenChange, onSelect }) {
  const [images, setImages] = useState([]);
  const [categories, setCategories] = useState([]);
  const [activeCategory, setActiveCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const base = getBackendUrl();
      const [imgRes, catRes] = await Promise.all([
        fetch(`${base}/room-gallery/images`).then(r => { if (!r.ok) throw new Error('Failed to load images'); return r.json(); }),
        fetch(`${base}/room-gallery/categories`).then(r => { if (!r.ok) throw new Error('Failed to load categories'); return r.json(); }),
      ]);
      setImages(imgRes.images || []);
      setCategories(catRes.categories || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      setSearchQuery('');
      setActiveCategory('all');
      setSelectedId(null);
      fetchAll();
    }
  }, [open, fetchAll]);

  const filtered = images.filter(img => {
    const matchCategory = activeCategory === 'all' || img.category_id === activeCategory;
    if (!searchQuery.trim()) return matchCategory;
    const q = searchQuery.toLowerCase();
    return matchCategory && (
      (img.display_name || '').toLowerCase().includes(q) ||
      (img.tags || []).some(t => (t || '').toLowerCase().includes(q))
    );
  });

  const handleSelect = (path) => {
    setSelectedId(images.find(i => i.path === path)?.id);
    onSelect?.(path);
    onOpenChange?.(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-full w-full h-full sm:h-full sm:max-w-full sm:rounded-none flex flex-col p-0 gap-0 border-0">
        <DialogHeader className="p-6 pb-3 shrink-0">
          <div className="flex items-start justify-between gap-4">
            <div>
              <DialogTitle>Room Image Gallery</DialogTitle>
              <DialogDescription>Select from themed gallery or upload custom image</DialogDescription>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onSelect?.(null);
                  onOpenChange?.(false);
                }}
              >
                <ImageOff className="h-4 w-4 mr-1" /> Clear Background
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onOpenChange?.(false)}
              >
                <X className="h-4 w-4 mr-1" /> Close
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="px-6 pb-3 shrink-0 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Search images..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          <div className="flex gap-2 overflow-x-auto pb-1">
            {categories.map(cat => (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className={cn(
                  'flex-shrink-0 rounded-full px-3 py-1 text-xs font-medium border transition-colors',
                  activeCategory === cat.id
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'bg-background text-muted-foreground border-border hover:bg-muted'
                )}
              >
                {cat.name} ({cat.count})
              </button>
            ))}
          </div>
        </div>

        <ScrollArea className="flex-1 px-6 pb-6">
          {loading && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="aspect-video rounded-lg bg-muted animate-pulse" />
              ))}
            </div>
          )}

          {error && !loading && (
            <div className="flex flex-col items-center justify-center h-40 text-muted-foreground gap-2">
              <p className="text-sm">{error}</p>
              <Button variant="outline" size="sm" onClick={fetchAll}>
                <RefreshCw className="h-3 w-3 mr-1" /> Retry
              </Button>
            </div>
          )}

          {!loading && !error && filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center h-40 text-muted-foreground gap-2">
              <ImageOff className="h-8 w-8" />
              {searchQuery || activeCategory !== 'all'
                ? <p className="text-sm">No images match your search</p>
                : <p className="text-sm">No images yet. Generate or upload some!</p>
              }
            </div>
          )}

          {!loading && !error && filtered.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
              {filtered.map(img => (
                <RoomImageGalleryCard
                  key={img.id}
                  image={img}
                  isSelected={selectedId === img.id}
                  onSelect={handleSelect}
                />
              ))}
            </div>
          )}
        </ScrollArea>

        <div className="p-3 border-t shrink-0 text-center">
          <span className="text-xs text-muted-foreground">
            {filtered.length} image{filtered.length !== 1 ? 's' : ''} available
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
