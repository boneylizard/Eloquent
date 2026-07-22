import React, { useCallback, useMemo, useState } from 'react';
import { ArrowRight, Bot, Check, Loader2, MessageCircle, Save, Send, SlidersHorizontal, Sparkles, UserRound } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { resolveUnifiedRequestRoute } from '../utils/requestRouting';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

const EMPTY_CHARACTER = {
  id: null,
  name: '',
  description: '',
  personality: '',
  background: '',
  model_instructions: '',
  speech_style: '',
  scenario: '',
  first_message: '',
  alternate_greetings: [],
  example_dialogue: [],
  loreEntries: [],
  creator_notes: '',
  post_history_instructions: '',
  tags: [],
  creator: '',
  character_version: '',
  chat_role: 'npc',
  avatar: null,
};

const STARTERS = [
  'A slow-burn fantasy companion with a life beyond the user',
  'A sharp-tongued detective for episodic mysteries',
  'A warm game master who never controls the player',
];

const FOLLOW_UPS = [
  'What relationship should this character have with the user?',
  'What should they want—and what should they refuse to do?',
  'What makes their voice unmistakably theirs?',
  'What kind of scene should the first conversation begin inside?',
];

const FIELD_GROUPS = [
  ['name', 'Name', 'input'],
  ['description', 'Description', 'textarea'],
  ['personality', 'Personality summary', 'textarea'],
  ['background', 'Background', 'textarea'],
  ['speech_style', 'Voice and speech', 'textarea'],
  ['scenario', 'Scenario', 'textarea'],
  ['model_instructions', 'Roleplay instructions', 'textarea'],
  ['first_message', 'Opening message', 'textarea'],
];

const normaliseCharacter = (value = {}) => ({
  ...EMPTY_CHARACTER,
  ...value,
  example_dialogue: Array.isArray(value.example_dialogue) ? value.example_dialogue : [],
  loreEntries: Array.isArray(value.loreEntries) ? value.loreEntries : [],
  alternate_greetings: Array.isArray(value.alternate_greetings) ? value.alternate_greetings : [],
  tags: Array.isArray(value.tags) ? value.tags : [],
});

const CharacterCreatorStudio = ({ onSave, onOpenFullEditor, onCancel }) => {
  const {
    PRIMARY_API_URL,
    primaryModel,
    primaryIsAPI,
    settings,
    storageHydrated,
  } = useApp();
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Tell me who you want to meet. A sentence is enough; we can discover the rest together.',
    },
  ]);
  const [input, setInput] = useState('');
  const [draft, setDraft] = useState(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('build');
  const [revisionCount, setRevisionCount] = useState(0);

  const userMessages = useMemo(
    () => messages.filter((message) => message.role === 'user'),
    [messages],
  );

  const route = useMemo(() => resolveUnifiedRequestRoute({
    primaryModel,
    primaryIsAPI,
    settings,
    requestPurpose: draft ? 'refine_character' : 'create_character',
  }), [draft, primaryIsAPI, primaryModel, settings]);

  const updateDraftField = useCallback((field, value) => {
    setDraft((current) => normaliseCharacter({ ...current, [field]: value }));
  }, []);

  const runBuilder = useCallback(async (text) => {
    const prompt = text.trim();
    if (!prompt || working) return;
    if (!route.effectiveModel) {
      setError('Select or load a text model before asking Mirid to build a character.');
      return;
    }

    const nextUserMessages = [...userMessages, { role: 'user', content: prompt }];
    setMessages((current) => [...current, { role: 'user', content: prompt }]);
    setInput('');
    setWorking(true);
    setError('');

    try {
      const endpoint = draft ? '/character/refine-generated' : '/character/generate-from-conversation';
      const body = draft
        ? {
            character_json: draft,
            feedback: prompt,
            original_messages: nextUserMessages,
            model_name: route.effectiveModel,
            selected_model: route.selectedModel,
            frontend_round_robin_enabled: route.autoEnabled,
            request_purpose: 'refine_character',
            gpu_id: 0,
          }
        : {
            messages: nextUserMessages,
            analysis: {},
            model_name: route.effectiveModel,
            selected_model: route.selectedModel,
            frontend_round_robin_enabled: route.autoEnabled,
            use_api: primaryIsAPI,
            gpu_id: 0,
            conversation_id: `character-builder-${Date.now()}`,
          };
      const response = await fetch(`${PRIMARY_API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !['success', 'partial'].includes(result.status) || !result.character_json) {
        throw new Error(result.detail || result.error || 'The selected model could not shape a character from that description.');
      }

      const nextDraft = normaliseCharacter(result.character_json);
      const nextRevision = revisionCount + 1;
      setDraft(nextDraft);
      setRevisionCount(nextRevision);
      setMessages((current) => [...current, {
        role: 'assistant',
        content: draft
          ? `I’ve revised ${nextDraft.name || 'the character'} without discarding the parts you kept. ${FOLLOW_UPS[nextRevision % FOLLOW_UPS.length]}`
          : `I’ve made a first draft of ${nextDraft.name || 'your character'}. ${FOLLOW_UPS[0]} You can also edit every field directly in Review.`,
      }]);
    } catch (builderError) {
      setError(builderError.message);
      setMessages((current) => [...current, {
        role: 'assistant',
        content: 'I couldn’t complete that revision. Your existing draft is untouched.',
        error: true,
      }]);
    } finally {
      setWorking(false);
    }
  }, [PRIMARY_API_URL, draft, primaryIsAPI, revisionCount, route, userMessages, working]);

  const saveDraft = useCallback(() => {
    if (!draft?.name?.trim() || !storageHydrated) return;
    onSave({
      ...normaliseCharacter(draft),
      created_at: draft.created_at || new Date().toISOString(),
    });
  }, [draft, onSave, storageHydrated]);

  return (
    <div className="space-y-4 pb-24">
      <div className="flex flex-col gap-3 rounded-2xl border border-border/70 bg-card/60 p-5 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Optional model-assisted draft</p>
          <h2 className="mt-1 text-2xl font-semibold">Build a first draft with Mirid</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Describe the idea in ordinary language. Your selected model drafts a card that you can inspect, rewrite or move into the full manual editor.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button variant="outline" onClick={() => onOpenFullEditor(draft || EMPTY_CHARACTER)}>
            <SlidersHorizontal className="mr-2 h-4 w-4" />Write manually
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Character Studio could not complete that turn</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid min-h-[640px] gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(380px,0.95fr)]">
        <div className="flex min-h-[600px] flex-col overflow-hidden rounded-2xl border border-border/70 bg-card/60">
          <div className="border-b p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <MessageCircle className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Build together</h3>
              </div>
              <Badge variant="outline">{route.effectiveModel || 'No model selected'}</Badge>
            </div>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto p-4">
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}>
                {message.role === 'assistant' && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
                    <Sparkles className="h-4 w-4" />
                  </div>
                )}
                <div className={`max-w-[84%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${message.role === 'user' ? 'bg-primary text-primary-foreground' : message.error ? 'bg-destructive/10 text-destructive' : 'bg-muted/70'}`}>
                  {message.content}
                </div>
              </div>
            ))}
            {working && (
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />Shaping the card and checking its voice…
              </div>
            )}
          </div>

          {userMessages.length === 0 && (
            <div className="flex flex-wrap gap-2 px-4 pb-3">
              {STARTERS.map((starter) => (
                <button key={starter} type="button" onClick={() => runBuilder(starter)} className="rounded-full border border-border/70 px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
                  {starter}
                </button>
              ))}
            </div>
          )}

          <div className="border-t p-4">
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    runBuilder(input);
                  }
                }}
                placeholder={draft ? 'Change their voice, history, relationship, boundaries…' : 'Describe the character you have in mind…'}
                className="min-h-[52px] resize-none"
              />
              <Button size="icon" className="h-[52px] w-[52px] shrink-0" onClick={() => runBuilder(input)} disabled={!input.trim() || working} aria-label="Send to character builder">
                {working ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground">Each message runs your selected model. Hosted endpoints may charge for the generation.</p>
          </div>
        </div>

        <div className="rounded-2xl border border-border/70 bg-card/60 p-4">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="build">Preview</TabsTrigger>
              <TabsTrigger value="review">Review card</TabsTrigger>
            </TabsList>

            <TabsContent value="build" className="mt-4 space-y-4">
              {draft ? (
                <>
                  <div className="rounded-xl border border-border/60 bg-background/50 p-5 text-center">
                    <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-primary/15 text-2xl font-semibold text-primary">
                      {draft.name?.trim()?.charAt(0) || <UserRound className="h-7 w-7" />}
                    </div>
                    <h3 className="mt-3 text-xl font-semibold">{draft.name || 'Unnamed character'}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{draft.description || 'The character’s core will appear here.'}</p>
                    <div className="mt-3 flex flex-wrap justify-center gap-1">
                      {draft.personality?.split(/[,;]+/).slice(0, 4).map((trait) => trait.trim()).filter(Boolean).map((trait) => <Badge key={trait} variant="secondary">{trait}</Badge>)}
                    </div>
                  </div>
                  <div className="rounded-xl border border-border/60 bg-background/50 p-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Opening scene</p>
                    <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">{draft.first_message || 'No opening message yet.'}</p>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {[
                      ['Voice', draft.speech_style || draft.model_instructions],
                      ['Setting', draft.scenario],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-lg border border-border/60 p-3">
                        <p className="text-xs font-medium">{label}</p>
                        <p className="mt-1 line-clamp-4 text-xs leading-relaxed text-muted-foreground">{value || 'Not established yet.'}</p>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="flex min-h-[480px] flex-col items-center justify-center px-8 text-center">
                  <Bot className="h-10 w-10 text-muted-foreground" />
                  <h3 className="mt-4 font-semibold">The card will take shape here</h3>
                  <p className="mt-2 text-sm text-muted-foreground">Begin with an archetype, a relationship, a scene—or simply a feeling.</p>
                </div>
              )}
            </TabsContent>

            <TabsContent value="review" className="mt-4 max-h-[720px] space-y-4 overflow-y-auto pr-1">
              {!draft ? (
                <p className="py-12 text-center text-sm text-muted-foreground">Create a first draft before reviewing its fields.</p>
              ) : (
                <>
                  <Alert>
                    <Check className="h-4 w-4" />
                    <AlertTitle>Nothing is hidden</AlertTitle>
                    <AlertDescription>These are the instructions Mirid will actually use. Edit any field before saving.</AlertDescription>
                  </Alert>
                  {FIELD_GROUPS.map(([field, label, kind]) => (
                    <div key={field} className="space-y-1.5">
                      <label htmlFor={`builder-${field}`} className="text-sm font-medium">{label}</label>
                      {kind === 'input' ? (
                        <Input id={`builder-${field}`} value={draft[field] || ''} onChange={(event) => updateDraftField(field, event.target.value)} />
                      ) : (
                        <Textarea id={`builder-${field}`} value={draft[field] || ''} onChange={(event) => updateDraftField(field, event.target.value)} className="min-h-[96px]" />
                      )}
                    </div>
                  ))}
                  <Button variant="outline" className="w-full" onClick={() => onOpenFullEditor(draft)}>
                    Open lore, dialogue and avatars <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {draft && (
        <div className="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-background/95 px-4 py-3 shadow-[0_-4px_24px_rgba(0,0,0,0.15)] backdrop-blur-md md:static md:rounded-xl md:border md:shadow-none">
          <div className="mx-auto flex max-w-4xl flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-muted-foreground">Draft {revisionCount} · Save now or continue refining through conversation.</p>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setActiveTab('review')}>Review instructions</Button>
              <Button onClick={saveDraft} disabled={!draft.name?.trim() || !storageHydrated}>
                <Save className="mr-2 h-4 w-4" />Save and chat
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CharacterCreatorStudio;
