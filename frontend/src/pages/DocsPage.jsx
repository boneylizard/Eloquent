import React, { useState } from 'react';
import {
  ArrowLeft,
  BookOpenCheck,
  Check,
  Copy,
  ExternalLink,
  HardDrive,
  KeyRound,
  MessageCircleQuestion,
  Mic2,
  ShieldCheck,
  Sparkles,
  Wand2,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import {
  buildProviderCampaignUrl,
  getPublicProviderPromotion,
} from '../config/providerPromotions';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';

const LinkButton = ({ href, children, primary = false }) => (
  <a
    href={href}
    target="_blank"
    rel="noreferrer"
    className={`inline-flex h-9 items-center justify-center gap-2 rounded-md border px-3 text-sm font-medium transition-colors ${primary ? 'border-primary bg-primary text-primary-foreground hover:brightness-110' : 'border-border bg-background hover:bg-accent'}`}
  >
    {children}<ExternalLink className="h-3.5 w-3.5" />
  </a>
);

const GuideCard = ({ icon: Icon, title, children }) => (
  <article className="rounded-2xl border border-border/70 bg-card/60 p-5">
    <div className="flex items-center gap-2">
      <Icon className="h-5 w-5 text-primary" />
      <h2 className="font-semibold">{title}</h2>
    </div>
    <div className="mt-3 space-y-3 text-sm leading-relaxed text-muted-foreground">{children}</div>
  </article>
);

const ProviderPromotionNotice = ({ promotion }) => {
  if (!promotion) return null;
  const isOffer = promotion.status === 'active';
  return (
    <div className="mt-4 rounded-xl border border-primary/25 bg-background/60 px-3 py-3 text-xs leading-relaxed text-muted-foreground">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline">{isOffer ? 'Mirid partner offer' : 'Mirid referral link'}</Badge>
        {isOffer && promotion.promoCode && (
          <code className="rounded bg-muted px-2 py-1 font-semibold text-foreground">{promotion.promoCode}</code>
        )}
      </div>
      <p className="mt-2">
        {isOffer ? promotion.customerBenefit : promotion.referralDisclosure}
      </p>
      {promotion.termsUrl && (
        <a className="mt-1 inline-block text-primary underline-offset-4 hover:underline" href={promotion.termsUrl} target="_blank" rel="noreferrer">
          Provider terms <ExternalLink className="ml-1 inline h-3 w-3" />
        </a>
      )}
    </div>
  );
};

const DocsPage = ({ onLeave, onOpenModelLibrary }) => {
  const { PRIMARY_API_URL } = useApp();
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState('');
  const nanoGptPromotion = getPublicProviderPromotion('nanogpt');
  const openRouterPromotion = getPublicProviderPromotion('openrouter');

  const copyTeachingPrompt = async () => {
    setCopyError('');
    try {
      const response = await fetch(`${PRIMARY_API_URL}/docs/llms.txt`);
      if (!response.ok) throw new Error('The local guide could not be read.');
      const guide = await response.text();
      const prompt = `Teach me how to use Mirid for my goal. Ask what I want to accomplish, then give me the shortest safe path. Explain unfamiliar terms as they arise. Warn me before anything that may spend money or download a large model. Use this product guide as your source of truth:\n\n${guide}`;
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2200);
    } catch (error) {
      setCopyError(error.message);
    }
  };

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 pb-14">
      <header className="relative overflow-hidden rounded-3xl border border-border/70 bg-card/70 p-6 md:p-8">
        <div className="absolute -right-20 -top-28 h-72 w-72 rounded-full bg-primary/10 blur-3xl" />
        <div className="relative">
          <Button variant="ghost" size="sm" onClick={onLeave} className="-ml-2 mb-5">
            <ArrowLeft className="mr-2 h-4 w-4" />Back to Mirid
          </Button>
          <p className="text-[11px] uppercase tracking-[0.25em] text-muted-foreground">Mirid Help Centre</p>
          <h1 className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight md:text-4xl">Learn what you need. Leave the plumbing to us.</h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted-foreground md:text-base">
            Start with a goal, not a settings checklist. This guide explains the shortest working path through local models, hosted APIs, characters, voice, images, and integrations.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button onClick={copyTeachingPrompt}>
              {copied ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
              {copied ? 'Teaching prompt copied' : 'Ask an AI to teach me Mirid'}
            </Button>
            <Button variant="outline" onClick={onOpenModelLibrary}>Open Model Library</Button>
          </div>
          {copyError && <p className="mt-3 text-sm text-destructive">{copyError}</p>}
        </div>
      </header>

      <Alert>
        <MessageCircleQuestion className="h-4 w-4" />
        <AlertTitle>The AI guide is deliberately portable</AlertTitle>
        <AlertDescription>
          The button copies Mirid's full local guide with a teaching prompt. Paste it into any capable assistant; it does not need access to your files, settings, conversations, or API keys.
        </AlertDescription>
      </Alert>

      <section id="first-chat" className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <GuideCard icon={Sparkles} title="Fastest first chat">
          <p>Open the Model Library. Connect OpenRouter for a free starting route, or choose NanoGPT for a larger hosted catalogue. Select a model, return to Chat, and begin.</p>
          <p>Mirid creates the endpoint. You should not need to type provider URLs by hand.</p>
        </GuideCard>
        <GuideCard icon={HardDrive} title="Private local chat">
          <p>Choose Hugging Face, open Mirid's Picks, and download the largest suggested GGUF that comfortably fits your detected VRAM.</p>
          <p>Local models cost nothing per message, but use disk space, RAM, VRAM, and electricity.</p>
        </GuideCard>
        <GuideCard icon={Wand2} title="Character roleplay">
          <p>Import a compatible PNG or JSON card, or let Character Studio interview you and build one. The card defines the character; the selected model performs it.</p>
        </GuideCard>
      </section>

      <section id="providers" className="space-y-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Hosted Models</p>
          <h2 className="mt-1 text-2xl font-semibold">Choose the account that matches your use</h2>
        </div>

        <article className="rounded-3xl border border-primary/40 bg-primary/5 p-5 md:p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-3xl">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-xl font-semibold">NanoGPT Pro</h3>
                <Badge>Mirid recommends</Badge>
                <Badge variant="outline">Frequent personal roleplay</Badge>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                NanoGPT currently advertises 60 million included input-token units each week and 100 included images per day. For ordinary personal roleplay, that is a substantial allowance. Most included text models count at 1×; models marked 2× consume it twice as quickly.
              </p>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                Included models and limits can change. The subscription is for personal, non-commercial use. Mirid keeps NanoGPT's standard API endpoint; eligible calls are covered by the subscription attached to your API-key account and appear as free in NanoGPT usage.
              </p>
              <ProviderPromotionNotice promotion={nanoGptPromotion} />
            </div>
            <div className="flex shrink-0 flex-wrap gap-2 lg:max-w-xs lg:justify-end">
              <LinkButton href={buildProviderCampaignUrl('nanogpt', 'subscription')} primary>Subscribe or manage</LinkButton>
              <LinkButton href={buildProviderCampaignUrl('nanogpt', 'keys')}>Create API key</LinkButton>
              <LinkButton href={buildProviderCampaignUrl('nanogpt', 'credits')}>Add credit</LinkButton>
              <LinkButton href={buildProviderCampaignUrl('nanogpt', 'models')}>Check live limits</LinkButton>
            </div>
          </div>
        </article>

        <div className="grid gap-4 md:grid-cols-2">
          <GuideCard icon={KeyRound} title="OpenRouter">
            <p>Use one key to compare many providers. Mirid places OpenRouter's free router first and shows published token prices where the catalogue supplies them.</p>
            <ProviderPromotionNotice promotion={openRouterPromotion} />
            <div className="flex flex-wrap gap-2">
              <LinkButton href={buildProviderCampaignUrl('openrouter', 'keys')}>Create key</LinkButton>
              <LinkButton href={buildProviderCampaignUrl('openrouter', 'credits')}>Add credit</LinkButton>
              <LinkButton href={buildProviderCampaignUrl('openrouter', 'models')}>Compare models</LinkButton>
            </div>
          </GuideCard>
          <GuideCard icon={ShieldCheck} title="Direct provider APIs">
            <p>OpenAI, Anthropic, Gemini, Mistral, xAI and similar developer APIs are billed separately from consumer chat subscriptions. Mirid reads the models available to your key.</p>
            <p>Use these when you specifically need that provider's models or account controls—not because the setup screen makes them look mandatory.</p>
          </GuideCard>
        </div>
      </section>

      <section id="features" className="grid gap-4 md:grid-cols-2">
        <GuideCard icon={Mic2} title="Voice and transcription">
          <p>Choose a local TTS or STT engine when you want processing on your machine. NanoGPT audio models are also available when its key is configured.</p>
          <p>The first local reply can take longer while the engine wakes and loads what it needs.</p>
        </GuideCard>
        <GuideCard icon={BookOpenCheck} title="Images and SillyTavern">
          <p>Mirid's local image engine uses stable-diffusion.cpp. Open image generation and choose <strong>Find an image model</strong> if no checkpoint is installed. Hugging Face is the primary source; Civitai is optional and only works where Civitai makes its service available.</p>
          <p>Start with a self-contained Safetensors or GGUF checkpoint. Some newer model families also need text encoders or a VAE, so read the model card before a large download.</p>
          <p>Mirid can also serve streaming chat, speech, transcription, and local images to SillyTavern through the Mirid Bridge.</p>
        </GuideCard>
      </section>

      <section id="safety" className="rounded-2xl border border-border/70 bg-card/60 p-5">
        <h2 className="font-semibold">Three rules worth remembering</h2>
        <ol className="mt-3 grid gap-3 text-sm text-muted-foreground md:grid-cols-3">
          <li><strong className="text-foreground">1. Keep keys out of chats.</strong><br />Paste them only into the labelled password fields in Settings.</li>
          <li><strong className="text-foreground">2. Prices are live facts.</strong><br />Check the provider before relying on an allowance or published rate.</li>
          <li><strong className="text-foreground">3. Read the exact error.</strong><br />Repeatedly retrying a paid request is not troubleshooting.</li>
        </ol>
      </section>
    </div>
  );
};

export default DocsPage;
