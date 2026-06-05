/**
 * User message body for /generate with request_purpose "book_chapter_json_outline".
 * Uses the same character + history packing as chat; model must reply with JSON only.
 * @param {string} rawBundle - Notes + uploaded text (may be empty if only direction + chat context).
 * @param {string} [adaptationGoal] - Optional: scenario, tone shift, audience, or rewrite goal for shaping chapters.
 */
export function buildBookChapterJsonOutlineUserMessage(rawBundle, adaptationGoal = '') {
  const raw = String(rawBundle || '').trim();
  const goal = String(adaptationGoal || '').trim();
  const goalBlock = goal
    ? `\n\n--- USER DIRECTION (follow closely) ---\n\n${goal}\n\nWhen building each chapter's "intent", reflect this direction: pacing, emphasis, what to foreground or avoid, genre hooks, POV constraints, or any scenario the user asked for. Still stay consistent with persona and story facts above.`
    : '';

  return `[STRUCTURED OUTPUT TASK — not in-world dialogue]

You are preparing a **book chapter queue** for this app. Stay fully consistent with the Character Persona, chat history, user profile, summaries, and any document context already in this request.

The user supplied **rough notes, messy chapter names, or unstructured fragments** below (may be empty if they only gave a direction). Infer an ordered list of chapters suitable for serial/novella generation. For each chapter:
- "title": short UI title (may be empty only if "intent" alone is self-explanatory).
- "intent": detailed **writer-facing** instructions for that chapter (beats, continuity, tone, must-happen events). This text will later be sent as the user's instructions to the story model — not in-character dialogue.

Reply with **nothing except** one JSON array in this exact shape (no markdown code fences, no commentary before or after):

[{"title":"string","intent":"string"}]

Rules:
- Output MUST be valid JSON that JSON.parse accepts in one shot.
- Use double quotes for keys and string values; escape internal double quotes as \\".
- At least one object; each object needs a non-empty "intent" (preferred) or a clear "title" plus intent.
- Order chapters in the sequence they should be written.
${goalBlock}

--- RAW MATERIAL FROM USER ---

${raw || '(none — derive chapters from conversation context; still obey USER DIRECTION if present.)'}`;
}

/** Extract [{ title, intent }, ...] from model output (allows ```json fences). */
export function parseChapterJsonOutlineFromModel(text) {
  let t = String(text || '').trim();
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) t = fence[1].trim();
  const bracket = t.indexOf('[');
  const bracketEnd = t.lastIndexOf(']');
  if (bracket !== -1 && bracketEnd > bracket) {
    t = t.slice(bracket, bracketEnd + 1);
  }
  try {
    const arr = JSON.parse(t);
    if (!Array.isArray(arr)) {
      return { ok: false, error: 'Model output was not a JSON array.', chapters: null };
    }
    const chapters = arr
      .map((row) => ({
        title: String(row?.title ?? '').trim(),
        intent: String(row?.intent ?? row?.body ?? row?.prompt ?? '').trim(),
      }))
      .filter((r) => r.intent || r.title);
    if (!chapters.length) {
      return { ok: false, error: 'Parsed array had no usable chapters (need title and/or intent).', chapters: null };
    }
    return { ok: true, chapters };
  } catch (e) {
    return { ok: false, error: e?.message ? `Invalid JSON: ${e.message}` : 'Invalid JSON.', chapters: null };
  }
}
