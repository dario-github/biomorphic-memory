// Auto-Recall Hook — Pure Biomorphic Memory (Spreading Activation + PPR)

import { writeFileSync, readFileSync, existsSync, appendFileSync, statSync } from 'node:fs';
import { join, basename } from 'node:path';
import { execSync } from 'node:child_process';

// --- Constants ---
const DATA_DIR = process.env.DATA_DIR || '/tmp';
const MSG_FILE = join(DATA_DIR, 'recall-latest.json');
const RECALL_FILE = join(DATA_DIR, 'auto-recall-dynamic.md');
const LOG_FILE = join(DATA_DIR, 'auto-recall.log');
const FEEDBACK_FILE = join(DATA_DIR, 'recall-feedback.log');
const WORKSPACE = process.env.WORKSPACE || process.cwd();
const MEMORY_DIR = join(WORKSPACE, 'memory');
const MAX_RESULTS = 5;
const MAX_SNIPPET = 400;
const STALE_MS = 30_000;
const SKIP_PATTERNS = [/^(hi|hello|ok|continue|status|update)$/i];
const LOG_MAX_BYTES = 10 * 1024 * 1024; // 10MB log rotation

// --- Spreading Activation Engine ---
const SA_TIMEOUT_MS = 1500;
const SA_MAX_CONSECUTIVE_TIMEOUTS = 3;
let _saTimeoutCount = 0;

function spreadingActivationSearch(query: string): SearchResult[] {
  if (_saTimeoutCount >= SA_MAX_CONSECUTIVE_TIMEOUTS) {
    safeAppendLog(LOG_FILE, {
      ts: new Date().toISOString(), event: 'sa_degraded',
      reason: `${_saTimeoutCount} consecutive timeouts, skipping`,
    });
    return [];
  }

  try {
    const t0 = Date.now();
    const escaped = query.replace(/'/g, "'\\''").slice(0, 500);
    const scriptPath = join(__dirname, '../src/spreading_activation.py');
    const output = execSync(
      `python3 ${scriptPath} '${escaped}' --json --top-k 10`,
      { 
        encoding: 'utf-8', 
        timeout: SA_TIMEOUT_MS, 
        stdio: ['pipe', 'pipe', 'pipe'],
        env: process.env 
      }
    ).trim();
    const elapsed = Date.now() - t0;

    if (!output) {
      _saTimeoutCount = 0;
      return [];
    }

    const results: any[] = JSON.parse(output);
    _saTimeoutCount = 0;

    return results
      .filter((r: any) => r.node_id !== '__LOW_CONFIDENCE__' && r.score > 0.1)
      .slice(0, MAX_RESULTS * 2)
      .map((r: any) => ({
        path: r.path.replace(WORKSPACE + '/memory/', 'memory/').replace(WORKSPACE + '/', ''),
        snippet: '',
        score: r.score,
        source: 'biomorphic' as const,
        _sa: {
          activation: r.activation,
          semantic: r.semantic,
          pagerank: r.pagerank,
          q_value: r.q_value,
          elapsed,
        },
      }));
  } catch (e: any) {
    const isTimeout = e.killed || e.signal === 'SIGTERM' || (e.message || '').includes('timed out');
    if (isTimeout) {
      _saTimeoutCount++;
      safeAppendLog(LOG_FILE, {
        ts: new Date().toISOString(), event: 'sa_timeout', count: _saTimeoutCount,
      });
    } else {
      _saTimeoutCount = 0;
      safeAppendLog(LOG_FILE, {
        ts: new Date().toISOString(), event: 'sa_error', error: (e.message || '').slice(0, 200),
      });
    }
    return [];
  }
}

// --- Result processing ---
interface SearchResult {
  path: string;
  snippet: string;
  score: number;
  source: 'biomorphic';
  _sa?: Record<string, any>;
}

function fillSnippets(results: SearchResult[]): SearchResult[] {
  for (const r of results) {
    if (r.snippet) continue;
    try {
      const fullPath = r.path.startsWith('/') ? r.path : join(WORKSPACE, r.path);
      r.snippet = readFileSync(fullPath, 'utf-8').slice(0, MAX_SNIPPET).replace(/\n{3,}/g, '\n\n');
    } catch {
      r.snippet = '(file not accessible)';
    }
  }
  return results;
}

// --- Layered scoring ---
const EVERGREEN_PATTERNS = [/\/rules\//i, /\/entities\//i, /\/people\//i];

function isEvergreen(path: string): boolean {
  return EVERGREEN_PATTERNS.some(p => p.test(path));
}

function getRecency(filePath: string): number {
  try {
    const fullPath = filePath.startsWith('/') ? filePath : join(WORKSPACE, filePath);
    const mtime = statSync(fullPath).mtimeMs;
    const ageDays = (Date.now() - mtime) / 86_400_000;
    return Math.exp(-ageDays / 30);
  } catch { return 0.3; }
}

let _refCounts: Map<string, number> | null = null;
let _refCountsLoadedAt = 0;

function getRefCounts(): Map<string, number> {
  if (_refCounts && Date.now() - _refCountsLoadedAt < 60_000) return _refCounts;
  _refCounts = new Map();
  _refCountsLoadedAt = Date.now();
  try {
    if (!existsSync(FEEDBACK_FILE)) return _refCounts;
    const lines = readFileSync(FEEDBACK_FILE, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        for (const path of (entry.referenced || [])) {
          _refCounts.set(path, (_refCounts.get(path) || 0) + 1);
        }
      } catch {}
    }
  } catch {}
  return _refCounts;
}

function getEnergy(path: string): number {
  const refs = getRefCounts();
  const count = refs.get(path) || 0;
  return Math.log2(count + 1) / 10;
}

function scoreAndFilter(results: SearchResult[]): SearchResult[] {
  const refCounts = getRefCounts();
  const totalEnergy = [...refCounts.values()].reduce((a, b) => a + b, 0);
  const coldStart = refCounts.size === 0 || totalEnergy < 20;

  const maxScore = Math.max(...results.map(r => r.score), 0.001);

  const scored = results.map(r => {
    const relevance = r.score / maxScore;
    const recency = getRecency(r.path);
    const energy = getEnergy(r.path);
    const evergreen = isEvergreen(r.path) ? 0.1 : 0;

    if (relevance < 0.15) return null;

    const finalScore = coldStart
      ? r.score * 0.7 + recency * 0.3 + evergreen
      : r.score * 0.55 + recency * 0.2 + energy * 0.25 + evergreen;

    return { ...r, score: finalScore };
  }).filter((r): r is NonNullable<typeof r> => r !== null);

  return scored.sort((a, b) => b.score - a.score).slice(0, MAX_RESULTS);
}

// --- Output formatting ---
function formatRecall(results: SearchResult[]): string {
  const lines = ['## 🧠 Auto-Recall\n'];
  for (const r of results) {
    lines.push(`**${r.path}** 🌊 (score: ${r.score.toFixed(3)})`);
    const clean = r.snippet.replace(/\n{3,}/g, '\n\n').replace(/\|/g, '¦');
    lines.push(clean);
    lines.push('');
  }
  lines.push(`_Recalled at ${new Date().toISOString()}_`);
  return lines.join('\n');
}

function safeAppendLog(file: string, entry: object) {
  try {
    if (existsSync(file) && statSync(file).size > LOG_MAX_BYTES) {
      writeFileSync(file, '');
    }
    appendFileSync(file, JSON.stringify(entry) + '\n');
  } catch {}
}

// --- Implicit feedback ---
function extractFileTitle(filePath: string): string {
  try {
    const fullPath = filePath.startsWith('/') ? filePath : join(WORKSPACE, filePath);
    const content = readFileSync(fullPath, 'utf-8').slice(0, 500);
    const match = content.match(/^#\s+(.+)/m);
    return match ? match[1].trim() : '';
  } catch { return ''; }
}

function checkReferenced(filePath: string, agentReply: string): boolean {
  const replyLower = agentReply.toLowerCase();
  const name = basename(filePath, '.md');
  
  if (replyLower.includes(filePath.toLowerCase()) || replyLower.includes(name.toLowerCase())) {
    return true;
  }

  const title = extractFileTitle(filePath);
  if (title && title.length > 2 && replyLower.includes(title.toLowerCase())) {
    return true;
  }

  return false;
}

// --- Main handler ---
const handler = async (event: any) => {

  // === Part 1: Capture latest message ===
  if (event.type === 'message' && event.action === 'preprocessed') {
    try {
      const content = event.context?.bodyForAgent || event.context?.body || '';
      if (content.length > 2) {
        writeFileSync(MSG_FILE, JSON.stringify({
          content: content.slice(0, 2000),
          sessionKey: event.sessionKey || '',
          ts: Date.now(),
        }));
      }
    } catch {}
    return;
  }

  // === Part 2: Biomorphic memory search ===
  if (event.type === 'agent' && event.action === 'bootstrap') {
    try {
      const files = event.context?.bootstrapFiles;
      if (!Array.isArray(files)) return;
      if (!existsSync(MSG_FILE)) return;

      const raw = readFileSync(MSG_FILE, 'utf-8');
      const msg = JSON.parse(raw);
      if (Date.now() - msg.ts > STALE_MS) return;

      const content = (msg.content || '').trim();
      if (!content || content.length < 3) return;
      if (SKIP_PATTERNS.some(p => p.test(content))) return;

      const saResults = fillSnippets(spreadingActivationSearch(content));
      const merged = scoreAndFilter(saResults);

      safeAppendLog(LOG_FILE, {
        ts: new Date().toISOString(),
        msg: content.slice(0, 100),
        hits: saResults.length,
        final: merged.length,
        paths: merged.map(r => r.path),
        scores: merged.map(r => ({ path: r.path, score: +r.score.toFixed(3), ...(r as any)._sa })),
      });

      if (merged.length === 0) return;

      const recallContent = formatRecall(merged);
      writeFileSync(RECALL_FILE, recallContent);
      files.push({
        name: 'auto-recall-dynamic.md',
        path: RECALL_FILE,
        content: recallContent,
      });
    } catch {}
  }

  // === Part 3: Implicit feedback ===
  if (event.type === 'message' && event.action === 'sent') {
    try {
      const agentReply = event.context?.content || '';
      if (!agentReply || agentReply.length < 10) return;
      if (agentReply === 'NO_REPLY' || agentReply === 'HEARTBEAT_OK') return;

      if (!existsSync(LOG_FILE)) return;
      const logContent = readFileSync(LOG_FILE, 'utf-8').trim();
      const logLines = logContent.split('\n').filter(Boolean);
      if (logLines.length === 0) return;

      const lastEntry = JSON.parse(logLines[logLines.length - 1]);
      const injectedPaths: string[] = lastEntry.paths || [];

      if (injectedPaths.length === 0) return;

      const referenced: string[] = [];
      const unreferenced: string[] = [];

      for (const path of injectedPaths) {
        if (checkReferenced(path, agentReply)) {
          referenced.push(path);
        } else {
          unreferenced.push(path);
        }
      }

      safeAppendLog(FEEDBACK_FILE, {
        ts: new Date().toISOString(),
        injected: injectedPaths,
        referenced,
        unreferenced,
        replyLen: agentReply.length,
      });
    } catch {}
  }
};

export default handler;
