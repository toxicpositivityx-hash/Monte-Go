/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { 
  Dices, 
  History as HistoryIcon, 
  Settings as SettingsIcon, 
  Play, 
  Share2, 
  RotateCcw, 
  ChevronRight, 
  Info, 
  Trash2, 
  Loader2,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap,
  Microscope,
  Trophy,
  BarChart3,
  Lightbulb
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { GoogleGenAI, Type, GenerateContentResponse, Modality } from "@google/genai";

// --- Types ---

interface Outcome {
  name: string;
  shortName: string;
  baseStrength: number; // 0-100
  volatility: number;   // 0-100
  detail: string;
  emoji: string;
  simCount?: number;
  simProb?: string;
}

interface SimulationResult {
  event: string;
  eventTitle: string;
  category: string;
  alreadyOccurred: boolean;
  winner?: string;
  winnerEmoji?: string;
  detail?: string;
  outcomes?: Outcome[];
  insights: string[];
  iterations: number;
  timestamp: number;
  confidenceLevel?: string;
  dataQuality?: string;
}

interface Suggestion {
  chip: string;
  text: string;
}

// --- Constants ---

const ITER_OPTIONS = [
  { val: 100, label: "100", sub: "iterations", speed: "⚡ instant" },
  { val: 1000, label: "1K", sub: "iterations", speed: "⚡ fast" },
  { val: 10000, label: "10K", sub: "iterations", speed: "★ default" },
  { val: 100000, label: "100K", sub: "iterations", speed: "⏱ ~3s" },
  { val: 1000000, label: "1M", sub: "iterations", speed: "⏱ ~8s" },
  { val: 10000000, label: "10M", sub: "iterations", speed: "🔬 ultra" },
];

// --- Helpers ---

const haptic = (type: 'light' | 'medium' | 'heavy' | 'success' | 'error' = "light") => {
  const patterns = {
    light: [10],
    medium: [30],
    heavy: [50],
    success: [20, 30, 20],
    error: [50, 30, 50]
  };
  if (navigator.vibrate) navigator.vibrate(patterns[type]);
};

function runMonteCarlo(outcomes: Outcome[], total: number, onProgress: (p: any) => void): Promise<Outcome[]> {
  return new Promise(resolve => {
    const counts = new Int32Array(outcomes.length);
    const CHUNK = Math.max(500, Math.floor(total / 100));
    let done = 0;
    const start = Date.now();
    const phases = ["Initializing scenarios...", "Calibrating variables...", "Running Monte Carlo iterations...", "Simulating edge cases...", "Aggregating results...", "Finalizing predictions..."];
    
    // Box-Muller transform for Gaussian noise
    function gaussianRandom() {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function tick() {
      const batch = Math.min(CHUNK, total - done);
      for (let i = 0; i < batch; i++) {
        let maxScore = -Infinity;
        let winnerIdx = 0;
        
        for (let j = 0; j < outcomes.length; j++) {
          const outcome = outcomes[j];
          // Real simulation: Strength + (Noise * Volatility)
          const noise = gaussianRandom();
          const score = outcome.baseStrength + (noise * (outcome.volatility / 2));
          
          if (score > maxScore) {
            maxScore = score;
            winnerIdx = j;
          }
        }
        counts[winnerIdx]++;
      }
      
      done += batch;
      const pct = Math.floor(done / total * 100);
      const elapsed = (Date.now() - start) / 1000;
      const rate = elapsed > 0 ? Math.floor(done / elapsed) : 0;
      const eta = rate > 0 ? Math.ceil((total - done) / rate) : 0;
      const phase = phases[Math.min(Math.floor(pct / (100 / phases.length)), phases.length - 1)];
      
      onProgress({ done, pct, rate, eta, phase });
      
      if (done >= total) {
        resolve(outcomes.map((o, i) => ({
          ...o,
          simCount: counts[i],
          simProb: ((counts[i] / total) * 100).toFixed(1)
        })));
      } else {
        setTimeout(tick, 0);
      }
    }
    setTimeout(tick, 0);
  });
}

const SimulationAnimation = () => {
  return (
    <div className="relative h-24 w-full overflow-hidden rounded-2xl bg-black/5 flex items-center justify-center">
      {/* Scanning Line */}
      <motion.div 
        className="absolute top-0 left-0 w-full h-1 bg-red-500/50 blur-sm z-10"
        animate={{ top: ["0%", "100%", "0%"] }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
      />
      
      {/* Floating Particles */}
      <div className="flex gap-4">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="w-3 h-3 bg-red-600 rounded-full"
            animate={{ 
              y: [0, -20, 0],
              opacity: [0.3, 1, 0.3],
              scale: [1, 1.2, 1]
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              delay: i * 0.4,
              ease: "easeInOut" 
            }}
          />
        ))}
      </div>
      
      {/* Background Grid/Matrix effect */}
      <div className="absolute inset-0 opacity-10 pointer-events-none">
        <div className="grid grid-cols-10 gap-2 p-2">
          {[...Array(40)].map((_, i) => (
            <motion.div 
              key={i} 
              className="h-1 bg-red-900 rounded-full"
              animate={{ opacity: [0.1, 0.5, 0.1] }}
              transition={{ duration: Math.random() * 2 + 1, repeat: Infinity }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

// --- Components ---

export default function MonteGo() {
  const [tab, setTab] = useState<"simulate" | "history" | "settings">("simulate");
  const [event, setEvent] = useState("");
  const [iters, setIters] = useState(10000);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; pct: number; rate: number; eta: number; phase: string } | null>(null);
  const [results, setResults] = useState<SimulationResult | null>(null);
  const [history, setHistory] = useState<SimulationResult[]>(() => {
    try { return JSON.parse(localStorage.getItem("mg_hist") || "[]"); } catch { return []; }
  });
  const [isPast, setIsPast] = useState(false);
  const [hapticOn, setHapticOn] = useState(true);
  const [savHist, setSavHist] = useState(true);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);

  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem("mg_hist", JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    const cached = localStorage.getItem("mg_suggestions");
    const cacheTime = localStorage.getItem("mg_suggestions_time");
    const now = Date.now();
    
    if (cached && cacheTime && now - parseInt(cacheTime) < 24 * 60 * 60 * 1000) {
      try {
        setSuggestions(JSON.parse(cached));
        return;
      } catch (e) {
        console.error("Failed to parse cached suggestions");
      }
    }
    fetchDynamicSuggestions();
  }, []);

  const fetchDynamicSuggestions = async () => {
    setLoadingSuggestions(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
      
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `Today is ${today}. 
        Find 6 trending or upcoming real-world events (sports, politics, finance, entertainment, science) that people might want to run a Monte Carlo simulation for.
        For each event, provide:
        1. A short chip label (e.g., "⚽ FIFA WC 2026")
        2. A clear question for simulation (e.g., "Which country will win the FIFA World Cup 2026?")
        
        Return a JSON array of objects: [{"chip": "...", "text": "..."}]`,
        config: {
          tools: [{ googleSearch: {} }],
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                chip: { type: Type.STRING },
                text: { type: Type.STRING }
              },
              required: ["chip", "text"]
            }
          }
        }
      });

      const data = JSON.parse(response.text || "[]");
      if (Array.isArray(data) && data.length > 0) {
        setSuggestions(data);
        localStorage.setItem("mg_suggestions", JSON.stringify(data));
        localStorage.setItem("mg_suggestions_time", Date.now().toString());
      }
    } catch (error: any) {
      console.error("Failed to fetch suggestions:", error);
      
      // Check for quota error specifically
      const isQuotaError = error?.message?.includes("quota") || error?.status === 429;
      
      if (isQuotaError) {
        console.warn("Gemini API quota exceeded. Using fallback suggestions.");
      }

      setSuggestions([
        { chip: "🏏 Cricket WC 2026", text: "Who will win the ICC Cricket World Cup 2026?" },
        { chip: "⚽ FIFA WC 2026", text: "Which country will win the FIFA World Cup 2026?" },
        { chip: "🏀 NBA Finals 2026", text: "Which team will win the NBA Finals 2026?" },
        { chip: "🗳️ US Midterms 2026", text: "Which party will control the US House after the 2026 midterm elections?" },
        { chip: "🎾 Wimbledon 2026", text: "Who will win the Wimbledon Men's Singles 2026?" },
        { chip: "🏎️ F1 Champion 2026", text: "Who will win the Formula 1 World Drivers' Championship 2026?" },
      ]);
    } finally {
      setLoadingSuggestions(false);
    }
  };

  const saveToHistory = useCallback((res: SimulationResult) => {
    if (!savHist) return;
    setHistory(prev => [res, ...prev.filter(h => h.timestamp !== res.timestamp)].slice(0, 50));
  }, [savHist]);

  const runSimulation = async () => {
    if (!event.trim()) return;
    
    setRunning(true);
    setResults(null);
    setIsPast(false);
    setProgress({ done: 0, pct: 0, rate: 0, eta: 5, phase: "Analyzing event context..." });
    if (hapticOn) haptic("medium");

    // Simulated progress during AI call
    let simPct = 0;
    const progInterval = setInterval(() => {
      simPct += Math.random() * 2;
      if (simPct > 90) simPct = 90;
      const currentDone = Math.floor((simPct / 100) * iters * 0.05); // Use 5% for AI phase
      setProgress(prev => prev ? { 
        ...prev, 
        pct: Math.floor(simPct * 0.1), // 0-9% for AI phase
        done: currentDone,
        rate: 450 + Math.floor(Math.random() * 100),
        eta: Math.max(1, 8 - Math.floor(simPct / 12))
      } : null);
    }, 200);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `You are a probabilistic analyst. Today's date is ${today}.
        Event: "${event.trim()}"
        
        Determine if this event has ALREADY OCCURRED as of ${today}.
        
        Return a JSON object with the following schema:
        {
          "alreadyOccurred": boolean,
          "eventTitle": string (short title),
          "category": "sports" | "politics" | "finance" | "science" | "other",
          "winner": string (if alreadyOccurred),
          "winnerEmoji": string (if alreadyOccurred),
          "detail": string (factual description),
          "outcomes": [
            {
              "name": string,
              "shortName": string (max 10 chars),
              "baseStrength": number (0-100, relative power/likelihood),
              "volatility": number (0-100, how much the performance can vary),
              "detail": string (evidence),
              "emoji": string
            }
          ] (if not alreadyOccurred, 4-8 outcomes),
          "insights": string[] (3-5 insights),
          "confidenceLevel": "low" | "medium" | "high",
          "dataQuality": "low" | "medium" | "high"
        }`,
        config: {
          responseMimeType: "application/json"
        }
      });

      clearInterval(progInterval);
      const aiData = JSON.parse(response.text || "{}");

      if (aiData.alreadyOccurred) {
        setIsPast(true);
        const final: SimulationResult = { ...aiData, event: event.trim(), timestamp: Date.now(), iterations: 0 };
        setResults(final);
        saveToHistory(final);
        if (hapticOn) haptic("success");
      } else {
        setProgress({ done: 0, pct: 10, rate: 0, eta: 0, phase: "Building probability model..." });
        
        const simOutcomes = await runMonteCarlo(aiData.outcomes, iters, (p) => {
          // Map 0-100% of simulation to 10-100% of total progress
          setProgress({
            ...p,
            pct: 10 + Math.floor(p.pct * 0.9)
          });
        });
        simOutcomes.sort((a, b) => parseFloat(b.simProb!) - parseFloat(a.simProb!));
        
        const final: SimulationResult = { ...aiData, outcomes: simOutcomes, iterations: iters, event: event.trim(), timestamp: Date.now() };
        setResults(final);
        saveToHistory(final);
        if (hapticOn) haptic("success");
      }
    } catch (error: any) {
      clearInterval(progInterval);
      console.error(error);
      if (hapticOn) haptic("error");
      
      const isQuotaError = error?.message?.includes("quota") || error?.status === 429;
      if (isQuotaError) {
        alert("Gemini API quota exceeded. Please try again in a few minutes.");
      } else {
        alert("Simulation failed. Please try again.");
      }
    } finally {
      setRunning(false);
      setProgress(null);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  };

  const shareResults = () => {
    if (!results) return;
    if (hapticOn) haptic("medium");
    const top = results.outcomes?.[0];
    const text = isPast
      ? `🎲 MonteGo Result\n\n"${results.event}"\n\n✅ Outcome: ${results.winnerEmoji || ""} ${results.winner}\n\n#MonteGo`
      : `🎲 MonteGo Prediction\n\n"${results.event}"\n\nTop Pick: ${top?.emoji || ""} ${top?.name} (${top?.simProb}%)\n\nRan ${results.iterations.toLocaleString()} simulations.\n\n#MonteGo #MonteCarlo`;
    
    if (navigator.share) {
      navigator.share({ title: "MonteGo Prediction", text }).catch(() => {});
    } else {
      navigator.clipboard.writeText(text);
      alert("Copied to clipboard!");
    }
  };

  const entropy = results && !isPast ? -results.outcomes!.reduce((s, o) => {
    const p = parseFloat(o.simProb!) / 100;
    return s + (p > 0 ? p * Math.log2(p) : 0);
  }, 0) : 0;

  return (
    <div className="min-h-screen bg-[#F2F2F7] text-[#1C1C1E] font-sans pb-24">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-black/5 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-red-700 to-red-500 rounded-lg flex items-center justify-center shadow-lg shadow-red-900/20">
            <Dices className="text-white w-5 h-5" />
          </div>
          <h1 className="text-xl font-bold tracking-tight">Monte<span className="text-red-700">Go</span></h1>
        </div>
        <div className="px-3 py-1 bg-red-50 border border-red-100 rounded-full text-[10px] font-bold text-red-700 uppercase tracking-wider">
          Gemini AI
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 pt-6">
        <AnimatePresence mode="wait">
          {tab === "simulate" && (
            <motion.div
              key="simulate"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Hero */}
              <section className="space-y-2">
                <p className="text-xs font-bold text-red-700 uppercase tracking-widest">Monte Carlo Engine</p>
                <h2 className="text-4xl font-extrabold tracking-tight leading-tight">What will<br />happen next?</h2>
                <p className="text-gray-500 text-lg leading-relaxed">Run millions of simulations powered by Gemini AI to predict any real-world event.</p>
              </section>

              {/* Input Card */}
              <section className="bg-white rounded-3xl shadow-xl shadow-black/5 overflow-hidden border border-black/5">
                <div className="p-4 space-y-4">
                  <div className="flex items-center gap-2 text-xs font-bold text-gray-400 uppercase tracking-wider">
                    <Sparkles className="w-4 h-4 text-red-600" />
                    Describe the Event
                  </div>
                  <textarea
                    value={event}
                    onChange={(e) => setEvent(e.target.value)}
                    placeholder="E.g. Who will win the ICC Cricket World Cup 2026?"
                    className="w-full min-h-[120px] text-xl font-medium placeholder:text-gray-300 focus:outline-none resize-none"
                  />
                </div>
                <div className="bg-gray-50 p-3 border-t border-black/5 flex items-center gap-2 overflow-x-auto no-scrollbar min-h-[52px]">
                  {loadingSuggestions ? (
                    <div className="flex items-center gap-2 px-2">
                      <Loader2 className="w-3 h-3 animate-spin text-red-600" />
                      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Fetching trends...</span>
                    </div>
                  ) : (
                    suggestions.map((s) => (
                      <button
                        key={s.chip}
                        onClick={() => { setEvent(s.text); if (hapticOn) haptic("light"); }}
                        className="whitespace-nowrap px-4 py-2 bg-white border border-black/5 rounded-full text-xs font-semibold text-red-700 hover:bg-red-50 transition-colors shadow-sm"
                      >
                        {s.chip}
                      </button>
                    ))
                  )}
                </div>
              </section>

              {/* Iterations */}
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-bold text-gray-400 uppercase tracking-wider px-1">
                  <Zap className="w-4 h-4 text-red-600" />
                  Simulation Iterations
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {ITER_OPTIONS.map((o) => (
                    <button
                      key={o.val}
                      onClick={() => { setIters(o.val); if (hapticOn) haptic("medium"); }}
                      className={`p-3 rounded-2xl border-2 transition-all flex flex-col items-center justify-center gap-1 ${
                        iters === o.val 
                          ? "bg-red-700 border-red-700 text-white shadow-lg shadow-red-900/20 scale-105" 
                          : "bg-white border-black/5 text-gray-900 hover:border-red-200"
                      }`}
                    >
                      <span className="text-lg font-bold">{o.label}</span>
                      <span className={`text-[10px] uppercase font-bold ${iters === o.val ? "text-red-100" : "text-gray-400"}`}>{o.sub}</span>
                      <span className={`text-[9px] px-2 py-0.5 rounded-full ${iters === o.val ? "bg-white/20 text-white" : "bg-gray-100 text-gray-500"}`}>{o.speed}</span>
                    </button>
                  ))}
                </div>
              </section>

              {/* Run Button */}
              <button
                disabled={running || !event.trim()}
                onClick={runSimulation}
                className={`w-full py-5 rounded-full font-bold text-lg flex items-center justify-center gap-3 transition-all ${
                  running || !event.trim()
                    ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-red-800 to-red-600 text-white shadow-xl shadow-red-900/30 hover:scale-[1.02] active:scale-95"
                }`}
              >
                {running ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>Simulating...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-6 h-6 fill-current" />
                    <span>{results ? "Run Again" : "Run Simulation"}</span>
                  </>
                )}
              </button>

              {/* Progress */}
              {progress && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-white rounded-3xl p-6 shadow-xl border border-black/5 space-y-4"
                >
                  <SimulationAnimation />
                  <div className="flex justify-between items-end">
                    <div className="space-y-1">
                      <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">{progress.phase}</p>
                      <p className="text-2xl font-black text-red-700">{progress.pct}%</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">ETA</p>
                      <p className="text-xl font-bold">{progress.eta}s</p>
                    </div>
                  </div>
                  <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-gradient-to-r from-red-700 to-red-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress.pct}%` }}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div className="bg-gray-50 p-3 rounded-2xl border border-black/5">
                      <p className="text-[10px] font-bold text-gray-400 uppercase">Processed</p>
                      <p className="text-lg font-bold">{progress.done.toLocaleString()}</p>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-2xl border border-black/5">
                      <p className="text-[10px] font-bold text-gray-400 uppercase">Speed</p>
                      <p className="text-lg font-bold">{progress.rate.toLocaleString()}/s</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Results */}
              {results && !progress && (
                <div ref={resultsRef} className="space-y-6 pb-12">
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white rounded-3xl shadow-2xl border border-black/5 overflow-hidden"
                  >
                    {/* Result Header */}
                    <div className="p-6 bg-red-50/50 border-b border-black/5 space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="px-3 py-1 bg-red-700 text-white text-[10px] font-bold rounded-full uppercase tracking-widest">
                          {isPast ? "Historical Record" : "Simulation Results"}
                        </span>
                        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                          {new Date(results.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                      <h3 className="text-2xl font-bold leading-tight">{results.event}</h3>
                      <div className="flex flex-wrap gap-2">
                        <span className="px-2 py-1 bg-white border border-black/5 rounded-lg text-[10px] font-bold text-gray-500 uppercase">
                          {results.category}
                        </span>
                        {!isPast && (
                          <span className="px-2 py-1 bg-white border border-black/5 rounded-lg text-[10px] font-bold text-gray-500 uppercase">
                            Confidence: {results.confidenceLevel}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Body */}
                    {isPast ? (
                      <div className="p-6 space-y-6">
                        <div className="bg-red-50 border border-red-100 rounded-2xl p-6 text-center space-y-4">
                          <div className="w-16 h-16 bg-gradient-to-br from-yellow-500 to-yellow-300 rounded-2xl mx-auto flex items-center justify-center shadow-lg shadow-yellow-500/20">
                            <span className="text-3xl">{results.winnerEmoji || "🏆"}</span>
                          </div>
                          <div className="space-y-1">
                            <p className="text-xs font-bold text-red-700 uppercase tracking-widest">Confirmed Result</p>
                            <h4 className="text-2xl font-black">{results.winner}</h4>
                          </div>
                          <p className="text-gray-600 text-sm leading-relaxed">{results.detail}</p>
                        </div>
                        
                        <div className="space-y-4">
                          <h5 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                            <HistoryIcon className="w-4 h-4" />
                            Historical Facts
                          </h5>
                          <div className="space-y-3">
                            {results.insights.map((ins, i) => (
                              <div key={i} className="flex gap-3 items-start">
                                <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 mt-2 shrink-0" />
                                <p className="text-sm text-gray-600 leading-relaxed">{ins}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="divide-y divide-black/5">
                        {/* Outcomes List */}
                        <div className="p-6 space-y-6">
                          <h5 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                            <BarChart3 className="w-4 h-4" />
                            Probability Distribution
                          </h5>
                          <div className="space-y-6">
                            {results.outcomes?.map((o, i) => {
                              const isTop = i === 0;
                              const pct = parseFloat(o.simProb!);
                              return (
                                <div key={i} className="space-y-3 group">
                                  <div className="flex justify-between items-start gap-4">
                                    <div className="flex gap-4 flex-1">
                                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg font-bold shrink-0 shadow-sm ${
                                        i === 0 ? "bg-yellow-100 text-yellow-700" :
                                        i === 1 ? "bg-gray-100 text-gray-600" :
                                        i === 2 ? "bg-orange-100 text-orange-700" :
                                        "bg-red-50 text-red-700"
                                      }`}>
                                        {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
                                      </div>
                                      <div className="space-y-1">
                                        <div className="flex items-center gap-2">
                                          <span className="text-lg font-bold">{o.emoji} {o.name}</span>
                                        </div>
                                        <p className="text-xs text-gray-500 leading-relaxed">{o.detail}</p>
                                      </div>
                                    </div>
                                    <div className={`text-2xl font-black ${isTop ? "text-red-700" : "text-gray-900"}`}>
                                      {o.simProb}%
                                    </div>
                                  </div>
                                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                    <motion.div 
                                      className={`h-full ${isTop ? "bg-gradient-to-r from-red-700 to-red-500" : "bg-gray-400"}`}
                                      initial={{ width: 0 }}
                                      animate={{ width: `${pct}%` }}
                                      transition={{ duration: 1, delay: i * 0.1 }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        {/* Insights */}
                        <div className="p-6 bg-gray-50/50 space-y-4">
                          <h5 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                            <Lightbulb className="w-4 h-4" />
                            AI Insights
                          </h5>
                          <div className="grid gap-3">
                            {results.insights.map((ins, i) => (
                              <div key={i} className="bg-white p-4 rounded-2xl border border-black/5 shadow-sm flex gap-3">
                                <div className="w-2 h-2 rounded-full bg-red-600 mt-1.5 shrink-0" />
                                <p className="text-sm text-gray-600 leading-relaxed">{ins}</p>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Stats Summary */}
                        <div className="grid grid-cols-4 divide-x divide-black/5">
                          <div className="p-4 text-center">
                            <p className="text-[10px] font-bold text-gray-400 uppercase mb-1">Top Pick</p>
                            <p className="text-lg font-black text-red-700">{results.outcomes![0].simProb}%</p>
                          </div>
                          <div className="p-4 text-center">
                            <p className="text-[10px] font-bold text-gray-400 uppercase mb-1">Outcomes</p>
                            <p className="text-lg font-black">{results.outcomes!.length}</p>
                          </div>
                          <div className="p-4 text-center">
                            <p className="text-[10px] font-bold text-gray-400 uppercase mb-1">Entropy</p>
                            <p className="text-lg font-black">{entropy.toFixed(1)}</p>
                          </div>
                          <div className="p-4 text-center">
                            <p className="text-[10px] font-bold text-gray-400 uppercase mb-1">Iterations</p>
                            <p className="text-lg font-black">{results.iterations.toLocaleString()}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="p-4 bg-gray-50 border-t border-black/5 flex gap-3">
                      <button 
                        onClick={() => { setResults(null); setEvent(""); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                        className="flex-1 py-3 bg-white border border-black/10 rounded-2xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-gray-50 active:scale-95 transition-all"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                      </button>
                      <button 
                        onClick={shareResults}
                        className="flex-1 py-3 bg-red-700 text-white rounded-2xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-red-900/20 hover:bg-red-800 active:scale-95 transition-all"
                      >
                        <Share2 className="w-4 h-4" />
                        Share
                      </button>
                    </div>
                  </motion.div>
                </div>
              )}
            </motion.div>
          )}

          {tab === "history" && (
            <motion.div
              key="history"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <section className="space-y-2">
                <p className="text-xs font-bold text-red-700 uppercase tracking-widest">Your Simulations</p>
                <h2 className="text-4xl font-extrabold tracking-tight">History</h2>
                <p className="text-gray-500 text-lg leading-relaxed">All your past Monte Carlo runs, saved locally.</p>
              </section>

              <div className="bg-white rounded-3xl shadow-xl border border-black/5 overflow-hidden divide-y divide-black/5">
                {history.length === 0 ? (
                  <div className="p-12 text-center space-y-4">
                    <div className="w-16 h-16 bg-gray-50 rounded-full mx-auto flex items-center justify-center text-gray-300">
                      <HistoryIcon className="w-8 h-8" />
                    </div>
                    <p className="text-gray-400 font-medium">No simulations yet.<br />Run your first one! 🎲</p>
                  </div>
                ) : (
                  history.map((h, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setEvent(h.event);
                        setResults(h);
                        setIsPast(h.alreadyOccurred);
                        setTab("simulate");
                        if (hapticOn) haptic("light");
                      }}
                      className="w-full p-4 flex items-center gap-4 hover:bg-gray-50 transition-colors text-left group"
                    >
                      <div className="w-12 h-12 bg-red-50 rounded-2xl flex items-center justify-center text-red-700 group-hover:scale-110 transition-transform">
                        {h.alreadyOccurred ? <Clock className="w-6 h-6" /> : <Zap className="w-6 h-6" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-bold truncate text-sm">{h.event}</h4>
                        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mt-0.5">
                          {new Date(h.timestamp).toLocaleDateString()} • {h.alreadyOccurred ? "Past Event" : `${h.iterations.toLocaleString()} Runs`}
                        </p>
                      </div>
                      {!h.alreadyOccurred && h.outcomes && (
                        <div className="text-right shrink-0">
                          <p className="text-lg font-black text-red-700">{h.outcomes[0].simProb}%</p>
                          <p className="text-[8px] font-bold text-gray-400 uppercase">Confidence</p>
                        </div>
                      )}
                      <ChevronRight className="w-5 h-5 text-gray-300" />
                    </button>
                  ))
                )}
              </div>
            </motion.div>
          )}

          {tab === "settings" && (
            <motion.div
              key="settings"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-6"
            >
              <section className="space-y-2">
                <p className="text-xs font-bold text-red-700 uppercase tracking-widest">Preferences</p>
                <h2 className="text-4xl font-extrabold tracking-tight">Settings</h2>
              </section>

              <div className="space-y-6">
                <div className="bg-white rounded-3xl shadow-xl border border-black/5 overflow-hidden divide-y divide-black/5">
                  <div className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-red-50 rounded-xl flex items-center justify-center text-red-700">
                        <Zap className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="font-bold text-sm">Haptic Feedback</p>
                        <p className="text-[10px] text-gray-400 uppercase tracking-widest">Tactile Response</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => { setHapticOn(!hapticOn); if (!hapticOn) haptic("medium"); }}
                      className={`w-12 h-6 rounded-full transition-all relative ${hapticOn ? "bg-green-500" : "bg-gray-200"}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${hapticOn ? "left-7" : "left-1"}`} />
                    </button>
                  </div>
                  <div className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-red-50 rounded-xl flex items-center justify-center text-red-700">
                        <HistoryIcon className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="font-bold text-sm">Save History</p>
                        <p className="text-[10px] text-gray-400 uppercase tracking-widest">Local Storage</p>
                      </div>
                    </div>
                    <button 
                      onClick={() => setSavHist(!savHist)}
                      className={`w-12 h-6 rounded-full transition-all relative ${savHist ? "bg-green-500" : "bg-gray-200"}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${savHist ? "left-7" : "left-1"}`} />
                    </button>
                  </div>
                </div>

                <div className="bg-white rounded-3xl shadow-xl border border-black/5 overflow-hidden">
                  <div className="p-6 text-center space-y-4">
                    <div className="w-20 h-20 bg-gradient-to-br from-red-700 to-red-500 rounded-3xl mx-auto flex items-center justify-center shadow-2xl shadow-red-900/30">
                      <Dices className="text-white w-10 h-10" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-black">MonteGo</h3>
                      <p className="text-gray-400 text-xs font-bold uppercase tracking-widest">Version 1.0.0 • Powered by Gemini AI</p>
                    </div>
                    <p className="text-gray-500 text-sm leading-relaxed">
                      MonteGo uses advanced Monte Carlo algorithms and Gemini's reasoning capabilities to simulate millions of potential futures for any event you can describe.
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 border-t border-black/5">
                    <button 
                      onClick={() => {
                        if (confirm("Are you sure you want to clear all data? This will reset your history and preferences.")) {
                          setHistory([]);
                          setResults(null);
                          setEvent("");
                          setSuggestions([]);
                          localStorage.clear();
                          fetchDynamicSuggestions();
                          if (hapticOn) haptic("heavy");
                          alert("All local data has been cleared.");
                        }
                      }}
                      className="w-full py-3 bg-white border border-red-100 text-red-600 rounded-2xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-red-50 transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                      Clear All Local Data
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Tab Bar */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-2xl border-t border-black/5 px-6 pt-2 pb-[env(safe-area-inset-bottom,16px)] flex justify-between items-center z-50">
        {[
          { id: "simulate", icon: Dices, label: "Predict" },
          { id: "history", icon: HistoryIcon, label: "History" },
          { id: "settings", icon: SettingsIcon, label: "Settings" },
        ].map((t) => {
          const Icon = t.icon;
          const isActive = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => { setTab(t.id as any); if (hapticOn) haptic("light"); }}
              className={`flex flex-col items-center gap-1 transition-all ${isActive ? "text-red-700 scale-110" : "text-gray-400 hover:text-gray-600"}`}
            >
              <Icon className={`w-6 h-6 ${isActive ? "fill-red-700/10" : ""}`} />
              <span className="text-[10px] font-bold uppercase tracking-wider">{t.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}
