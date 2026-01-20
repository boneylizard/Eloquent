import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Upload, FileText, X, Search, RotateCcw } from 'lucide-react';

const ForensicLinguistics = ({ isOpen = true, onClose }) => {
  const { BACKEND, clearError, apiError } = useApp();

  // Don't render if explicitly closed (for overlay mode)
  if (isOpen === false) return null;

  // --- 1. STATE RESTORATION ---

  const [embeddingModels, setEmbeddingModels] = useState({
    models: {
      gme: { enabled: false, dimensions: null, priority: 1 },
      mxbai_large: { enabled: false, dimensions: 1024, priority: 2 },
      multilingual_e5: { enabled: false, dimensions: 1024, priority: 3 },
      qwen3_8b: { enabled: false, dimensions: null, priority: 4 },
      qwen3_4b: { enabled: false, dimensions: null, priority: 5 },
      frida: { enabled: false, dimensions: 1024, priority: 6 },
      bge_m3: { enabled: false, dimensions: 1024, priority: 7 },
      gte_qwen2: { enabled: false, dimensions: 3584, priority: 8 },
      inf_retriever: { enabled: false, dimensions: 3584, priority: 9 },
      sentence_t5: { enabled: false, dimensions: 768, priority: 10 },
      star: { enabled: false, dimensions: 1024, priority: 11 },
      roberta: { enabled: false, dimensions: 768, priority: 12 },
      jina_v3: { enabled: false, dimensions: 1024, priority: 13 },
      nomic_v1_5: { enabled: false, dimensions: 768, priority: 14 },
      arctic_embed: { enabled: false, dimensions: 768, priority: 15 }
    },
    active_model: 'None'
  });

  // Main states
  const [activeTab, setActiveTab] = useState('analyze');
  const [availableFigures, setAvailableFigures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [savedResults, setSavedResults] = useState([]);
  const [averagedResult, setAveragedResult] = useState(null);
  const [error, setError] = useState(null);
  const [initializingGME, setInitializingGME] = useState(false);
  const [success, setSuccess] = useState(null);
  const [gmeAvailable, setGmeAvailable] = useState(false);

  // File upload states
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [processingFiles, setProcessingFiles] = useState(false);
  const fileInputRef = useRef(null);

  // Corpus building states
  const [buildingCorpus, setBuildingCorpus] = useState(false);
  const [corpusProgress, setCorpusProgress] = useState({});

  // Analysis states
  const [selectedFigure, setSelectedFigure] = useState('');
  const [statement, setStatement] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStatus, setAnalysisStatus] = useState('');
  const [analysisTaskId, setAnalysisTaskId] = useState(null);

  // Comparison states
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [text1Label, setText1Label] = useState('Text 1');
  const [text2Label, setText2Label] = useState('Text 2');
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparing, setComparing] = useState(false);

  // Corpus building form states
  const [newFigureName, setNewFigureName] = useState('');
  const [selectedPlatforms, setSelectedPlatforms] = useState(['twitter', 'speeches']);
  const [maxDocuments, setMaxDocuments] = useState(1000);

  // File upload for corpus building
  const [corpusFiles, setCorpusFiles] = useState([]);
  const [buildMethod, setBuildMethod] = useState('auto'); // 'auto' or 'upload'
  const fileInputRefCorpus = useRef(null);

  // --- 2. LOGIC RESTORATION ---

  const getScoreInterpretation = (score) => {
    if (score >= 0.8) return "Very high similarity - likely same author";
    if (score >= 0.6) return "High similarity - probably same author";
    if (score >= 0.4) return "Moderate similarity - uncertain authorship";
    if (score >= 0.2) return "Low similarity - probably different author";
    return "Very low similarity - likely different author";
  };

  const formatScore = (score) => {
    if (typeof score !== 'number') return 'N/A';
    return `${(score * 100).toFixed(1)}%`;
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  const readFileAsText = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  // Poll for analysis progress
  useEffect(() => {
    if (!analysisTaskId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${BACKEND}/forensic/progress/${analysisTaskId}`);

        if (response.status === 404) {
          setError("Analysis task not found. It may have expired or been interrupted.");
          clearInterval(interval);
          setAnalysisTaskId(null);
          setAnalyzing(false);
          return;
        }

        if (!response.ok) throw new Error(`Server status: ${response.status}`);

        const data = await response.json();
        setAnalysisProgress(data.progress);
        setAnalysisStatus(data.status);

        if (data.progress >= 100) {
          clearInterval(interval);
          setAnalysisTaskId(null);
          setAnalyzing(false);

          if (data.result && !data.result.error) {
            setAnalysisResult(data.result);
            setSuccess("Analysis complete.");
          } else {
            setError(data.status || "Analysis failed.");
            setAnalysisResult(null);
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
        // On mobile, sometimes the connection drops briefly, don't kill the task immediately unless persistent
        // For now, we'll keep polling.
      }
    }, 500);

    return () => clearInterval(interval);
  }, [analysisTaskId, BACKEND]);

  const fetchEmbeddingStatus = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND}/forensic/embedding-status`);
      if (response.ok) {
        const data = await response.json();
        setEmbeddingModels(data);
        setGmeAvailable(data.gme_enabled);
      }
    } catch (error) {
      console.error('Error fetching embedding status:', error);
      // Check for finding network mismatch (Mobile initialization phase)
      const isNetworkMismatch = (BACKEND.includes('localhost') || BACKEND.includes('127.0.0.1') || BACKEND.includes('0.0.0.0')) &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1';

      if (isNetworkMismatch) {
        console.log("Suppressing embedding status error during network init");
      }
    }
  }, [BACKEND]);

  const loadAvailableFigures = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BACKEND}/forensic/available-figures`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setAvailableFigures(data.figures || []);
      setError(null);
    } catch (err) {
      console.error('Error loading figures:', err);

      // Check for finding network mismatch (Mobile initialization phase)
      const isNetworkMismatch = (BACKEND.includes('localhost') || BACKEND.includes('127.0.0.1') || BACKEND.includes('0.0.0.0')) &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1';

      if (isNetworkMismatch || (err.message.includes('fetch') && window.location.hostname !== 'localhost')) {
        console.log("Suppressing fetch error during network init");
        setError(null);
        if (clearError) clearError();
      } else {
        setError(`Failed to load available figures: ${err.message}`);
        // Auto-clear error after 3 seconds if it's just a fetch error
        if (err.message.includes('fetch') || err.message.includes('HTTP')) {
          setTimeout(() => setError(null), 3000);
        }
      }
    } finally {
      setLoading(false);
    }
  }, [BACKEND, clearError]);

  useEffect(() => {
    if (clearError) clearError();
    loadAvailableFigures();
    fetchEmbeddingStatus();
  }, [loadAvailableFigures, fetchEmbeddingStatus, clearError]);

  // --- API ACTIONS ---

  const handleFileUpload = useCallback(async (files) => {
    const validFiles = Array.from(files).filter(file => {
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return ['.txt', '.md', '.csv', '.json', '.xml'].includes(extension) && file.size <= 10 * 1024 * 1024;
    });

    if (validFiles.length === 0) {
      setError('Please upload valid files (.txt, .md, .csv, .json, .xml) under 10MB');
      return;
    }

    setProcessingFiles(true);
    const newFiles = [];

    for (const file of validFiles) {
      try {
        const text = await readFileAsText(file);
        newFiles.push({
          id: Date.now() + Math.random(),
          name: file.name,
          type: file.type || 'text/plain',
          size: file.size,
          content: text,
          uploadedAt: new Date().toISOString()
        });
      } catch (err) {
        console.error(`Error reading file ${file.name}:`, err);
        setError(`Failed to read file: ${file.name}`);
      }
    }

    setUploadedFiles(prev => [...prev, ...newFiles]);
    setProcessingFiles(false);
  }, []);

  const removeFile = useCallback((fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileUpload(e.dataTransfer.files);
  }, [handleFileUpload]);

  const analyzeStatement = useCallback(async () => {
    if (!statement.trim() || !selectedFigure) return;

    try {
      setAnalyzing(true);
      setError(null);
      setAnalysisResult(null);
      setAnalysisProgress(0);
      setAnalysisStatus('Starting analysis...');

      const response = await fetch(`${BACKEND}/forensic/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          statement: statement.trim(),
          person_name: selectedFigure
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.task_id) {
        setAnalysisTaskId(data.task_id);
      } else {
        throw new Error("Backend did not return a task ID.");
      }
    } catch (err) {
      console.error('Error starting analysis:', err);
      setError(`Failed to start analysis: ${err.message}`);
      setAnalyzing(false);
    }
  }, [statement, selectedFigure, BACKEND]);

  const analyzeUploadedFiles = useCallback(async () => {
    if (uploadedFiles.length === 0) {
      setError('Please upload files first');
      return;
    }
    try {
      setAnalyzing(true);
      setError(null);
      const combinedText = uploadedFiles.map(file => `=== ${file.name} ===\n${file.content}\n\n`).join('');
      const response = await fetch(`${BACKEND}/forensic/extract-features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: combinedText })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setAnalysisResult({ ...data, file_analysis: true, files_analyzed: uploadedFiles.map(f => ({ name: f.name, size: f.size })) });
    } catch (err) {
      setError(`File analysis failed: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  }, [uploadedFiles, BACKEND]);

  const compareTexts = useCallback(async () => {
    if (!text1.trim() || !text2.trim()) return;
    try {
      setComparing(true);
      setError(null);
      const response = await fetch(`${BACKEND}/forensic/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text1: text1.trim(),
          text2: text2.trim(),
          text1_label: text1Label,
          text2_label: text2Label
        })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setComparisonResult(data.comparison);
    } catch (err) {
      setError(`Comparison failed: ${err.message}`);
    } finally {
      setComparing(false);
    }
  }, [text1, text2, text1Label, text2Label, BACKEND]);

  const buildCorpus = useCallback(async () => {
    if (!newFigureName.trim()) {
      setError('Please enter a figure name');
      return;
    }
    if (buildMethod === 'upload' && corpusFiles.length === 0) {
      setError('Please upload files to build the corpus');
      return;
    }

    try {
      setBuildingCorpus(true);
      setError(null);
      setCorpusProgress({ status: 'starting', progress: 0 });

      if (buildMethod === 'upload') {
        const formData = new FormData();
        formData.append('person_name', newFigureName.trim());
        corpusFiles.forEach(file => {
          formData.append('files', file.file);
        });
        const response = await fetch(`${BACKEND}/forensic/build-corpus-from-files`, {
          method: 'POST',
          body: formData
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setCorpusProgress({
          status: 'success',
          message: `Built corpus from ${data.corpus_stats.files_processed} files`
        });
      } else {
        const response = await fetch(`${BACKEND}/forensic/build-corpus`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            person_name: newFigureName.trim(),
            platforms: selectedPlatforms,
            max_documents: maxDocuments
          })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setCorpusProgress({ status: 'success', message: data.message });
      }
      setNewFigureName('');
      setCorpusFiles([]);
      setTimeout(() => { loadAvailableFigures(); setCorpusProgress({}); }, 2000);
    } catch (err) {
      setError(`Corpus building failed: ${err.message}`);
      setCorpusProgress({});
    } finally {
      setBuildingCorpus(false);
    }
  }, [newFigureName, selectedPlatforms, maxDocuments, buildMethod, corpusFiles, BACKEND, loadAvailableFigures]);

  const handleCorpusFileUpload = useCallback((files) => {
    const validFiles = Array.from(files).filter(file => {
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return ['.txt', '.md', '.csv', '.json', '.xml'].includes(extension);
    });
    const newFiles = validFiles.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      size: file.size,
      file: file
    }));
    setCorpusFiles(prev => [...prev, ...newFiles]);
  }, []);

  const removeCorpusFile = useCallback((fileId) => {
    setCorpusFiles(prev => prev.filter(f => f.id !== fileId));
  }, []);

  const deleteCorpus = useCallback(async (figureName) => {
    if (!confirm(`Delete corpus for ${figureName}?`)) return;
    try {
      const response = await fetch(`${BACKEND}/forensic/corpus/${encodeURIComponent(figureName)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      await loadAvailableFigures();
      if (selectedFigure === figureName) setSelectedFigure('');
    } catch (err) {
      setError(`Failed to delete corpus: ${err.message}`);
    }
  }, [BACKEND, loadAvailableFigures, selectedFigure]);

  const initializeEmbeddingModel = useCallback(async (modelType, gpuId = 0) => {
    try {
      setInitializingGME(true);
      setError(null);
      setSuccess(null);
      const response = await fetch(`${BACKEND}/forensic/initialize-embedding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelType, gpu_id: gpuId })
      });
      if (response.ok) {
        await fetchEmbeddingStatus();
        setSuccess(`${modelType.toUpperCase()} initialized successfully on GPU ${gpuId}!`);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to initialize ${modelType}`);
      }
    } catch (err) {
      setError(`Failed to initialize ${modelType}: ${err.message}`);
    } finally {
      setInitializingGME(false);
    }
  }, [BACKEND, fetchEmbeddingStatus]);

  const handleSaveResult = useCallback(() => {
    if (!analysisResult) return;
    const resultToSave = {
      ...analysisResult,
      id: `res-${Date.now()}`,
      modelUsed: embeddingModels.active_model || 'Unknown',
    };
    setSavedResults(prev => [...prev, resultToSave]);
    setSuccess("Analysis result saved.");
  }, [analysisResult, embeddingModels.active_model]);

  const handleClearSaved = useCallback(() => {
    setSavedResults([]);
    setAveragedResult(null);
  }, []);

  const handleAverageResults = useCallback(() => {
    if (savedResults.length < 2) {
      setError("Please save at least two results to average.");
      return;
    }
    const total = savedResults.length;
    const avgScores = {
      overall_similarity: 0,
      function_words: 0,
      syntactic_patterns: 0,
      character_patterns: 0,
      lexical_complexity: 0,
      punctuation_style: 0,
      semantic_score: 0,
      confidence: 0,
    };
    savedResults.forEach(result => {
      for (const key in avgScores) {
        avgScores[key] += (result.similarity_scores[key] || 0) / total;
      }
    });
    setAveragedResult({
      person_analyzed: savedResults[0].person_analyzed,
      interpretation: getScoreInterpretation(avgScores.overall_similarity),
      similarity_scores: avgScores,
      models_averaged: savedResults.map(r => r.modelUsed),
    });
    setSuccess(`Averaged ${total} results.`);
  }, [savedResults]);

  // --- 3. UI RENDERING (RESPONSIVE) ---

  const ResultDisplay = ({ result, title, onSave, isAveraged = false }) => {
    if (!result) return null;
    const scores = result.similarity_scores;
    if (!scores && !result.file_analysis) return null;

    if (result.file_analysis) {
      // File analysis view
      const features = result.features;
      return (
        <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border mt-4">
          <h3 className="text-xl font-bold mb-2">File Analysis Results</h3>
          <p className="text-sm text-muted-foreground mb-4">Files: {result.files_analyzed?.map(f => f.name).join(', ')}</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-blue-500/10 rounded border border-blue-500/20">
              <h4 className="font-semibold text-blue-300">Lexical</h4>
              <div className="text-sm">Word Len: {features.lexical_features?.avg_word_length}</div>
              <div className="text-sm">Vocab: {features.lexical_features?.vocab_richness}</div>
            </div>
            <div className="p-3 bg-green-500/10 rounded border border-green-500/20">
              <h4 className="font-semibold text-green-300">Stylistic</h4>
              <div className="text-sm">Passive: {formatScore(features.stylistic_features?.passive_voice_ratio)}</div>
            </div>
            <div className="p-3 bg-purple-500/10 rounded border border-purple-500/20">
              <h4 className="font-semibold text-purple-300">Stats</h4>
              <div className="text-sm">Words: {features.text_statistics?.word_count}</div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border mt-4">
        <div className="flex flex-col md:flex-row justify-between items-start mb-4 gap-2">
          <div>
            <h3 className="text-xl font-bold text-foreground">{title}</h3>
            <p className="text-muted-foreground text-sm">
              {isAveraged
                ? `Averaged across: ${result.models_averaged.join(', ')}`
                : `Analyzed against ${result.person_analyzed}`
              }
            </p>
          </div>
          {!isAveraged && onSave && (
            <Button onClick={onSave} size="sm" className="w-full md:w-auto">
              💾 Save Result
            </Button>
          )}
        </div>

        <div className="mb-6 p-4 bg-muted rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-foreground">Overall Similarity</span>
            <span className={`text-2xl font-bold ${getScoreColor(scores.overall_similarity)}`}>
              {formatScore(scores.overall_similarity)}
            </span>
          </div>
          <div className="w-full bg-muted-foreground/20 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${scores.overall_similarity >= 0.8 ? 'bg-green-500' :
                scores.overall_similarity >= 0.6 ? 'bg-yellow-500' :
                  scores.overall_similarity >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
                }`}
              style={{ width: `${scores.overall_similarity * 100}%` }}
            />
          </div>
          <p className="text-sm text-muted-foreground mt-2">{result.interpretation}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Detailed Scores */}
          {[
            { label: 'Semantic (Embedding)', score: scores.semantic_score, color: 'indigo', sub: '30% weight' },
            { label: 'Function Words', score: scores.function_words, color: 'blue', sub: '25% weight' },
            { label: 'Syntactic Patterns', score: scores.syntactic_patterns, color: 'green', sub: '20% weight' },
            { label: 'Character Patterns', score: scores.character_patterns, color: 'purple', sub: '15% weight' },
            { label: 'Lexical Complexity', score: scores.lexical_complexity, color: 'orange', sub: '5% weight' },
            { label: 'Punctuation Style', score: scores.punctuation_style, color: 'cyan', sub: '5% weight' }
          ].map((item, i) => (
            <div key={i} className={`p-3 bg-${item.color}-500/10 rounded-lg border border-${item.color}-500/20`}>
              <div className={`font-semibold text-${item.color}-300`}>{item.label}</div>
              <div className={`text-2xl font-bold text-${item.color}-400`}>{formatScore(item.score || 0)}</div>
              <div className={`text-xs text-${item.color}-300`}>{item.sub}</div>
            </div>
          ))}

          {scores.topic_warning && (
            <div className="col-span-1 md:col-span-2 lg:col-span-3 p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
              <div className="font-semibold text-yellow-300 text-sm">⚠️ {scores.topic_warning}</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 bg-background md:bg-black/80 flex items-center justify-center p-0 md:p-4">
      {/* Container: Full screen mobile, centered modal desktop */}
      <div className="relative bg-background w-full h-full md:h-[90vh] md:rounded-lg overflow-hidden flex flex-col shadow-2xl">

        {/* Sticky Header */}
        <div className="flex-none p-4 md:p-6 border-b border-border bg-background/95 backdrop-blur z-20 sticky top-0 flex items-center justify-between">
          <div className="pr-8">
            <h1 className="text-xl md:text-3xl font-bold text-foreground">Forensic Linguistics</h1>
            <p className="text-xs md:text-base text-muted-foreground truncate">
              Authorship attribution & verification
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-10 w-10 bg-muted/50 hover:bg-muted rounded-full"
          >
            <X className="h-6 w-6" />
          </Button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6">

          {/* Status Messages */}
          {(error || apiError) && (
            <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex justify-between">
              <p className="text-destructive text-sm">{error || apiError}</p>
              <button onClick={() => { setError(null); clearError(); }} className="text-destructive">✕</button>
            </div>
          )}
          {success && (
            <div className="mb-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg flex justify-between">
              <p className="text-green-300 text-sm">{success}</p>
              <button onClick={() => setSuccess(null)} className="text-green-300">✕</button>
            </div>
          )}

          {/* Mobile Scrollable Tabs */}
          {/* Mobile Grid Layout Tabs (2x2) */}
          <div className="mb-6">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4 md:space-x-1 md:gap-0">
              {[
                { id: 'analyze', label: 'Analysis', icon: '🔍' },
                { id: 'compare', label: 'Compare', icon: '⚖️' },
                { id: 'corpus', label: 'Corpora', icon: '📚' },
                { id: 'files', label: 'Files', icon: '📁' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex flex-col md:flex-row items-center justify-center py-3 md:py-2 px-2 rounded-lg md:rounded-md text-sm transition-colors border ${activeTab === tab.id
                    ? 'bg-primary text-primary-foreground border-primary shadow-sm'
                    : 'bg-muted hover:bg-muted/50 border-transparent text-muted-foreground'
                    }`}
                >
                  <span className="mb-1 md:mb-0 md:mr-2 text-lg md:text-base">{tab.icon}</span>
                  <span className="text-xs md:text-sm font-medium">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* === 1. FILE ANALYSIS TAB === */}
          {activeTab === 'files' && (
            <div className="space-y-6">
              <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
                <h2 className="text-xl font-bold mb-4 text-foreground">Upload Files</h2>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${isDragOver ? 'border-primary bg-primary/10' : 'border-border bg-muted/30'}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-sm md:text-lg font-medium text-foreground mb-2">Drop files or click to browse</p>
                  <Button onClick={() => fileInputRef.current?.click()} disabled={processingFiles} variant="outline">
                    <FileText className="mr-2 h-4 w-4" /> Select Files
                  </Button>
                  <input ref={fileInputRef} type="file" multiple accept=".txt,.md,.csv,.json,.xml" className="hidden" onChange={(e) => handleFileUpload(e.target.files)} />
                </div>

                {uploadedFiles.length > 0 && (
                  <div className="mt-6">
                    <h3 className="font-medium mb-3">Uploaded ({uploadedFiles.length})</h3>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {uploadedFiles.map(file => (
                        <div key={file.id} className="flex justify-between p-3 bg-muted rounded-lg text-sm">
                          <span className="truncate max-w-[200px]">{file.name}</span>
                          <Button size="sm" variant="ghost" onClick={() => removeFile(file.id)}><X className="h-4 w-4" /></Button>
                        </div>
                      ))}
                    </div>
                    <Button className="mt-4 w-full" onClick={analyzeUploadedFiles} disabled={analyzing}>
                      {analyzing ? 'Analyzing...' : 'Analyze Files'}
                    </Button>
                  </div>
                )}
              </div>

              <ResultDisplay result={analysisResult} title="Analysis Results" onSave={handleSaveResult} />

              {savedResults.length > 0 && (
                <div className="bg-card rounded-lg p-4 mt-6 border border-border">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="font-bold">Saved Results ({savedResults.length})</h3>
                    <div className="space-x-2">
                      <Button size="sm" onClick={handleAverageResults}>Average</Button>
                      <Button size="sm" variant="destructive" onClick={handleClearSaved}>Clear</Button>
                    </div>
                  </div>
                  {averagedResult && <ResultDisplay result={averagedResult} title="Averaged Results" isAveraged={true} />}
                </div>
              )}
            </div>
          )}

          {/* === 2. STATEMENT ANALYSIS TAB === */}
          {activeTab === 'analyze' && (
            <div className="space-y-6">
              <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border">
                <h2 className="text-xl font-bold mb-4 text-foreground">Analyze Authorship</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2 text-foreground">Select Public Figure</label>
                    <select
                      value={selectedFigure}
                      onChange={(e) => setSelectedFigure(e.target.value)}
                      className="w-full p-3 border border-border rounded-lg bg-background text-foreground"
                      disabled={analyzing}
                    >
                      <option value="">Choose a figure...</option>
                      {availableFigures.map(figure => (
                        <option key={figure.name} value={figure.name}>{figure.name} ({figure.corpus_size} docs)</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="mt-6">
                  <label className="block text-sm font-medium mb-2 text-foreground">Statement to Analyze</label>
                  <textarea
                    value={statement}
                    onChange={(e) => setStatement(e.target.value)}
                    placeholder="Enter the disputed statement..."
                    className="w-full h-40 p-3 border border-border rounded-lg bg-background text-foreground"
                    disabled={analyzing}
                  />
                </div>

                <button
                  onClick={analyzeStatement}
                  disabled={analyzing || !statement.trim() || !selectedFigure}
                  className="mt-4 w-full md:w-auto px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:bg-muted"
                >
                  {analyzing ? 'Analyzing...' : '🔍 Analyze Statement'}
                </button>

                {analyzing && (
                  <div className="mt-4 p-4 bg-muted rounded-lg border border-border">
                    <div className="flex justify-between text-sm mb-2">
                      <span>{analysisStatus}</span>
                      <span>{analysisProgress}%</span>
                    </div>
                    <div className="w-full bg-muted-foreground/20 rounded-full h-2">
                      <div className="bg-primary h-2 rounded-full transition-all duration-300" style={{ width: `${analysisProgress}%` }} />
                    </div>
                  </div>
                )}
              </div>

              <ResultDisplay result={analysisResult} title="Analysis Results" onSave={handleSaveResult} />
            </div>
          )}

          {/* === 3. COMPARE TAB === */}
          {activeTab === 'compare' && (
            <div className="space-y-6">
              <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border">
                <h2 className="text-xl font-bold mb-4 text-foreground">Compare Two Texts</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">Text 1</label>
                    <textarea
                      value={text1}
                      onChange={(e) => setText1(e.target.value)}
                      className="w-full h-40 p-3 border border-border rounded-lg bg-background text-foreground"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Text 2</label>
                    <textarea
                      value={text2}
                      onChange={(e) => setText2(e.target.value)}
                      className="w-full h-40 p-3 border border-border rounded-lg bg-background text-foreground"
                    />
                  </div>
                </div>
                <button
                  onClick={compareTexts}
                  disabled={comparing || !text1.trim() || !text2.trim()}
                  className="mt-4 w-full md:w-auto px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-muted"
                >
                  {comparing ? 'Comparing...' : '⚖️ Compare Texts'}
                </button>
              </div>

              {comparisonResult && (
                <ResultDisplay
                  result={{ similarity_scores: comparisonResult.similarity_scores, interpretation: 'Comparison Complete' }}
                  title="Comparison Results"
                />
              )}
            </div>
          )}

          {/* === 4. CORPUS & MODELS TAB === */}
          {activeTab === 'corpus' && (
            <div className="space-y-6">

              {/* Build Corpus Section */}
              <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border">
                <h2 className="text-xl font-bold mb-4">Build New Corpus</h2>

                <div className="space-y-4">
                  <input
                    type="text"
                    value={newFigureName}
                    onChange={(e) => setNewFigureName(e.target.value)}
                    placeholder="Public Figure Name (e.g. Elon Musk)"
                    className="w-full p-3 border border-border rounded-lg bg-background text-foreground"
                  />

                  {/* Build Method Radio */}
                  <div className="flex flex-col md:flex-row gap-4">
                    <label className="flex items-center space-x-2 border p-3 rounded-lg cursor-pointer hover:bg-muted/50">
                      <input type="radio" value="auto" checked={buildMethod === 'auto'} onChange={(e) => setBuildMethod(e.target.value)} />
                      <span>Auto-scrape (Web)</span>
                    </label>
                    <label className="flex items-center space-x-2 border p-3 rounded-lg cursor-pointer hover:bg-muted/50">
                      <input type="radio" value="upload" checked={buildMethod === 'upload'} onChange={(e) => setBuildMethod(e.target.value)} />
                      <span>Upload Files</span>
                    </label>
                  </div>

                  {/* Auto Scrape Options */}
                  {buildMethod === 'auto' && (
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <label className="block text-sm font-medium mb-2">Sources</label>
                      <div className="flex flex-wrap gap-3 mb-4">
                        {['twitter', 'speeches', 'interviews', 'articles'].map(p => (
                          <label key={p} className="flex items-center space-x-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={selectedPlatforms.includes(p)}
                              onChange={(e) => {
                                if (e.target.checked) setSelectedPlatforms([...selectedPlatforms, p]);
                                else setSelectedPlatforms(selectedPlatforms.filter(x => x !== p));
                              }}
                            />
                            <span className="capitalize">{p}</span>
                          </label>
                        ))}
                      </div>
                      <label className="block text-sm font-medium mb-2">Max Docs: {maxDocuments}</label>
                      <input
                        type="range" min="100" max="5000" step="100"
                        value={maxDocuments} onChange={(e) => setMaxDocuments(parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  )}

                  {/* Upload Options */}
                  {buildMethod === 'upload' && (
                    <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                      <Button variant="outline" onClick={() => fileInputRefCorpus.current?.click()}>
                        <Upload className="mr-2 h-4 w-4" /> Select Files
                      </Button>
                      <input ref={fileInputRefCorpus} type="file" multiple className="hidden" onChange={(e) => handleCorpusFileUpload(e.target.files)} />
                      {corpusFiles.length > 0 && <p className="mt-2 text-sm">{corpusFiles.length} files selected</p>}
                    </div>
                  )}
                </div>

                <button
                  onClick={buildCorpus}
                  disabled={buildingCorpus}
                  className="mt-4 w-full md:w-auto px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-muted"
                >
                  {buildingCorpus ? 'Building...' : '📚 Build Corpus'}
                </button>
                {corpusProgress.status && <p className="mt-2 text-primary">{corpusProgress.message || corpusProgress.status}</p>}
              </div>

              {/* Embedding Models Grid (RESTORED FULL LIST) */}
              <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border">
                <h2 className="text-xl font-bold mb-4">Embedding Models</h2>
                <div className="p-3 bg-muted rounded-lg mb-4 flex justify-between items-center">
                  <span className="font-medium">Active:</span>
                  <span className="px-2 py-1 bg-primary/20 text-primary rounded">{embeddingModels.active_model}</span>
                </div>

                {/* GME Special Highlight */}
                {!gmeAvailable && (
                  <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg mb-4">
                    <h4 className="font-bold text-blue-400">Upgrade to GME-Qwen2-VL</h4>
                    <p className="text-sm text-muted-foreground mb-3">State-of-the-art multimodal embeddings (4096D).</p>
                    <Button onClick={() => initializeEmbeddingModel('gme', 0)} disabled={initializingGME}>
                      {initializingGME ? 'Initializing...' : 'Initialize GME'}
                    </Button>
                  </div>
                )}

                {/* Full Grid of Models - Responsive (1 col mobile, 3 col desktop) */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries({
                    'gme': 'GME-Qwen2', 'mxbai_large': 'MXBai Large', 'multilingual_e5': 'E5 Multi',
                    'qwen3_8b': 'Qwen3 8B', 'qwen3_4b': 'Qwen3 4B', 'frida': 'FRIDA',
                    'bge_m3': 'BGE-M3', 'gte_qwen2': 'GTE-Qwen2', 'inf_retriever': 'INF Retriever',
                    'sentence_t5': 'Sentence T5', 'star': 'STAR', 'roberta': 'RoBERTa',
                    'jina_v3': 'Jina V3', 'nomic_v1_5': 'Nomic 1.5', 'arctic_embed': 'Arctic Embed'
                  }).map(([key, name]) => {
                    const info = embeddingModels.models?.[key];
                    return (
                      <div key={key} className={`p-3 rounded-lg border ${info?.enabled ? 'border-green-500/50 bg-green-500/10' : 'border-border bg-card'}`}>
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-medium text-sm">{name}</span>
                          <div className={`w-2 h-2 rounded-full ${info?.enabled ? 'bg-green-500' : 'bg-muted-foreground'}`} />
                        </div>
                        <div className="text-xs text-muted-foreground mb-2">{info?.dimensions ? `${info.dimensions}D` : 'Not Loaded'}</div>
                        <Button
                          size="sm"
                          variant={info?.enabled ? "secondary" : "outline"}
                          className="w-full h-8 text-xs"
                          onClick={() => initializeEmbeddingModel(key, 0)}
                          disabled={initializingGME || info?.enabled}
                        >
                          {info?.enabled ? 'Ready' : 'Load'}
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Existing Corpora List */}
              <div className="bg-card rounded-lg p-4 md:p-6 shadow-lg border border-border">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Existing Corpora</h2>
                  <Button size="sm" variant="ghost" onClick={loadAvailableFigures}><RotateCcw className="h-4 w-4" /></Button>
                </div>

                {loading ? (
                  <p className="text-center py-4">Loading...</p>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {availableFigures.map(f => (
                      <div key={f.name} className="border border-border rounded-lg p-4 flex flex-col justify-between">
                        <div>
                          <div className="flex justify-between">
                            <h3 className="font-bold">{f.name}</h3>
                            <button onClick={() => deleteCorpus(f.name)} className="text-destructive">🗑️</button>
                          </div>
                          <p className="text-sm text-muted-foreground">{f.corpus_size} documents</p>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {f.platforms?.map(p => (
                              <span key={p} className="text-xs bg-muted px-2 py-1 rounded capitalize">{p}</span>
                            ))}
                          </div>
                        </div>
                        <Button
                          className="mt-3 w-full"
                          onClick={() => { setSelectedFigure(f.name); setActiveTab('analyze'); }}
                        >
                          Use for Analysis
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ForensicLinguistics;
