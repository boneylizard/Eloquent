import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Upload, FileText, X, Search, RotateCcw } from 'lucide-react';


const ForensicLinguistics = () => {
    const getScoreInterpretation = (score) => {
    if (score >= 0.8) return "Very high similarity - likely same author";
    if (score >= 0.6) return "High similarity - probably same author";
    if (score >= 0.4) return "Moderate similarity - uncertain authorship";
    if (score >= 0.2) return "Low similarity - probably different author";
    return "Very low similarity - likely different author";
  };
  const { BACKEND, clearError, apiError, setTaskProgress, taskProgress } = useApp();

const [embeddingModels, setEmbeddingModels] = useState({
  models: {
    gme: { enabled: false, dimensions: null, priority: 1 },
    star: { enabled: false, dimensions: 1024, priority: 2 },
    roberta: { enabled: false, dimensions: 768, priority: 3 }
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


  useEffect(() => {
  console.log("🐛 useEffect triggered, analysisTaskId:", analysisTaskId); // ADD THIS
  // If there's no active analysis task, do nothing.
  if (!analysisTaskId) {
    return;
  }

 

  // Helper to get interpretation text based on score


    // Start polling the backend every 500 milliseconds.
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${BACKEND}/forensic/progress/${analysisTaskId}`);
        
        // Handle cases where the task is not found (e.g., server restart)
        if (response.status === 404) {
          setError("Analysis task not found. It may have expired or been interrupted.");
          clearInterval(interval);
          setAnalysisTaskId(null);
          setAnalyzing(false);
          return;
        }

        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();
       
        // --- THIS IS THE FIX ---
        // Update the local state variables that are directly tied to the UI.
        setAnalysisProgress(data.progress);
        setAnalysisStatus(data.status);
      
        // Check if the task is complete.
        if (data.progress >= 100) {
          clearInterval(interval); // Stop polling
          setAnalysisTaskId(null); // Clear the task ID
          setAnalyzing(false);     // Reset the analyzing state

          // Handle the final result
          if (data.result && !data.result.error) {
            setAnalysisResult(data.result);
            setSuccess("Analysis complete.");
          } else {
            setError(data.status || "Analysis failed or returned an error.");
            setAnalysisResult(null);
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
        setError("Failed to get analysis progress. Check the console for details.");
        clearInterval(interval);
        setAnalysisTaskId(null);
        setAnalyzing(false);
      }
    }, 500); // Poll every 500 milliseconds

    // Cleanup function: stop polling when the component unmounts or the task ID changes.
    return () => clearInterval(interval);
  }, [analysisTaskId, BACKEND]); // This effect re-runs only when analysisTaskId changes.
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
  }
}, [BACKEND]);
  // Load available figures on component mount
  useEffect(() => {
    loadAvailableFigures();
  }, []);

  const loadAvailableFigures = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BACKEND}/forensic/available-figures`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setAvailableFigures(data.figures || []);
    } catch (err) {
      console.error('Error loading figures:', err);
      setError(`Failed to load available figures: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [BACKEND]);

  // File handling functions
  const handleFileUpload = useCallback(async (files) => {
    const validFiles = Array.from(files).filter(file => {
      const validTypes = ['.txt', '.md', '.csv', '.json', '.xml'];
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return validTypes.includes(extension) && file.size <= 10 * 1024 * 1024; // 10MB limit
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

  const readFileAsText = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

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
      setAnalysisResult(null); // Clear previous results
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
 
  setAnalysisTaskId(data.task_id); // This will trigger the polling useEffect
  
      } else {
        throw new Error("Backend did not return a task ID.");
      }

    } catch (err) {
      console.error('Error starting analysis:', err);
      setError(`Failed to start analysis: ${err.message}`);
      setAnalyzing(false);
    } 
    // Note: We no longer setAnalyzing(false) here, as the polling effect will do it on completion.
  }, [statement, selectedFigure, BACKEND]);

  const analyzeUploadedFiles = useCallback(async () => {
    if (uploadedFiles.length === 0) {
      setError('Please upload files first');
      return;
    }

    try {
      setAnalyzing(true);
      setError(null);

      // Combine all file contents for analysis
      const combinedText = uploadedFiles.map(file => 
        `=== ${file.name} ===\n${file.content}\n\n`
      ).join('');

      const response = await fetch(`${BACKEND}/forensic/extract-features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: combinedText })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      setAnalysisResult({
        ...data,
        file_analysis: true,
        files_analyzed: uploadedFiles.map(f => ({ name: f.name, size: f.size }))
      });
    } catch (err) {
      console.error('Error analyzing files:', err);
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
      console.error('Error comparing texts:', err);
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
        // Build corpus from uploaded files
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
          message: `Built corpus from ${data.corpus_stats.files_processed} files (${data.corpus_stats.total_documents} documents, ${data.corpus_stats.total_words.toLocaleString()} words)`
        });
      } else {
        // Auto-scrape corpus (original functionality)
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
      
      // Refresh available figures
      setTimeout(() => {
        loadAvailableFigures();
        setCorpusProgress({});
      }, 2000);
      
    } catch (err) {
      console.error('Error building corpus:', err);
      setError(`Corpus building failed: ${err.message}`);
      setCorpusProgress({});
    } finally {
      setBuildingCorpus(false);
    }
  }, [newFigureName, selectedPlatforms, maxDocuments, buildMethod, corpusFiles, BACKEND, loadAvailableFigures]);

  const handleCorpusFileUpload = useCallback((files) => {
    const validFiles = Array.from(files).filter(file => {
      const validTypes = ['.txt', '.md', '.csv', '.json', '.xml'];
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return validTypes.includes(extension) && file.size <= 10 * 1024 * 1024; // 10MB limit
    });

    if (validFiles.length === 0) {
      setError('Please upload valid files (.txt, .md, .csv, .json, .xml) under 10MB');
      return;
    }

    const newFiles = validFiles.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      size: file.size,
      type: file.type || 'text/plain',
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
      const response = await fetch(`${BACKEND}/forensic/corpus/${encodeURIComponent(figureName)}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      await loadAvailableFigures();
      if (selectedFigure === figureName) {
        setSelectedFigure('');
      }
    } catch (err) {
      console.error('Error deleting corpus:', err);
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
        setSuccess(`${modelType.toUpperCase()} model initialized successfully on GPU ${gpuId}!`);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to initialize ${modelType}`);
      }
    } catch (err) {
      console.error(`Error initializing ${modelType}:`, err);
      setError(`Failed to initialize ${modelType}: ${err.message}`);
    } finally {
      setInitializingGME(false);
    }
  }, [BACKEND, fetchEmbeddingStatus]);
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

    const handleSetActiveModel = useCallback(async (modelKey) => {
    try {
      setError(null);
      const response = await fetch(`${BACKEND}/forensic/set-active-embedding-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey })
      });

      if (response.ok) {
        const data = await response.json();
        setSuccess(`Successfully set ${modelKey.toUpperCase()} as the active model.`);
        setEmbeddingModels(data.embedding_status);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to set active model`);
      }
    } catch (err) {
      setError(`Error setting active model: ${err.message}`);
    }
  }, [BACKEND]);    
  const initializeGME = useCallback(async (gpuId = 0) => {
    // This now correctly calls the single, unified endpoint
    await initializeEmbeddingModel('gme', gpuId);
  }, [initializeEmbeddingModel]);
  const handleSaveResult = useCallback(() => {
    if (!analysisResult) return;

    // Add the active model name and a unique ID to the result before saving
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
      lexical_similarity: 0,
      syntactic_similarity: 0,
      semantic_similarity: 0,
      stylistic_similarity: 0,
      confidence: 0,
    };

    savedResults.forEach(result => {
      for (const key in avgScores) {
        avgScores[key] += result.similarity_scores[key] / total;
      }
    });
    
    const interpretation = getScoreInterpretation(avgScores.overall_similarity);

    setAveragedResult({
      person_analyzed: savedResults[0].person_analyzed,
      interpretation: interpretation,
      similarity_scores: avgScores,
      models_averaged: savedResults.map(r => r.modelUsed),
    });
    setSuccess(`Averaged ${total} results.`);
  }, [savedResults]);

useEffect(() => {
  fetchEmbeddingStatus();
}, [fetchEmbeddingStatus]);
const ResultDisplay = ({ result, title, onSave, isAveraged = false }) => {
  if (!result) return null;

  const scores = result.similarity_scores;
  if (!scores) return null;

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  const formatScore = (score) => {
    if (typeof score !== 'number') return 'N/A';
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-foreground">{title}</h3>
          <p className="text-muted-foreground">
            {isAveraged 
              ? `Averaged across: ${result.models_averaged.join(', ')}`
              : `Statement analyzed against ${result.person_analyzed}`
            }
          </p>
        </div>
        {!isAveraged && onSave && (
          <Button onClick={onSave} size="sm">
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
            className={`h-3 rounded-full transition-all duration-500 ${
              scores.overall_similarity >= 0.8 ? 'bg-green-500' :
              scores.overall_similarity >= 0.6 ? 'bg-yellow-500' :
              scores.overall_similarity >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
            }`}
            style={{ width: `${scores.overall_similarity * 100}%` }}
          />
        </div>
        <p className="text-sm text-muted-foreground mt-2">{result.interpretation}</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Score Breakdown */}
        <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
          <div className="font-semibold text-blue-300">Lexical Similarity</div>
          <div className="text-2xl font-bold text-blue-400">{formatScore(scores.lexical_similarity)}</div>
          <div className="text-xs text-blue-300">Word choice & vocabulary</div>
        </div>
        <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/20">
          <div className="font-semibold text-green-300">Syntactic Similarity</div>
          <div className="text-2xl font-bold text-green-400">{formatScore(scores.syntactic_similarity)}</div>
          <div className="text-xs text-green-300">Sentence structure</div>
        </div>
        <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
          <div className="font-semibold text-purple-300">Semantic Similarity</div>
          <div className="text-2xl font-bold text-purple-400">{formatScore(scores.semantic_similarity)}</div>
          <div className="text-xs text-purple-300">Meaning & context</div>
        </div>
        <div className="p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
          <div className="font-semibold text-orange-300">Stylistic Similarity</div>
          <div className="text-2xl font-bold text-orange-400">{formatScore(scores.stylistic_similarity)}</div>
          <div className="text-xs text-orange-300">Writing style</div>
        </div>
      </div>
    </div>
  );
};
const EmbeddingModelSection = () => (
  <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
    <h3 className="text-xl font-bold mb-4 text-foreground">Embedding Models</h3>
    
    {/* Current Active Model */}
    <div className="mb-4 p-3 bg-muted rounded-lg">
      <div className="flex items-center justify-between">
        <span className="font-medium">Active Model:</span>
        <span className={`px-2 py-1 rounded text-sm font-medium ${
          embeddingModels.active_model === 'GME' ? 'bg-green-100 text-green-800' :
          embeddingModels.active_model === 'STAR' ? 'bg-blue-100 text-blue-800' :
          embeddingModels.active_model === 'ROBERTA' ? 'bg-purple-100 text-purple-800' :
          'bg-gray-100 text-gray-800'
        }`}>
          {embeddingModels.active_model}
          {embeddingModels.active_model !== 'None' && embeddingModels.models[embeddingModels.active_model.toLowerCase()]?.dimensions && 
            ` (${embeddingModels.models[embeddingModels.active_model.toLowerCase()].dimensions}D)`
          }
        </span>
      </div>
    </div>

    {/* Model Status Grid */}
<div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
  {Object.entries(embeddingModels.models).map(([modelName, modelInfo]) => (
    <div key={modelName} className={`p-3 rounded-lg border ${
      modelInfo.enabled ? 'border-green-300 bg-green-50' : 'border-gray-300 bg-gray-50'
    }`}>
      <div className="flex items-center justify-between mb-1">
        <span className="font-medium text-sm uppercase">{modelName.replace('_', '-')}</span>
        <span className={`w-2 h-2 rounded-full ${
          modelInfo.enabled ? 'bg-green-500' : 'bg-gray-400'
        }`}></span>
      </div>
      <div className="text-xs text-muted-foreground">
        {modelInfo.enabled ? `${modelInfo.dimensions}D` : 'Not loaded'}
      </div>
      <div className="text-xs text-muted-foreground">
        Priority: {modelInfo.priority}
      </div>
    </div>
  ))}
</div>

    {/* GME Initialization */}
    {!gmeAvailable && (
      <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="flex items-start justify-between">
          <div>
            <h4 className="font-medium text-blue-900 mb-1">Upgrade to GME Embeddings</h4>
            <p className="text-sm text-blue-700 mb-3">
              GME-Qwen2-VL provides much higher quality embeddings (4096D) specifically designed for 
              authorship attribution and forensic analysis. This will significantly improve accuracy.
            </p>
          </div>
        </div>
        
        <button
          onClick={() => initializeGME(0)} // Default to GPU 0
          disabled={initializingGME}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white py-2 px-4 rounded-lg font-medium transition-colors"
        >
          {initializingGME ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Initializing GME Model...
            </div>
          ) : (
            'Initialize GME Model (GPU 0)'
          )}
        </button>
      </div>
    )}

    {/* GME Active Status */}
    {gmeAvailable && (
      <div className="p-4 bg-green-50 rounded-lg border border-green-200">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
          <div>
            <h4 className="font-medium text-green-900">GME Model Active</h4>
            <p className="text-sm text-green-700">
              Using high-quality GME embeddings for enhanced forensic analysis
            </p>
          </div>
        </div>
      </div>
    )}

{/* Other Model Initializers */}
<div className="mt-4 p-4 bg-muted rounded-lg border border-border">
  <h4 className="font-medium text-foreground mb-3">Initialize Other Models (GPU 0)</h4>
  <div className="space-y-2">
    {[
      { type: 'inf_retriever', name: 'INF-Retriever-v1' },
      { type: 'gte_qwen2', name: 'GTE-Qwen2-7B' }
    ].map(model => (
      <div key={model.type} className="flex items-center justify-between">
        <span className="text-sm text-foreground">{model.name}</span>
        <button
          onClick={() => initializeEmbeddingModel(model.type, 0)}
          disabled={initializingGME || embeddingModels.models?.[model.type]?.enabled}
          className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground text-xs py-1 px-2 rounded"
        >
          {embeddingModels.models?.[model.type]?.enabled ? 'Loaded' : initializingGME ? '...' : 'Load'}
        </button>
      </div>
    ))}
  </div>
</div>



    {/* Refresh Button */}
    <div className="mt-4">
      <button
        onClick={fetchEmbeddingStatus}
        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        🔄 Refresh Status
      </button>
    </div>
  </div>
);

  const renderAnalysisResults = () => {
    if (!analysisResult) return null;

    if (analysisResult.file_analysis) {
      // File analysis results
      const features = analysisResult.features;
      return (
        <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
          <div className="mb-6">
            <h3 className="text-xl font-bold mb-2 text-foreground">File Analysis Results</h3>
            <p className="text-muted-foreground">Stylometric features extracted from uploaded files</p>
            <div className="mt-2 text-sm text-muted-foreground">
              Files analyzed: {analysisResult.files_analyzed?.map(f => f.name).join(', ')}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
              <h4 className="font-semibold text-blue-300 mb-2">Lexical Features</h4>
              <div className="space-y-1 text-sm text-foreground">
                <div>Avg Word Length: {features.lexical_features?.avg_word_length}</div>
                <div>Vocab Richness: {features.lexical_features?.vocab_richness}</div>
                <div>Sentence Length: {features.lexical_features?.avg_sentence_length}</div>
              </div>
            </div>
            
            <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
              <h4 className="font-semibold text-green-300 mb-2">Stylistic Features</h4>
              <div className="space-y-1 text-sm text-foreground">
                <div>Questions: {formatScore(features.stylistic_features?.question_ratio)}</div>
                <div>Exclamations: {formatScore(features.stylistic_features?.exclamation_ratio)}</div>
                <div>Passive Voice: {formatScore(features.stylistic_features?.passive_voice_ratio)}</div>
              </div>
            </div>
            
            <div className="p-4 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <h4 className="font-semibold text-purple-300 mb-2">Text Statistics</h4>
              <div className="space-y-1 text-sm text-foreground">
                <div>Characters: {features.text_statistics?.character_count}</div>
                <div>Words: {features.text_statistics?.word_count}</div>
                <div>Sentences: {features.text_statistics?.sentence_count}</div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // Regular analysis results
    const scores = analysisResult.similarity_scores;
if (!scores) {
  return (
    <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
      <h3 className="text-xl font-bold mb-2 text-foreground">Analysis Error</h3>
      <p className="text-destructive">
        {analysisResult.error || "Analysis failed - please try again"}
      </p>
    </div>
  );
}
    
    return (
      <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
        <div className="mb-6">
          <h3 className="text-xl font-bold mb-2 text-foreground">Analysis Results</h3>
          <p className="text-muted-foreground">Statement analyzed against {analysisResult.person_analyzed}</p>
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
              className={`h-3 rounded-full transition-all duration-500 ${
                scores.overall_similarity >= 0.8 ? 'bg-green-500' :
                scores.overall_similarity >= 0.6 ? 'bg-yellow-500' :
                scores.overall_similarity >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
              }`}
              style={{ width: `${scores.overall_similarity * 100}%` }}
            />
          </div>
          <p className="text-sm text-muted-foreground mt-2">{analysisResult.interpretation}</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
            <div className="font-semibold text-blue-300">Lexical Similarity</div>
            <div className="text-2xl font-bold text-blue-400">{formatScore(scores.lexical_similarity)}</div>
            <div className="text-xs text-blue-300">Word choice & vocabulary</div>
          </div>
          
          <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/20">
            <div className="font-semibold text-green-300">Syntactic Similarity</div>
            <div className="text-2xl font-bold text-green-400">{formatScore(scores.syntactic_similarity)}</div>
            <div className="text-xs text-green-300">Sentence structure</div>
          </div>
          
          <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
            <div className="font-semibold text-purple-300">Semantic Similarity</div>
            <div className="text-2xl font-bold text-purple-400">{formatScore(scores.semantic_similarity)}</div>
            <div className="text-xs text-purple-300">Meaning & context</div>
          </div>
          
          <div className="p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
            <div className="font-semibold text-orange-300">Stylistic Similarity</div>
            <div className="text-2xl font-bold text-orange-400">{formatScore(scores.stylistic_similarity)}</div>
            <div className="text-xs text-orange-300">Writing style</div>
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonResults = () => {
    if (!comparisonResult) return null;

    const scores = comparisonResult.similarity_scores;
    
    return (
      <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
        <div className="mb-6">
          <h3 className="text-xl font-bold mb-2 text-foreground">Comparison Results</h3>
          <p className="text-muted-foreground">{text1Label} vs {text2Label}</p>
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
              className={`h-3 rounded-full transition-all duration-500 ${
                scores.overall_similarity >= 0.8 ? 'bg-green-500' :
                scores.overall_similarity >= 0.6 ? 'bg-yellow-500' :
                scores.overall_similarity >= 0.4 ? 'bg-orange-500' : 'bg-red-500'
              }`}
              style={{ width: `${scores.overall_similarity * 100}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-4 gap-3">
          <div className="text-center p-3 bg-blue-500/10 rounded border border-blue-500/20">
            <div className="font-semibold text-blue-300">Lexical</div>
            <div className="text-lg font-bold text-blue-400">{formatScore(scores.lexical_similarity)}</div>
          </div>
          <div className="text-center p-3 bg-green-500/10 rounded border border-green-500/20">
            <div className="font-semibold text-green-300">Syntactic</div>
            <div className="text-lg font-bold text-green-400">{formatScore(scores.syntactic_similarity)}</div>
          </div>
          <div className="text-center p-3 bg-purple-500/10 rounded border border-purple-500/20">
            <div className="font-semibold text-purple-300">Semantic</div>
            <div className="text-lg font-bold text-purple-400">{formatScore(scores.semantic_similarity)}</div>
          </div>
          <div className="text-center p-3 bg-orange-500/10 rounded border border-orange-500/20">
            <div className="font-semibold text-orange-300">Stylistic</div>
            <div className="text-lg font-bold text-orange-400">{formatScore(scores.stylistic_similarity)}</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 max-w-6xl mx-auto bg-background text-foreground">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 text-foreground">Forensic Linguistics Analysis</h1>
        <p className="text-muted-foreground">
          AI-powered stylometric analysis for authorship attribution and text verification
        </p>
      </div>

      {/* Error Display */}
      {(error || apiError) && (
        <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <div className="flex justify-between items-center">
            <p className="text-destructive">{error || apiError}</p>
            <button 
              onClick={() => { setError(null); clearError(); }}
              className="text-destructive hover:text-destructive/80"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      {/* Success Display */}
      {success && (
        <div className="mb-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex justify-between items-center">
            <p className="text-green-300">{success}</p>
            <button 
              onClick={() => setSuccess(null)}
              className="text-green-300 hover:text-green-300/80"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      {/* Tab Navigation */}
      <div className="mb-6">
        <div className="flex space-x-1 bg-muted p-1 rounded-lg">
          {[
            { id: 'analyze', label: 'Statement Analysis', icon: '🔍' },
            { id: 'compare', label: 'Text Comparison', icon: '⚖️' },
            { id: 'corpus', label: 'Manage Corpora', icon: '📚' },
            { id: 'files', label: 'File Analysis', icon: '📁' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center py-2 px-4 rounded-md transition-colors ${
                activeTab === tab.id 
                  ? 'bg-background shadow-sm border border-border' 
                  : 'hover:bg-muted/50'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* File Analysis Tab */}
      {activeTab === 'files' && (
        <div className="space-y-6">
          {/* File Upload Area */}
          <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
            <h2 className="text-xl font-bold mb-4 text-foreground">Upload Files for Analysis</h2>
            
            {/* Drag and Drop Area */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragOver ? 'border-primary bg-primary/10' : 'border-border bg-muted/30'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium text-foreground mb-2">
                Drop files here or click to browse
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                Supports: .txt, .md, .csv, .json, .xml (max 10MB each)
              </p>
              <Button 
                onClick={() => fileInputRef.current?.click()}
                disabled={processingFiles}
              >
                <FileText className="mr-2 h-4 w-4" />
                Select Files
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.csv,.json,.xml"
                className="hidden"
                onChange={(e) => handleFileUpload(e.target.files)}
              />
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="mt-6">
                <h3 className="font-medium mb-3 text-foreground">Uploaded Files ({uploadedFiles.length})</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {uploadedFiles.map(file => (
                    <div key={file.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex items-center space-x-3">
                        <FileText className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium text-sm text-foreground">{file.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {(file.size / 1024).toFixed(1)} KB • {file.type}
                          </p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeFile(file.id)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <Button
                  className="mt-4 w-full"
                  onClick={analyzeUploadedFiles}
                  disabled={analyzing || uploadedFiles.length === 0}
                >
                  {analyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                      Analyzing Files...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-4 w-4" />
                      Analyze Files
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>

              {analysisResult && (
      <ResultDisplay 
        result={analysisResult} 
        title="Analysis Results"
        onSave={handleSaveResult}
      />
    )}

    {/* Section for saved and averaged results */}
    {savedResults.length > 0 && (
      <div className="bg-card rounded-lg p-6 shadow-lg border border-border mt-6">
        <h3 className="text-xl font-bold mb-4 text-foreground">Saved Results</h3>
        <div className="space-y-2 mb-4">
          {savedResults.map((res) => (
            <div key={res.id} className="flex justify-between items-center p-2 bg-muted rounded-lg">
              <span className="font-medium text-foreground">
                {res.modelUsed}: <span className={getScoreColor(res.similarity_scores.overall_similarity)}>{formatScore(res.similarity_scores.overall_similarity)}</span>
              </span>
              <Button variant="ghost" size="sm" onClick={() => setSavedResults(prev => prev.filter(r => r.id !== res.id))}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
        <div className="flex space-x-2">
          <Button onClick={handleAverageResults} disabled={savedResults.length < 2}>
            📊 Average {savedResults.length} Results
          </Button>
          <Button variant="destructive" onClick={handleClearSaved}>
            🗑️ Clear All
          </Button>
        </div>
      </div>
    )}

    {averagedResult && (
       <ResultDisplay 
        result={averagedResult} 
        title="Averaged Results"
        isAveraged={true}
      />
    )}
        </div>
      )}

{/* Statement Analysis Tab */}
{activeTab === 'analyze' && (
  <div className="space-y-6">
    <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
      <h2 className="text-xl font-bold mb-4 text-foreground">Analyze Statement Authorship</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium mb-2 text-foreground">Select Public Figure</label>
          <select
            value={selectedFigure}
            onChange={(e) => setSelectedFigure(e.target.value)}
            className="w-full p-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent bg-background text-foreground"
            disabled={analyzing}
          >
            <option value="">Choose a figure...</option>
            {availableFigures.map(figure => (
              <option key={figure.name} value={figure.name}>
                {figure.name} ({figure.corpus_size} docs)
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-6">
        <label className="block text-sm font-medium mb-2 text-foreground">Statement to Analyze</label>
        <textarea
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          placeholder="Enter the disputed statement or quote here..."
          className="w-full h-32 p-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent bg-background text-foreground placeholder:text-muted-foreground"
          disabled={analyzing}
        />
        <div className="text-sm text-muted-foreground mt-1">
          {statement.length} characters • {statement.split(' ').length} words
        </div>
      </div>

      <button
        onClick={analyzeStatement}
        disabled={analyzing || !statement.trim() || !selectedFigure}
        className="mt-4 px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed disabled:text-muted-foreground flex items-center"
      >
        {analyzing ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
            Analyzing...
          </>
        ) : (
          <>🔍 Analyze Statement</>
        )}
      </button>
      {/* --- NEW: Progress Bar --- */}
      {analyzing && (
        <div className="mt-4 p-4 bg-muted rounded-lg border border-border">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-foreground">{analysisStatus}</span>
            <span className="text-sm font-bold text-primary">{analysisProgress}%</span>
          </div>
          <div className="w-full bg-muted-foreground/20 rounded-full h-2">
            <div 
              className="bg-primary h-2 rounded-full transition-all duration-300"
              style={{ width: `${analysisProgress}%` }}
            />
          </div>
        </div>
      )}
    </div>

    {renderAnalysisResults()}
  </div>
)}
{/* Text Comparison Tab */}
{activeTab === 'compare' && (
  <div className="space-y-6">
    <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
      <h2 className="text-xl font-bold mb-4 text-foreground">Compare Two Texts</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium mb-2 text-foreground">
            Text 1
          </label>
          <textarea
            value={text1}
            onChange={(e) => setText1(e.target.value)}
            placeholder="Enter first text..."
            className="w-full h-40 p-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent bg-background text-foreground placeholder:text-muted-foreground"
            disabled={comparing}
          />
          <div className="text-sm text-muted-foreground mt-1">
            {text1.length} characters
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2 text-foreground">
            Text 2
          </label>
          <textarea
            value={text2}
            onChange={(e) => setText2(e.target.value)}
            placeholder="Enter second text..."
            className="w-full h-40 p-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent bg-background text-foreground placeholder:text-muted-foreground"
            disabled={comparing}
          />
          <div className="text-sm text-muted-foreground mt-1">
            {text2.length} characters
          </div>
        </div>
      </div>

      <button
        onClick={compareTexts}
        disabled={comparing || !text1.trim() || !text2.trim()}
        className="mt-4 px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-muted disabled:cursor-not-allowed disabled:text-muted-foreground flex items-center"
      >
        {comparing ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
            Comparing...
          </>
        ) : (
          <>⚖️ Compare Texts</>
        )}
      </button>
    </div>

    {renderComparisonResults()}
  </div>
)}
{/* Corpus Management Tab */}
      {activeTab === 'corpus' && (
        <div className="space-y-6">
          {/* Build New Corpus */}
          <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
            <h2 className="text-xl font-bold mb-4 text-foreground">Build New Corpus</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-foreground">Public Figure Name</label>
                <input
                  type="text"
                  value={newFigureName}
                  onChange={(e) => setNewFigureName(e.target.value)}
                  placeholder="e.g., Donald Trump, Barack Obama"
                  className="w-full p-3 border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent bg-background text-foreground placeholder:text-muted-foreground"
                  disabled={buildingCorpus}
                />
              </div>

{/* Build Method Selection */}
<div>
  <label className="block text-sm font-medium mb-2 text-foreground">Build Method</label>
  <div className="space-y-4">
    <label className="flex items-center text-foreground">
      <input
        type="radio"
        value="auto"
        checked={buildMethod === 'auto'}
        onChange={(e) => setBuildMethod(e.target.value)}
        disabled={buildingCorpus}
        className="mr-2"
      />
      Auto-scrape from internet
    </label>
    <label className="block text-foreground">
      <div className="flex items-center">
        <input
          type="radio"
          value="upload"
          checked={buildMethod === 'upload'}
          onChange={(e) => setBuildMethod(e.target.value)}
          disabled={buildingCorpus}
          className="mr-2"
        />
        Upload your own files
      </div>
      <div className="ml-6 text-sm text-muted-foreground italic">
        recommended
      </div>
    </label>
  </div>
</div>

              {/* Auto-scrape options */}
              {buildMethod === 'auto' && (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-2 text-foreground">Data Sources</label>
                    <div className="flex flex-wrap gap-2">
                      {['twitter', 'speeches', 'interviews', 'articles'].map(platform => (
                        <label key={platform} className="flex items-center text-foreground">
                          <input
                            type="checkbox"
                            checked={selectedPlatforms.includes(platform)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedPlatforms(prev => [...prev, platform]);
                              } else {
                                setSelectedPlatforms(prev => prev.filter(p => p !== platform));
                              }
                            }}
                            disabled={buildingCorpus}
                            className="mr-2"
                          />
                          {platform.charAt(0).toUpperCase() + platform.slice(1)}
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2 text-foreground">
                      Max Documents: {maxDocuments}
                    </label>
                    <input
                      type="range"
                      min="100"
                      max="5000"
                      step="100"
                      value={maxDocuments}
                      onChange={(e) => setMaxDocuments(parseInt(e.target.value))}
                      disabled={buildingCorpus}
                      className="w-full"
                    />
                  </div>
                </>
              )}

              {/* File upload options */}
              {buildMethod === 'upload' && (
                <div>
                  <label className="block text-sm font-medium mb-2 text-foreground">Upload Files</label>
                  <div className="border-2 border-dashed border-border rounded-lg p-4 text-center">
                    <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Upload your dataset files (.txt, .md, .csv, .json, .xml)
                    </p>
                    <Button 
                      onClick={() => fileInputRefCorpus.current?.click()}
                      disabled={buildingCorpus}
                      variant="outline"
                      size="sm"
                    >
                      Select Files
                    </Button>
                    <input
                      ref={fileInputRefCorpus}
                      type="file"
                      multiple
                      accept=".txt,.md,.csv,.json,.xml"
                      className="hidden"
                      onChange={(e) => handleCorpusFileUpload(e.target.files)}
                    />
                  </div>

                  {/* Uploaded files list */}
                  {corpusFiles.length > 0 && (
                    <div className="mt-3">
                      <p className="text-sm font-medium text-foreground mb-2">
                        Selected Files ({corpusFiles.length})
                      </p>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {corpusFiles.map(file => (
                          <div key={file.id} className="flex items-center justify-between p-2 bg-muted rounded text-sm">
                            <span className="text-foreground">{file.name}</span>
                            <button
                              onClick={() => removeCorpusFile(file.id)}
                              className="text-destructive hover:text-destructive/80"
                              disabled={buildingCorpus}
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <button
              onClick={buildCorpus}
              disabled={
                buildingCorpus || 
                !newFigureName.trim() || 
                (buildMethod === 'auto' && selectedPlatforms.length === 0) ||
                (buildMethod === 'upload' && corpusFiles.length === 0)
              }
              className="mt-4 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-muted disabled:cursor-not-allowed disabled:text-muted-foreground flex items-center"
            >
              {buildingCorpus ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                  Building Corpus...
                </>
              ) : (
                <>📚 Build Corpus</>
              )}
            </button>

            {corpusProgress.status && (
              <div className="mt-4 p-3 bg-primary/10 border border-primary/20 rounded-lg">
                <p className="text-primary">{corpusProgress.message || corpusProgress.status}</p>
              </div>
            )}
          </div>

          {/* Embedding Models Section */}
          <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
            <h2 className="text-xl font-bold mb-4 text-foreground">Embedding Models</h2>
            
            <div className="mb-4 p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-between">
                <span className="font-medium text-foreground">Active Model:</span>
                <span className={`px-2 py-1 rounded text-sm font-medium ${
                  embeddingModels.active_model === 'GME' ? 'bg-green-100 text-green-800' :
                  embeddingModels.active_model === 'GTE_QWEN2' ? 'bg-blue-100 text-blue-800' :
                  embeddingModels.active_model === 'BGE_M3' ? 'bg-yellow-100 text-yellow-800' :   
                  embeddingModels.active_model === 'STAR' ? 'bg-teal-100 text-teal-800' :
                  embeddingModels.active_model === 'ROBERTA' ? 'bg-pink-100 text-pink-800' :
                  embeddingModels.active_model === 'JINA_V3' ? 'bg-purple-100 text-purple-800' :
                  embeddingModels.active_model === 'NOMIC_V1_5' ? 'bg-orange-100 text-orange-800' :
                  embeddingModels.active_model === 'ARCTIC_EMBED' ? 'bg-gray-100 text-gray-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {embeddingModels.active_model}
                </span>
              </div>
            </div>

            {/* Always show initialization options */}
            <div className="p-4 bg-muted rounded-lg border border-border mb-4">
              <h3 className="font-medium text-foreground mb-2">
                {embeddingModels.active_model === 'None' ? '🚀 Initialize Embedding Model' : '🔄 Switch or Add Models'}
              </h3>
              <p className="text-sm text-muted-foreground mb-3">
                {embeddingModels.active_model === 'None' 
                  ? 'Choose a high-quality embedding model for enhanced forensic analysis:'
                  : 'Initialize additional models or switch to a different one:'
                }
              </p>
              
              <div className="space-y-2">
                {/* GME */}
                <div className="flex items-center justify-between p-2 bg-card rounded border border-border">
                  <div>
                    <div className="font-medium text-sm text-foreground">GME-Qwen2-VL (4096D)</div>
                    <div className="text-xs text-muted-foreground">Multimodal, vision-enabled embeddings</div>
                  </div>
                  <button
                    onClick={() => initializeGME()}
                    disabled={initializingGME || embeddingModels.models?.gme?.enabled}
                    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
                  >
                    {embeddingModels.models?.gme?.enabled ? 'Ready' : initializingGME ? 'Loading...' : 'Initialize'}
                  </button>
                </div>
                
                {/* GTE-Qwen2 */}
                <div className="flex items-center justify-between p-2 bg-card rounded border border-border">
                  <div>
                    <div className="font-medium text-sm text-foreground">GTE-Qwen2-7B (3584D)</div>
                    <div className="text-xs text-muted-foreground">Large-scale multilingual embeddings</div>
                  </div>
                  <button
                    onClick={async () => {
                      try {
                        setInitializingGME(true);
                        setError(null);
                        const response = await fetch(`${BACKEND}/forensic/initialize-embedding`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ model_type: 'gte_qwen2', gpu_id: 0 })
                        });
                        if (response.ok) {
                          await fetchEmbeddingStatus();
                          setSuccess('GTE-Qwen2 model initialized successfully!');
                        } else {
                          const errorData = await response.json();
                          throw new Error(errorData.detail || 'Failed to initialize GTE-Qwen2');
                        }
                      } catch (err) {
                        setError(`Failed to initialize GTE-Qwen2: ${err.message}`);
                      } finally {
                        setInitializingGME(false);
                      }
                    }}
                    disabled={initializingGME || embeddingModels.models?.gte_qwen2?.enabled}
                    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
                  >
                    {embeddingModels.models?.gte_qwen2?.enabled ? 'Ready' : initializingGME ? 'Loading...' : 'Initialize'}
                  </button>
                </div>

                {/* Jina v3 */}
<div className="flex items-center justify-between p-2 bg-card rounded border border-border">
  <div>
    <div className="font-medium text-sm text-foreground">Jina Embeddings v3 (1024D)</div>
    <div className="text-xs text-muted-foreground">Task-specific LoRA adapters, multilingual</div>
  </div>
  <button
    onClick={() => initializeEmbeddingModel('jina_v3', 0)}
    disabled={initializingGME || embeddingModels.models?.jina_v3?.enabled}
    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
  >
    {embeddingModels.models?.jina_v3?.enabled ? 'Ready' : initializingGME ? 'Loading...' : 'Initialize'}
  </button>
</div>

{/* Nomic v1.5 */}
<div className="flex items-center justify-between p-2 bg-card rounded border border-border">
  <div>
    <div className="font-medium text-sm text-foreground">Nomic Embed v1.5 (768D)</div>
    <div className="text-xs text-muted-foreground">Popular open-source, instruction-tuned</div>
  </div>
  <button
    onClick={() => initializeEmbeddingModel('nomic_v1_5', 0)}
    disabled={initializingGME || embeddingModels.models?.nomic_v1_5?.enabled}
    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
  >
    {embeddingModels.models?.nomic_v1_5?.enabled ? 'Ready' : initializingGME ? 'Loading...' : 'Initialize'}
  </button>
</div>

{/* Arctic Embed */}
<div className="flex items-center justify-between p-2 bg-card rounded border border-border">
  <div>
    <div className="font-medium text-sm text-foreground">Arctic Embed m-v1.5 (768D)</div>
    <div className="text-xs text-muted-foreground">Production-optimized, compression-friendly</div>
  </div>
  <button
    onClick={() => initializeEmbeddingModel('arctic_embed', 0)}
    disabled={initializingGME || embeddingModels.models?.arctic_embed?.enabled}
    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
  >
    {embeddingModels.models?.arctic_embed?.enabled ? 'Ready' : initializingGME ? 'Loading...' : 'Initialize'}
  </button>
</div>

{/* BGE-M3 */}
<div className="flex items-center justify-between p-2 bg-card rounded border border-border">
  <div>
    <div className="font-medium text-sm text-foreground">BGE-M3 (1024D)</div>
    <div className="text-xs text-muted-foreground">Multi-lingual, multi-functionality</div>
                  </div>
                  <button
                    onClick={() => initializeEmbeddingModel('bge_m3', 0)}
                    disabled={initializingGME}
                    className="bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
                  >
                    {initializingGME ? 'Loading...' : 'Initialize'}
                  </button>
                </div>
                {/* STAR */}
                <div className="flex items-center justify-between p-2 bg-card rounded border border-border">
                  <div>
                    <div className="font-medium text-sm text-foreground">STAR (1024D)</div>
                    <div className="text-xs text-muted-foreground">Specialized for authorship attribution</div>
                  </div>
                  <button
                    onClick={() => handleSetActiveModel('star')}
                    disabled={embeddingModels.active_model === 'STAR' || !embeddingModels.models?.star?.enabled}
                    className="bg-primary hover:bg-primary/90 disabled:bg-muted disabled:opacity-50 text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
                  >
                    {embeddingModels.active_model === 'STAR' ? 'Active' : 'Set Active'}
                  </button>
                </div>

                {/* RoBERTa */}
                <div className="flex items-center justify-between p-2 bg-card rounded border border-border">
                  <div>
                    <div className="font-medium text-sm text-foreground">RoBERTa (768D)</div>
                    <div className="text-xs text-muted-foreground">Established semantic embeddings</div>
                  </div>
                  <button
                    onClick={() => handleSetActiveModel('roberta')}
                    disabled={embeddingModels.active_model === 'ROBERTA' || !embeddingModels.models?.roberta?.enabled}
                    className="bg-primary hover:bg-primary/90 disabled:bg-muted disabled:opacity-50 text-primary-foreground py-1 px-3 rounded text-sm font-medium transition-colors"
                  >
                    {embeddingModels.active_model === 'ROBERTA' ? 'Active' : 'Set Active'}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Available Corpora */}
          <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-foreground">Available Corpora</h2>
              <button
                onClick={loadAvailableFigures}
                disabled={loading}
                className="flex items-center px-3 py-1 text-sm bg-muted hover:bg-muted/80 rounded-lg text-foreground"
              >
                <RotateCcw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>

            {loading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
                <p className="mt-2 text-muted-foreground">Loading corpora...</p>
              </div>
            ) : availableFigures.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No corpora available. Build one to get started!</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {availableFigures.map(figure => (
                  <div key={figure.name} className="border border-border rounded-lg p-4 bg-card">
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-semibold text-lg text-foreground">{figure.name}</h3>
                      <button
                        onClick={() => deleteCorpus(figure.name)}
                        className="text-destructive hover:text-destructive/80 text-sm"
                        title="Delete corpus"
                      >
                        🗑️
                      </button>
                    </div>
                    
                    <div className="text-sm text-muted-foreground space-y-1">
                      <p><strong className="text-foreground">Documents:</strong> {figure.corpus_size.toLocaleString()}</p>
                      <p><strong className="text-foreground">Platforms:</strong> {figure.platforms.join(', ')}</p>
                      
                      <div className="mt-2">
                        <strong className="text-foreground">Platform breakdown:</strong>
                        <div className="grid grid-cols-2 gap-1 mt-1 text-xs">
                          {Object.entries(figure.platform_breakdown).map(([platform, count]) => (
                            <div key={platform} className="bg-muted rounded px-2 py-1 text-foreground">
                              {platform}: {count}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={() => {
                        setSelectedFigure(figure.name);
                        setActiveTab('analyze');
                      }}
                      className="mt-3 w-full px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 text-sm"
                    >
                      Use for Analysis
                    </button>
                  </div>
                ))}
                
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
  
};


export default ForensicLinguistics;