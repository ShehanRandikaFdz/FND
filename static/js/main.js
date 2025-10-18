/**
 * Main JavaScript for Fake News Detector
 * Handles API calls, UI updates, and history management
 */

// Global variables
let isAnalyzing = false;
let currentHistory = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('🚀 Initializing Fake News Detector...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load history
    loadHistory();
    
    // Set up dark mode toggle
    setupDarkModeToggle();
    
    console.log('✅ App initialized successfully!');
}

function setupEventListeners() {
    // Check credibility button
    const checkBtn = document.getElementById('check-credibility-btn');
    if (checkBtn) {
        checkBtn.addEventListener('click', checkCredibility);
    }
    
    
    // Clear history button
    const clearBtn = document.getElementById('clear-history-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearHistory);
    }
    
    // Analyze URL button
    const analyzeUrlBtn = document.getElementById('analyze-url-btn');
    if (analyzeUrlBtn) {
        analyzeUrlBtn.addEventListener('click', analyzeUrl);
    }
    
    // Enter key in textarea
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                checkCredibility();
            }
        });
    }
}

function setupDarkModeToggle() {
    const toggle = document.getElementById('dark-mode-toggle');
    if (toggle) {
        // Remove any existing event listeners to prevent duplicates
        const newToggle = toggle.cloneNode(true);
        toggle.parentNode.replaceChild(newToggle, toggle);
        
        newToggle.addEventListener('click', function() {
            const html = document.documentElement;
            const isDark = html.classList.contains('dark');
            
            if (isDark) {
                html.classList.remove('dark');
                newToggle.textContent = '🌙';
                newToggle.innerHTML = '<span class="text-xl">🌙</span>';
                localStorage.setItem('darkMode', 'false');
            } else {
                html.classList.add('dark');
                newToggle.textContent = '☀️';
                newToggle.innerHTML = '<span class="text-xl">☀️</span>';
                localStorage.setItem('darkMode', 'true');
            }
        });
        
        // Load saved preference
        const savedMode = localStorage.getItem('darkMode');
        if (savedMode === 'true') {
            document.documentElement.classList.add('dark');
            newToggle.innerHTML = '<span class="text-xl">☀️</span>';
        } else {
            document.documentElement.classList.remove('dark');
            newToggle.innerHTML = '<span class="text-xl">🌙</span>';
        }
    }
}

async function checkCredibility() {
    const textarea = document.getElementById('news-text');
    const text = textarea.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (isAnalyzing) {
        return; // Prevent multiple simultaneous requests
    }
    
    isAnalyzing = true;
    updateAnalyzingUI(true);
    
    try {
        console.log('🔍 Analyzing text...');
        
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Analysis complete:', result);
        
        updateUI(result);
        updateHistory();
        
    } catch (error) {
        console.error('❌ Error analyzing text:', error);
        showError('Failed to analyze text. Please try again.');
    } finally {
        isAnalyzing = false;
        updateAnalyzingUI(false);
    }
}

async function analyzeUrl() {
    const urlInput = document.getElementById('news-url');
    const url = urlInput.value.trim();
    
    if (!url) {
        showError('Please enter a URL to analyze.');
        return;
    }
    
    // Basic URL validation
    try {
        new URL(url);
    } catch (e) {
        showError('Please enter a valid URL.');
        return;
    }
    
    if (isAnalyzing) {
        return; // Prevent multiple simultaneous requests
    }
    
    isAnalyzing = true;
    updateAnalyzingUI(true);
    
    try {
        console.log('🔗 Analyzing URL...');
        
        const response = await fetch('/analyze-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ URL analysis complete:', result);
        
        // Update UI with results
        updateUI(result);
        updateHistory();
        
    } catch (error) {
        console.error('❌ Error analyzing URL:', error);
        showError('Failed to analyze URL. Please try again.');
    } finally {
        isAnalyzing = false;
        updateAnalyzingUI(false);
    }
}



function updateUI(result) {
    console.log('🎨 Updating UI with result:', result);
    
    // Update status
    updateStatus(result.prediction, result.confidence);
    
    // Update confidence score
    updateConfidence(result.confidence);
    
    // Update online verification section
    updateOnlineVerification(result);
    
    // Update AI analysis result
    updateAIAnalysis(result);
    
    // Update source info (if available)
    updateSourceInfo(result);
    
    // Update URL-specific info (if analyzing URL)
    if (result.url) {
        updateUrlInfo(result);
    }
    
    // Update detailed explanation
    updateDetailedExplanation(result);
    
    console.log('✅ UI update complete');
}

function updateStatus(prediction, confidence) {
    const statusEl = document.getElementById('status');
    if (!statusEl) return;
    
    let statusClass, statusText, statusIcon;
    
    if (prediction === 'FAKE') {
        statusClass = 'bg-red-500/20 text-red-500';
        statusText = '🔴 Fake News';
        statusIcon = '🔴';
    } else if (prediction === 'TRUE') {
        statusClass = 'bg-green-500/20 text-green-500';
        statusText = '🟢 Real News';
        statusIcon = '🟢';
    } else {
        statusClass = 'bg-yellow-500/20 text-yellow-500';
        statusText = '⚠️ Suspicious';
        statusIcon = '⚠️';
    }
    
    statusEl.className = `px-3 py-1 rounded-full text-sm font-medium ${statusClass}`;
    statusEl.textContent = statusText;
}

function updateConfidence(confidence) {
    const confidenceEl = document.getElementById('confidence');
    const confidenceBarEl = document.getElementById('confidence-bar');
    
    if (confidenceEl) {
        confidenceEl.textContent = `${Math.round(confidence)}%`;
    }
    
    if (confidenceBarEl) {
        confidenceBarEl.style.width = `${Math.round(confidence)}%`;
    }
}

function updateOnlineVerification(result) {
    const onlineVerificationEl = document.getElementById('online-verification');
    if (!onlineVerificationEl) {
        console.log('❌ Online verification element not found');
        return;
    }
    
    // Check if NewsAPI found articles - use the correct structure
    if (result.news_api_results && result.news_api_results.found_online && result.news_api_results.all_matches) {
        const articles = result.news_api_results.all_matches;
        const bestMatch = result.news_api_results.best_match;
        
        // Show the online verification section
        onlineVerificationEl.classList.remove('hidden');
        
        // Update best match
        if (bestMatch) {
            const similarity = Math.round((bestMatch.similarity_score || 0) * 100);
            const publishedDate = bestMatch.publishedAt ? new Date(bestMatch.publishedAt).toLocaleDateString() : 'Unknown Date';
            
            const similarityEl = document.getElementById('best-match-similarity');
            const titleEl = document.getElementById('best-match-title');
            const sourceEl = document.getElementById('best-match-source');
            const publishedEl = document.getElementById('best-match-published');
            const urlEl = document.getElementById('best-match-url');
            
            if (similarityEl) similarityEl.textContent = `${similarity}% Similar`;
            if (titleEl) titleEl.textContent = bestMatch.title || 'Unknown Title';
            if (sourceEl) sourceEl.textContent = bestMatch.source?.name || 'Unknown Source';
            if (publishedEl) publishedEl.textContent = publishedDate;
            
            if (urlEl && bestMatch.url) {
                urlEl.href = bestMatch.url;
                urlEl.textContent = 'Read Original Article →';
            }
        }
        
        // Update other matches
        const otherMatchesCount = Math.max(0, articles.length - 1);
        const countEl = document.getElementById('other-matches-count');
        if (countEl) countEl.textContent = otherMatchesCount;
        
        const otherMatchesList = document.getElementById('other-matches-list');
        if (otherMatchesList && articles.length > 1) {
            let html = '';
            articles.slice(1, 5).forEach((article, index) => {
                const similarity = Math.round((article.similarity_score || 0) * 100);
                html += `
                    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
                        <div class="flex justify-between items-start mb-2">
                            <h6 class="font-semibold text-gray-800 dark:text-gray-200 text-sm leading-tight">${article.title || 'Unknown Title'}</h6>
                            <span class="bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full text-xs font-medium ml-2">${similarity}% similar</span>
                        </div>
                        <div class="text-xs text-gray-600 dark:text-gray-400">
                            <span class="font-medium">${article.source?.name || 'Unknown Source'}</span>
                            ${article.url ? `<span class="mx-2">•</span><a href="${article.url}" target="_blank" class="text-primary hover:underline">Read →</a>` : ''}
                        </div>
                    </div>
                `;
            });
            otherMatchesList.innerHTML = html;
        }
    } else {
        // Hide the online verification section
        onlineVerificationEl.classList.add('hidden');
    }
}

function updateAIAnalysis(result) {
    const aiAnalysisText = document.getElementById('ai-analysis-text');
    const aiPatternText = document.getElementById('ai-pattern-text');
    const individualResults = document.getElementById('individual-results');
    const modelResultsList = document.getElementById('model-results-list');
    
    if (!aiAnalysisText || !aiPatternText) {
        console.log('❌ AI analysis elements not found');
        return;
    }
    
    const prediction = result.prediction;
    const confidence = Math.round(result.confidence);
    
    // Update AI analysis text
    if (prediction === 'FAKE') {
        aiAnalysisText.textContent = `AI ANALYSIS: High probability of misinformation detected (confidence: ${confidence}%)`;
        aiPatternText.textContent = 'The text contains patterns commonly found in fabricated or satirical content.';
    } else if (prediction === 'TRUE') {
        aiAnalysisText.textContent = `AI ANALYSIS: High probability of credible content (confidence: ${confidence}%)`;
        aiPatternText.textContent = 'The text patterns are consistent with factual reporting.';
    } else {
        aiAnalysisText.textContent = `AI ANALYSIS: Inconclusive result (confidence: ${confidence}%)`;
        aiPatternText.textContent = 'Additional verification may be needed.';
    }
    
    // Update individual model results - handle the correct structure
    if (result.individual_results && Object.keys(result.individual_results).length > 0) {
        individualResults.classList.remove('hidden');
        
        let html = '';
        Object.entries(result.individual_results).forEach(([modelName, modelResult]) => {
            if (modelResult && typeof modelResult === 'object' && 'prediction' in modelResult) {
                const modelPred = modelResult.prediction;
                const modelConf = Math.round(modelResult.confidence || 0);
                html += `<div>• ${modelName}: ${modelPred} (${modelConf}%)</div>`;
            }
        });
        
        if (modelResultsList) {
            modelResultsList.innerHTML = html;
        }
    } else {
        individualResults.classList.add('hidden');
    }
}

function updateDetailedExplanation(result) {
    const explanationEl = document.getElementById('explanation-content');
    if (!explanationEl) return;
    
    let html = '';
    
    // Add ML explanation (formatted)
    if (result.explanation) {
        console.log('🤖 AI Explanation found:', result.explanation);
        
        // Format the explanation with proper line breaks and styling
        const formattedExplanation = result.explanation
            .replace(/\n/g, '<br>')
            .replace(/✅/g, '<span class="text-green-600 dark:text-green-400">✅</span>')
            .replace(/❌/g, '<span class="text-red-600 dark:text-red-400">❌</span>')
            .replace(/⚠️/g, '<span class="text-yellow-600 dark:text-yellow-400">⚠️</span>')
            .replace(/📰/g, '<span class="text-blue-600 dark:text-blue-400">📰</span>')
            .replace(/🏢/g, '<span class="text-purple-600 dark:text-purple-400">🏢</span>')
            .replace(/📅/g, '<span class="text-indigo-600 dark:text-indigo-400">📅</span>')
            .replace(/🎯/g, '<span class="text-pink-600 dark:text-pink-400">🎯</span>')
            .replace(/🔗/g, '<span class="text-cyan-600 dark:text-cyan-400">🔗</span>')
            .replace(/📚/g, '<span class="text-orange-600 dark:text-orange-400">📚</span>')
            .replace(/🤖/g, '<span class="text-blue-600 dark:text-blue-400">🤖</span>')
            .replace(/📊/g, '<span class="text-purple-600 dark:text-purple-400">📊</span>');
        
        html += `<div class="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">${formattedExplanation}</div>`;
    } else {
        console.log('⚠️ No AI explanation found in result:', result);
        
        // Fallback explanation
        const prediction = result.prediction;
        const confidence = Math.round(result.confidence);
        
        if (prediction === 'FAKE') {
            html += `<p class="text-sm text-red-600 dark:text-red-400">❌ High probability of misinformation (confidence: ${confidence}%). Patterns detected are commonly found in fabricated or satirical content.</p>`;
        } else if (prediction === 'TRUE') {
            html += `<p class="text-sm text-green-600 dark:text-green-400">✅ High probability of credible content (confidence: ${confidence}%). Text patterns are consistent with factual reporting.</p>`;
        } else {
            html += `<p class="text-sm text-yellow-600 dark:text-yellow-400">⚠️ Analysis is inconclusive (confidence: ${confidence}%). Additional verification may be needed.</p>`;
        }
    }
    
    explanationEl.innerHTML = html;
}

function updateSourceInfo(result) {
    // This would be populated if we had source information
    // For now, we'll show generic info
    const sourceEl = document.getElementById('source');
    const publishedEl = document.getElementById('published');
    const urlEl = document.getElementById('url');
    
    if (sourceEl) sourceEl.textContent = 'User Input';
    if (publishedEl) publishedEl.textContent = new Date().toLocaleDateString();
    if (urlEl) {
        urlEl.textContent = 'N/A';
        urlEl.href = '#';
    }
}

function updateUrlInfo(result) {
    // Update source info with URL analysis results
    const sourceEl = document.getElementById('source');
    const publishedEl = document.getElementById('published');
    const urlEl = document.getElementById('url');
    
    if (sourceEl && result.article_source) {
        sourceEl.textContent = result.article_source;
    }
    
    if (publishedEl) {
        publishedEl.textContent = new Date().toLocaleDateString();
    }
    
    if (urlEl && result.url) {
        urlEl.textContent = 'View Original';
        urlEl.href = result.url;
        urlEl.target = '_blank';
    }
    
    // Show article title if available
    if (result.article_title && result.article_title !== 'Unknown Title') {
        // You could add a title display element here if needed
        console.log('📰 Article Title:', result.article_title);
    }
}

function updateAnalyzingUI(analyzing) {
    const checkBtn = document.getElementById('check-credibility-btn');
    const statusEl = document.getElementById('status');
    
    if (analyzing) {
        if (checkBtn) {
            checkBtn.textContent = '🔄 Analyzing...';
            checkBtn.disabled = true;
        }
        if (statusEl) {
            statusEl.textContent = '⏳ Analyzing...';
            statusEl.className = 'px-3 py-1 rounded-full text-sm font-medium bg-gray-500/20 text-gray-500';
        }
    } else {
        if (checkBtn) {
            checkBtn.textContent = 'Check Credibility';
            checkBtn.disabled = false;
        }
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        if (response.ok) {
            const data = await response.json();
            currentHistory = data.history || [];
            displayHistory();
        }
    } catch (error) {
        console.error('❌ Error loading history:', error);
    }
}

function updateHistory() {
    // Update the history display
    displayHistory();
}

function displayHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    if (currentHistory.length === 0) {
        historyList.innerHTML = `
            <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                <p>No analysis history yet.</p>
                <p class="text-sm mt-2">Start by analyzing some text above!</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    currentHistory.slice(-10).reverse().forEach((item, index) => {
        const prediction = item.prediction;
        const confidence = Math.round(item.confidence);
        const timestamp = new Date(item.timestamp).toLocaleString();
        const text = item.text || 'Unknown text';
        
        let statusClass, statusText, statusIcon;
        
        if (prediction === 'FAKE') {
            statusClass = 'bg-red-500/20 text-red-500';
            statusText = '🔴 Fake';
            statusIcon = '🔴';
        } else if (prediction === 'TRUE') {
            statusClass = 'bg-green-500/20 text-green-500';
            statusText = '🟢 Real';
            statusIcon = '🟢';
        } else {
            statusClass = 'bg-yellow-500/20 text-yellow-500';
            statusText = '⚠️ Suspicious';
            statusIcon = '⚠️';
        }
        
        html += `
            <div class="p-4 rounded-md border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-background-dark/20 transition cursor-pointer" onclick="loadFromHistory('${text.replace(/'/g, "\\'")}')">
                <div class="flex justify-between items-start">
                    <p class="font-semibold text-gray-800 dark:text-gray-200 text-sm">${text.length > 60 ? text.substring(0, 60) + '...' : text}</p>
                    <span class="px-2 py-0.5 rounded-full text-xs font-medium ${statusClass} ml-2 whitespace-nowrap">${statusText}</span>
                </div>
                <div class="text-sm text-gray-500 dark:text-gray-400 mt-2 flex justify-between items-center">
                    <span>Confidence: ${confidence}%</span>
                    <span>${timestamp}</span>
                </div>
            </div>
        `;
    });
    
    historyList.innerHTML = html;
}

function loadFromHistory(text) {
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.value = text;
        textarea.scrollTop = 0;
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all analysis history?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            currentHistory = [];
            displayHistory();
            showSuccess('History cleared successfully.');
        } else {
            throw new Error('Failed to clear history');
        }
    } catch (error) {
        console.error('❌ Error clearing history:', error);
        showError('Failed to clear history. Please try again.');
    }
}

function showError(message) {
    // Simple error display - could be enhanced with toast notifications
    console.error('❌ Error:', message);
    alert('Error: ' + message);
}

function showSuccess(message) {
    // Simple success display - could be enhanced with toast notifications
    console.log('✅ Success:', message);
    // Could add a toast notification here
}

// Utility functions
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
