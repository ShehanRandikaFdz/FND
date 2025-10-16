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
    
    // Fetch news button
    const fetchBtn = document.getElementById('fetch-news-btn');
    if (fetchBtn) {
        fetchBtn.addEventListener('click', fetchLatestNews);
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
        toggle.addEventListener('click', function() {
            const html = document.documentElement;
            const isDark = html.classList.contains('dark');
            
            if (isDark) {
                html.classList.remove('dark');
                toggle.innerHTML = '<span class="text-xl">🌙</span>';
                localStorage.setItem('darkMode', 'false');
            } else {
                html.classList.add('dark');
                toggle.innerHTML = '<span class="text-xl">☀️</span>';
                localStorage.setItem('darkMode', 'true');
            }
        });
        
        // Load saved preference
        const savedMode = localStorage.getItem('darkMode');
        if (savedMode === 'true') {
            document.documentElement.classList.add('dark');
            toggle.innerHTML = '<span class="text-xl">☀️</span>';
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
            const errorData = await response.json();
            console.error('❌ Server error:', errorData);
            
            // Handle specific error cases
            if (response.status === 401) {
                showError('Text analysis requires authentication. Please login or register first.');
            } else if (response.status === 402) {
                showError('Analysis limit exceeded. Please upgrade your subscription or wait for next month.');
            } else if (errorData.error) {
                showError(errorData.error);
            } else {
                showError('Failed to analyze text. Please try again.');
            }
            return;
        }
        
        const result = await response.json();
        console.log('✅ Analysis complete:', result);
        
        // Add to history
        const historyItem = {
            text: text,
            prediction: result.prediction,
            confidence: result.confidence,
            timestamp: new Date().toISOString(),
            explanation: result.explanation
        };
        currentHistory.push(historyItem);
        
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
            const errorData = await response.json();
            console.error('❌ Server error:', errorData);
            
            // Handle specific error cases
            if (response.status === 401) {
                showError('URL analysis requires authentication. Please login or register first.');
            } else if (response.status === 402) {
                showError('URL analysis requires Professional plan or higher. Please upgrade your subscription.');
            } else if (errorData.error) {
                showError(errorData.error);
            } else {
                showError('Failed to analyze URL. Please try again.');
            }
            return;
        }
        
        const result = await response.json();
        console.log('✅ URL analysis complete:', result);
        
        // Add to history
        const historyItem = {
            text: result.article_text || result.text || url,
            prediction: result.prediction,
            confidence: result.confidence,
            timestamp: new Date().toISOString(),
            explanation: result.explanation,
            url: url,
            article_title: result.article_title,
            article_source: result.article_source
        };
        currentHistory.push(historyItem);
        
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

async function fetchLatestNews() {
    const fetchBtn = document.getElementById('fetch-news-btn');
    const originalText = fetchBtn.textContent;
    
    try {
        fetchBtn.textContent = '📡 Fetching...';
        fetchBtn.disabled = true;
        
        console.log('📰 Fetching latest news...');
        
        const response = await fetch('/fetch-news-public', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                country: 'us',
                category: 'general',
                page_size: 5
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ News fetched:', data);
        
        displayLatestNews(data.articles || []);
        
    } catch (error) {
        console.error('❌ Error fetching news:', error);
        showError('Failed to fetch latest news. Please try again.');
    } finally {
        fetchBtn.textContent = originalText;
        fetchBtn.disabled = false;
    }
}

function displayLatestNews(articles) {
    if (!articles || articles.length === 0) {
        showError('No news articles found.');
        return;
    }
    
    // Display first article in textarea
    const firstArticle = articles[0];
    const textarea = document.getElementById('news-text');
    
    let newsText = firstArticle.title || '';
    if (firstArticle.description) {
        newsText += '\n\n' + firstArticle.description;
    }
    if (firstArticle.content) {
        newsText += '\n\n' + firstArticle.content;
    }
    
    textarea.value = newsText;
    
    // Show success message
    showSuccess(`Fetched ${articles.length} news articles. First article loaded for analysis.`);
}

function updateUI(result) {
    console.log('🎨 Updating UI with result:', result);
    
    // Update status
    updateStatus(result.prediction, result.confidence);
    
    // Update confidence score
    updateConfidence(result.confidence);
    
    // Update explanation
    updateExplanation(result);
    
    // Update source info (if available)
    updateSourceInfo(result);
    
    // Update URL-specific info (if analyzing URL)
    if (result.url) {
        updateUrlInfo(result);
    }
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

function updateExplanation(result) {
    const explanationEl = document.getElementById('explanation-content');
    if (!explanationEl) return;
    
    let html = '';
    
    // Check if NewsAPI found articles
    if (result.news_api_results && result.news_api_results.found_online && result.news_api_results.all_matches) {
        const articles = result.news_api_results.all_matches;
        const bestMatch = result.news_api_results.best_match;
        
        html += '<div class="mb-6">';
        html += '<div class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">';
        html += '<h4 class="font-semibold text-green-700 dark:text-green-300 mb-3 flex items-center">';
        html += '<span class="mr-2">✅</span> Online Verification Found';
        html += '</h4>';
        
        // Best match article
        if (bestMatch) {
            const similarity = Math.round((bestMatch.similarity_score || 0) * 100);
            const publishedDate = bestMatch.publishedAt ? new Date(bestMatch.publishedAt).toLocaleDateString() : 'Unknown Date';
            
            html += '<div class="bg-white dark:bg-gray-800 rounded-lg p-4 mb-3 border border-green-200 dark:border-green-700">';
            html += '<div class="flex items-start justify-between mb-2">';
            html += '<h5 class="font-semibold text-gray-900 dark:text-gray-100 text-sm leading-tight">📰 Best Match</h5>';
            html += `<span class="bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-200 px-2 py-1 rounded-full text-xs font-medium">${similarity}% Similar</span>`;
            html += '</div>';
            html += `<p class="font-medium text-gray-800 dark:text-gray-200 mb-2">${bestMatch.title || 'Unknown Title'}</p>`;
            html += '<div class="text-xs text-gray-600 dark:text-gray-400 space-y-1">';
            html += `<p><strong>🏢 Source:</strong> ${bestMatch.source?.name || 'Unknown Source'}</p>`;
            html += `<p><strong>📅 Published:</strong> ${publishedDate}</p>`;
            if (bestMatch.url) {
                html += `<p><strong>🔗 Link:</strong> <a href="${bestMatch.url}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline">Read Original Article →</a></p>`;
            }
            html += '</div>';
            html += '</div>';
        }
        
        // Additional articles
        if (articles.length > 1) {
            html += '<div class="mt-3">';
            html += `<h6 class="font-medium text-gray-700 dark:text-gray-300 mb-2">📚 Other Matches (${articles.length - 1} more):</h6>`;
            html += '<div class="space-y-2">';
            
            articles.slice(1, 4).forEach((article, index) => {
                const similarity = Math.round((article.similarity_score || 0) * 100);
                html += `
                    <div class="bg-gray-50 dark:bg-gray-700 rounded p-3 border-l-4 border-blue-400">
                        <p class="font-medium text-sm text-gray-800 dark:text-gray-200 mb-1">${article.title || 'Unknown Title'}</p>
                        <div class="text-xs text-gray-600 dark:text-gray-400">
                            <span class="font-medium">${article.source?.name || 'Unknown Source'}</span>
                            <span class="mx-2">•</span>
                            <span>${similarity}% similar</span>
                            ${article.url ? `<span class="mx-2">•</span><a href="${article.url}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline">Read →</a>` : ''}
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            html += '</div>';
        }
        
        html += '</div>';
        html += '</div>';
    } else {
        // Show warning if no online sources found
        html += `
            <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-4">
                <div class="flex items-center">
                    <span class="mr-2">⚠️</span>
                    <p class="text-yellow-800 dark:text-yellow-200 text-sm font-medium">
                        No matching articles found in trusted online sources for verification.
                    </p>
                </div>
            </div>
        `;
    }
    
    // Enhanced AI Explanation with detailed analysis
    html += '<div class="space-y-4">';
    
    // News Content Information
    html += '<div class="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/20 dark:to-slate-900/20 border border-gray-200 dark:border-gray-800 rounded-lg p-4">';
    html += '<h5 class="font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">';
    html += '<span class="mr-2">📰</span> News Content Analysis';
    html += '</h5>';
    
    // Show analyzed text preview
    const analyzedText = result.text || result.article_text || '';
    const textPreview = analyzedText.length > 200 ? analyzedText.substring(0, 200) + '...' : analyzedText;
    html += `<div class="bg-white dark:bg-gray-800 rounded-lg p-3 mb-3 border border-gray-200 dark:border-gray-700">`;
    html += `<h6 class="font-medium text-gray-600 dark:text-gray-400 mb-2">📝 Analyzed Content:</h6>`;
    html += `<p class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">"${textPreview}"</p>`;
    html += '</div>';
    
    // Content statistics
    const wordCount = analyzedText.split(' ').length;
    const sentenceCount = analyzedText.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const avgWordsPerSentence = sentenceCount > 0 ? Math.round(wordCount / sentenceCount) : 0;
    
    html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">';
    html += `<div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-2 text-center">`;
    html += `<div class="font-semibold text-blue-700 dark:text-blue-300">${wordCount}</div>`;
    html += `<div class="text-xs text-blue-600 dark:text-blue-400">Words</div>`;
    html += '</div>';
    
    html += `<div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-2 text-center">`;
    html += `<div class="font-semibold text-green-700 dark:text-green-300">${sentenceCount}</div>`;
    html += `<div class="text-xs text-green-600 dark:text-green-400">Sentences</div>`;
    html += '</div>';
    
    html += `<div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-2 text-center">`;
    html += `<div class="font-semibold text-purple-700 dark:text-purple-300">${avgWordsPerSentence}</div>`;
    html += `<div class="text-xs text-purple-600 dark:text-purple-400">Avg/Sentence</div>`;
    html += '</div>';
    
    html += `<div class="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-2 text-center">`;
    html += `<div class="font-semibold text-orange-700 dark:text-orange-300">${Math.round(analyzedText.length / 5)}</div>`;
    html += `<div class="text-xs text-orange-600 dark:text-orange-400">Characters</div>`;
    html += '</div>';
    
    html += '</div>';
    html += '</div>';
    
    // Source Information (if available)
    if (result.url || result.article_title || result.article_source) {
        html += '<div class="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-indigo-700 dark:text-indigo-300 mb-3 flex items-center">';
        html += '<span class="mr-2">🔗</span> Source Information';
        html += '</h5>';
        
        if (result.article_title) {
            html += `<div class="mb-2">`;
            html += `<h6 class="font-medium text-gray-600 dark:text-gray-400 text-sm">📰 Article Title:</h6>`;
            html += `<p class="text-gray-800 dark:text-gray-200 text-sm">${result.article_title}</p>`;
            html += '</div>';
        }
        
        if (result.article_source) {
            html += `<div class="mb-2">`;
            html += `<h6 class="font-medium text-gray-600 dark:text-gray-400 text-sm">🏢 Source:</h6>`;
            html += `<p class="text-gray-800 dark:text-gray-200 text-sm">${result.article_source}</p>`;
            html += '</div>';
        }
        
        if (result.url) {
            html += `<div class="mb-2">`;
            html += `<h6 class="font-medium text-gray-600 dark:text-gray-400 text-sm">🌐 URL:</h6>`;
            html += `<a href="${result.url}" target="_blank" class="text-blue-600 dark:text-blue-400 hover:underline text-sm break-all">${result.url}</a>`;
            html += '</div>';
        }
        
        html += '</div>';
    }
    
    // Main AI Analysis Result
    if (result.explanation) {
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
            .replace(/📚/g, '<span class="text-orange-600 dark:text-orange-400">📚</span>');
        
        html += '<div class="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-blue-700 dark:text-blue-300 mb-3 flex items-center">';
        html += '<span class="mr-2">🤖</span> AI Analysis Summary';
        html += '</h5>';
        html += `<div class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">${formattedExplanation}</div>`;
        html += '</div>';
    }
    
    // Detailed Analysis Breakdown
    if (result.individual_results && result.individual_results.enhanced_heuristic) {
        const analysis = result.individual_results.enhanced_heuristic;
        const details = analysis.analysis_details;
        
        html += '<div class="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-green-700 dark:text-green-300 mb-3 flex items-center">';
        html += '<span class="mr-2">🔍</span> Detailed Analysis Breakdown';
        html += '</h5>';
        
        html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">';
        
        // Fake News Indicators
        if (details.fake_indicators_found > 0) {
            html += '<div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">';
            html += '<h6 class="font-medium text-red-700 dark:text-red-300 mb-2 flex items-center">';
            html += '<span class="mr-1">🚨</span> Fake News Indicators';
            html += '</h6>';
            html += `<p class="text-red-600 dark:text-red-400">Found ${details.fake_indicators_found} suspicious patterns</p>`;
            html += '<p class="text-xs text-red-500 dark:text-red-400 mt-1">Look for: clickbait, emotional language, unverified claims</p>';
            html += '</div>';
        }
        
        // Real News Indicators
        if (details.real_indicators_found > 0) {
            html += '<div class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3">';
            html += '<h6 class="font-medium text-green-700 dark:text-green-300 mb-2 flex items-center">';
            html += '<span class="mr-1">✅</span> Credible Indicators';
            html += '</h6>';
            html += `<p class="text-green-600 dark:text-green-400">Found ${details.real_indicators_found} credible patterns</p>`;
            html += '<p class="text-xs text-green-500 dark:text-green-400 mt-1">Includes: official sources, research citations, expert quotes</p>';
            html += '</div>';
        }
        
        // Emotional Manipulation
        if (details.emotional_indicators_found > 0) {
            html += '<div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">';
            html += '<h6 class="font-medium text-yellow-700 dark:text-yellow-300 mb-2 flex items-center">';
            html += '<span class="mr-1">⚠️</span> Emotional Manipulation';
            html += '</h6>';
            html += `<p class="text-yellow-600 dark:text-yellow-400">Found ${details.emotional_indicators_found} emotional triggers</p>`;
            html += '<p class="text-xs text-yellow-500 dark:text-yellow-400 mt-1">Watch for: urgent language, fear tactics, outrage</p>';
            html += '</div>';
        }
        
        // Text Quality Analysis
        html += '<div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">';
        html += '<h6 class="font-medium text-blue-700 dark:text-blue-300 mb-2 flex items-center">';
        html += '<span class="mr-1">📝</span> Text Quality';
        html += '</h6>';
        html += `<p class="text-blue-600 dark:text-blue-400">Length: ${details.text_length} words</p>`;
        html += `<p class="text-blue-600 dark:text-blue-400">Avg. sentence: ${details.avg_sentence_length} words</p>`;
        if (details.caps_ratio > 10) {
            html += `<p class="text-orange-600 dark:text-orange-400">⚠️ High caps usage: ${details.caps_ratio}%</p>`;
        }
        if (details.exclamation_ratio > 1) {
            html += `<p class="text-orange-600 dark:text-orange-400">⚠️ Excessive exclamations: ${details.exclamation_ratio}%</p>`;
        }
        html += '</div>';
        
        html += '</div>';
        html += '</div>';
    }
    
    // Confidence Level Explanation
    const confidence = result.confidence || 0;
    let confidenceColor, confidenceText, confidenceIcon;
    
    if (confidence >= 80) {
        confidenceColor = 'text-green-600 dark:text-green-400';
        confidenceText = 'High Confidence';
        confidenceIcon = '🎯';
    } else if (confidence >= 60) {
        confidenceColor = 'text-yellow-600 dark:text-yellow-400';
        confidenceText = 'Medium Confidence';
        confidenceIcon = '⚖️';
    } else {
        confidenceColor = 'text-red-600 dark:text-red-400';
        confidenceText = 'Low Confidence';
        confidenceIcon = '⚠️';
    }
    
    html += '<div class="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">';
    html += '<h5 class="font-semibold text-purple-700 dark:text-purple-300 mb-3 flex items-center">';
    html += `<span class="mr-2">${confidenceIcon}</span> Confidence Level: ${confidence}%`;
    html += '</h5>';
    html += `<p class="text-sm text-gray-600 dark:text-gray-400">${confidenceText} - ${confidence >= 80 ? 'The AI is very certain about this analysis.' : confidence >= 60 ? 'The AI is moderately confident in this analysis.' : 'The AI has some uncertainty about this analysis.'}</p>`;
    html += '</div>';
    
    html += '</div>';
    
    if (!result.explanation) {
        // Fallback explanation
        const prediction = result.prediction;
        const confidence = Math.round(result.confidence);
        
        html += '<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">';
        html += '<h5 class="font-semibold text-gray-700 dark:text-gray-300 mb-2">🤖 AI Analysis Result:</h5>';
        
        if (prediction === 'FAKE') {
            html += `<p class="text-sm text-red-600 dark:text-red-400">❌ High probability of misinformation (confidence: ${confidence}%). Patterns detected are commonly found in fabricated or satirical content.</p>`;
        } else if (prediction === 'TRUE') {
            html += `<p class="text-sm text-green-600 dark:text-green-400">✅ High probability of credible content (confidence: ${confidence}%). Text patterns are consistent with factual reporting.</p>`;
        } else {
            html += `<p class="text-sm text-yellow-600 dark:text-yellow-400">⚠️ Analysis is inconclusive (confidence: ${confidence}%). Additional verification may be needed.</p>`;
        }
        
        html += '</div>';
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
        
        const displayText = item.article_title || text;
        const sourceInfo = item.article_source ? ` (${item.article_source})` : '';
        const urlInfo = item.url ? `<br><span class="text-xs text-blue-500">🔗 ${item.url}</span>` : '';
        
        html += `
            <div class="p-4 rounded-md border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-background-dark/20 transition cursor-pointer" onclick="loadFromHistory('${text.replace(/'/g, "\\'")}')">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <p class="font-semibold text-gray-800 dark:text-gray-200 text-sm">${displayText.length > 60 ? displayText.substring(0, 60) + '...' : displayText}</p>
                        ${sourceInfo ? `<p class="text-xs text-gray-500 dark:text-gray-400 mt-1">${sourceInfo}</p>` : ''}
                        ${urlInfo}
                    </div>
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
